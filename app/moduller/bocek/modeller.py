"""Böcek teşhis modülünün kendi tablosu.

NEDEN ÇEKİRDEK `Analiz` TABLOSUNA YAZILMIYOR?
    Bu akış bitki analizi değil, TÜR SORGUSUDUR. Analiz tablosuna yazılsaydı:
      - "şu serada 12 tespit" sayısı anlamını yitirirdi (bir kısmı hastalık
        tespiti değil, avuç içinde fotoğraflanmış bir böcek olurdu),
      - yaygınlık haritası bitkiye ait olmayan noktalarla dolardı,
      - hastalık istatistikleri bozulurdu.

    Ayrı tablo ayrıca modülü taşınabilir tutar: klasör silindiğinde çekirdek
    şema etkilenmez (konum modülü de aynı deseni kullanır).

NEDEN DOĞRULAMA ALANI VAR?
    Model KAPALI KÜMEDİR — yalnızca 6 tür bilir ve "bilmiyorum" diyemez.
    Kullanıcının "bu doğru / bu yanlış / bu listede yok" demesi iki işe yarar:
      1. Kayıt düzelir; geçmişte yanlış tür yazılı kalmaz.
      2. Modelin SAHADAKİ isabetini ölçebiliriz. Doğrulama olmadan "model iyi
         mi" sorusunun cevabı yoktur; eğitim metriği laboratuvar koşuludur.
    'listede_yok' işaretlenen kayıtlar ileride toplanacak saha zararlı
    verisinin çekirdeğidir.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.database import Base

# Doğrulama durumları — arayüz bunların dışında bir değer yazmamalı.
DOGRULAMA = {
    '': 'doğrulanmadı',
    'dogru': 'doğru',
    'yanlis': 'yanlış',
    'listede_yok': 'listede yok',
}


class BocekKaydi(Base):
    __tablename__ = 'bocek_kayitlari'

    id = Column(Integer, primary_key=True)
    zaman = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    gorsel = Column(String(200), default='')     # storage altındaki göreli yol

    # Modelin cevabı
    tur = Column(String(80), default='', index=True)   # boş = böcek bulunamadı
    guven = Column(Float, default=0.0)
    kararsiz = Column(Boolean, default=False)
    adaylar_json = Column(Text, default='')            # [{"ad":..., "guven":...}]

    # Kullanıcının doğrulaması
    dogrulama = Column(String(20), default='', index=True)
    dogru_tur = Column(String(80), default='')   # 'yanlis' ise gerçek tür
    notlar = Column(Text, default='')

    # Nerede bulundu (isteğe bağlı, serbest metin — GPS zorlamıyoruz)
    yer = Column(String(120), default='')

    @property
    def adaylar(self) -> list:
        if not self.adaylar_json:
            return []
        try:
            return json.loads(self.adaylar_json)
        except (ValueError, TypeError):
            return []

    @property
    def dogrulama_etiketi(self) -> str:
        return DOGRULAMA.get(self.dogrulama or '', 'doğrulanmadı')

    @property
    def gecerli_tur(self) -> str:
        """Kayda GEÇERLİ sayılan tür: kullanıcı düzelttiyse onunki.

        Geçmiş listesi ve istatistik bunu kullanır; modelin ilk cevabı
        `tur` alanında saklı kalır ki isabet ölçülebilsin.
        """
        if self.dogrulama == 'yanlis' and self.dogru_tur:
            return self.dogru_tur
        if self.dogrulama == 'listede_yok':
            return ''
        return self.tur


def tablolar_olustur(engine):
    Base.metadata.create_all(engine, tables=[BocekKaydi.__table__])


def isabet(kayitlar) -> dict:
    """Sahadaki isabet — yalnızca DOĞRULANMIŞ kayıtlar üzerinden.

    Eğitim metriği (mAP) laboratuvar koşuludur: aynı dağılımdan ayrılmış
    doğrulama setinde ölçülür. Buradaki sayı gerçek kullanımı gösterir ve
    ikisi ciddi şekilde farklı olabilir.
    """
    dogru = sum(1 for k in kayitlar if k.dogrulama == 'dogru')
    yanlis = sum(1 for k in kayitlar if k.dogrulama == 'yanlis')
    disinda = sum(1 for k in kayitlar if k.dogrulama == 'listede_yok')
    degerlendirilen = dogru + yanlis
    return {
        'toplam': len(kayitlar),
        'dogru': dogru,
        'yanlis': yanlis,
        'listede_yok': disinda,
        'dogrulanmamis': len(kayitlar) - degerlendirilen - disinda,
        'oran': (100 * dogru / degerlendirilen) if degerlendirilen else None,
    }
