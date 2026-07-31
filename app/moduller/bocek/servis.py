"""Böcek teşhis modelinin çalıştırılması — saf mantık, HTTP bilmez.

NEDEN AYRI AKIŞ?
    Bu model 416x416 MAKRO böcek fotoğraflarıyla eğitildi; kutu alanı
    medyanı karenin %14,8'i. Hiyerarşik boru hattı ise saha görüntüsünden
    kırpılmış yaprak/meyve parçası verir — orada bir zararlı kırpıntının
    %1'inden azını kaplar. İki alan birbirine benzemez.

    Bu yüzden kütükte `tetik: []` ile durur ve ROI akışına HİÇ girmez
    (bkz. configs/urunler/<urun>/modeller.yaml, tests/test_tekil_model.py).
    Buradan açıkça çağrılır.

KAPALI KÜME SORUNU — ARAYÜZÜN ASIL İŞİ
    Model yalnızca 6 tür bilir ve "bilmiyorum" diyemez: gördüğü her böceği
    bu 6'dan birine sokar. Kullanıcı yaprak biti fotoğrafı çekerse
    "Toprak Larvası %70" cevabı alabilir ve buna güvenirse yanlış mücadele
    yapar.

    Bu yüzden tek bir cevap DÖNMÜYORUZ: ilk 3 aday güvenleriyle birlikte
    verilir ve aralarındaki fark küçükse "kararsız" işaretlenir. Kullanıcı
    modelin ne kadar emin olduğunu görerek karar verir.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

MODEL_ADI = 'bocek_teshis'

# Bu değerin altındaki hiçbir aday gösterilmez — model her karede bir şey
# bulmaya çalışır, düşük güvenli çöp tespitler kullanıcıyı yanıltır.
EN_DUSUK_GUVEN = 0.20

# İlk iki aday birbirine bu kadar yakınsa model KARARSIZDIR. Tek cevap
# göstermek burada yanıltıcı olur.
KARARSIZLIK_FARKI = 0.15


@dataclass
class Aday:
    ad: str                  # eğitimdeki ad (Mole Cricket)
    guven: float

    @property
    def yuzde(self) -> int:
        return int(round(self.guven * 100))


@dataclass
class Sonuc:
    adaylar: List[Aday] = field(default_factory=list)
    kararsiz: bool = False
    hata: str = ''

    @property
    def bulundu(self) -> bool:
        return bool(self.adaylar)

    @property
    def en_iyi(self) -> Optional[Aday]:
        return self.adaylar[0] if self.adaylar else None


def hazir(urun=None) -> bool:
    """Model dosyası kurulu mu?"""
    from app import modeller
    t = modeller.tanim(MODEL_ADI, urun)
    return bool(t and t.aktif and t.var)


def taniyabildikleri(urun=None) -> List[str]:
    """Modelin bildiği türler — arayüz bunu AÇIKÇA göstermeli.

    Kapalı küme olduğu için kullanıcı listeyi görmeden fotoğraf çekerse,
    listede olmayan bir böcek için kendinden emin ama yanlış cevap alır.
    """
    from app import modeller
    t = modeller.tanim(MODEL_ADI, urun)
    return list(t.siniflar) if t else []


def tani(goruntu, urun=None, imgsz: int = 416) -> Sonuc:
    """Bir kareyi modelden geçirip aday listesi döner.

    goruntu : BGR numpy dizisi (cv2.imread / imdecode çıktısı)
    """
    from app import modeller

    model = modeller.yukle(MODEL_ADI, urun)
    if model is None:
        return Sonuc(hata='Model kurulu değil.')

    try:
        r = model(goruntu, conf=EN_DUSUK_GUVEN, imgsz=imgsz, verbose=False)[0]
    except Exception as e:
        logger.warning(f'Böcek teşhisi başarısız: {e}')
        return Sonuc(hata=f'Görüntü işlenemedi ({type(e).__name__}).')

    # Aynı tür birden çok kutuda çıkabilir (birkaç birey); tür başına EN
    # YÜKSEK güven alınır. Kutu sayısı burada anlam taşımaz — soru "bu ne",
    # "kaç tane" değil.
    en_iyi = {}
    adlar = r.names
    for b in r.boxes:
        cid = int(b.cls[0])
        ad = adlar.get(cid, str(cid)) if isinstance(adlar, dict) else adlar[cid]
        g = float(b.conf[0])
        if g > en_iyi.get(ad, 0.0):
            en_iyi[ad] = g

    adaylar = [Aday(ad, g) for ad, g in
               sorted(en_iyi.items(), key=lambda x: -x[1])][:3]

    kararsiz = (len(adaylar) >= 2
                and adaylar[0].guven - adaylar[1].guven < KARARSIZLIK_FARKI)
    return Sonuc(adaylar=adaylar, kararsiz=kararsiz)


def oneri(sinif_adi: str, urun=None) -> dict:
    """Tür için mücadele önerisi.

    Organ verilmez: makro fotoğrafta böcek bitkinin neresinde bilinmez.
    tedavi.coz() bu durumda ORTAK metni döner (organ blokları kullanılmaz).
    """
    from app import tedavi
    return tedavi.coz(tedavi.yukle(urun), sinif_adi)
