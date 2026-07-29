"""Arayüz dili ve sınıf adlarının yerelleştirilmesi.

NEDEN GEREKLİ?
    Model sınıf adları İngilizce üretir (`Gray Mold`, `strawberry_unripe`) —
    bunlar EĞİTİMDEKİ adlardır ve DEĞİŞTİRİLEMEZ: etiket dosyaları, dışa
    aktarım ve yeniden eğitim bu adlara bağlıdır. Sahada çalışan kişi ise
    "Kurşuni Küf" görmek ister.

    Bu modül ikisini ayırır: veritabanı ve dışa aktarım hep İngilizce adı
    saklar, EKRANDA seçilen dile göre çevirisi gösterilir. Böylece dil
    değiştirmek eğitim verisini hiç etkilemez.

NASIL SEÇİLİR?
    Kullanıcı üstteki seçimden dili değiştirir → `dil` çerezine yazılır →
    her istekte ara katman okuyup bağlam değişkenine koyar → şablonlardaki
    `|sinif` süzgeci o dile göre yazar.
"""

import os
from contextvars import ContextVar
from pathlib import Path

import yaml

DILLER = {'tr': '🇹🇷 Türkçe', 'en': '🇬🇧 English'}
VARSAYILAN = os.environ.get('DIL', 'tr') if os.environ.get('DIL') in DILLER else 'tr'
CEREZ = 'dil'

# İstek başına dil. Ara katman doldurur; şablon süzgeçleri okur.
# (FastAPI eşzamanlı rotaları iş parçacığı havuzunda çalıştırırken bağlamı
#  kopyalar, bu yüzden ContextVar burada güvenlidir.)
_aktif: ContextVar = ContextVar('aktif_dil', default=VARSAYILAN)


def _adlari_yukle() -> dict:
    """Sınıf adı çevirileri: {'Gray Mold': {'tr': 'Kurşuni Küf', ...}}.

    Kaynak configs/sinif_adlari.yaml; yoksa tedavi dosyasındaki `ad` alanına
    düşülür (orada zaten Türkçe karşılıklar var).
    """
    from app import urunler
    for ad in ('siniflar.yaml', 'sinif_adlari.yaml'):     # kütük → eski dosya
        p = urunler.yapilandirma(None, ad)
        if p.exists():
            return yaml.safe_load(p.read_text(encoding='utf-8')) or {}

    kok = Path(__file__).resolve().parent.parent / 'configs'
    tedavi = urunler.yapilandirma(None, 'tedavi_onerileri.yaml')
    if tedavi.exists():
        veri = yaml.safe_load(tedavi.read_text(encoding='utf-8')) or {}
        return {ad: {'tr': (bilgi or {}).get('ad', ad)} for ad, bilgi in veri.items()}
    return {}


ADLAR = _adlari_yukle()


def ayarla(kod: str) -> str:
    kod = kod if kod in DILLER else VARSAYILAN
    _aktif.set(kod)
    return kod


def aktif() -> str:
    return _aktif.get()


def istekten_oku(request) -> str:
    """Çerez → dil kodu. Çerez yoksa varsayılan."""
    try:
        kod = request.cookies.get(CEREZ, '')
    except Exception:
        kod = ''
    return kod if kod in DILLER else VARSAYILAN


def _okunakli(ad: str) -> str:
    """`strawberry_unripe` → `Strawberry Unripe` (İngilizce görünümü düzeltir)."""
    return ad.replace('_', ' ').title() if '_' in ad else ad


def sinif_adi(ad: str, kod: str = None) -> str:
    """Model sınıf adının seçili dildeki karşılığı.

    Çeviri yoksa İngilizce ad olduğu gibi döner — eksik çeviri yüzünden ekranda
    boşluk çıkmaz.
    """
    if not ad:
        return ''
    kod = kod or aktif()
    karsilik = (ADLAR.get(ad) or {}).get(kod)
    if karsilik:
        return karsilik
    return _okunakli(ad) if kod == 'en' else _okunakli(ad)


def sinif_sozlugu(siniflar: dict, kod: str = None) -> dict:
    """{id: ad} sözlüğünü seçili dile çevirir (etiketleme listesi, canlı akış)."""
    return {int(i): sinif_adi(ad, kod) for i, ad in siniflar.items()}
