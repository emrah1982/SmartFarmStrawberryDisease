"""Sınıf kütüğü — tek yetkili liste.

NE İŞE YARAR?
    Hangi sınıflar var, ID'leri ne, ekranda hangi adla görünüyor, hangi güven
    eşiğiyle kabul ediliyor ve açık mı — hepsi configs/siniflar.yaml'da tek
    yerde durur. Yeni bir zararlı/hastalık eklemek KOD DEĞİŞTİRMEZ.

ÜÇ AYRI KAVRAM, KARIŞTIRILMAMALI
    1. Eğitimdeki ad (İngilizce)  → etiket dosyaları ve modelin ürettiği ad.
       ASLA değişmez; değişirse eski eğitim verisi geçersiz olur.
    2. Ekrandaki ad (tr/en)       → yalnızca görünüm.
    3. ID                         → etiket dosyalarındaki sayı. Bir kez verilir,
       BİR DAHA DEĞİŞTİRİLMEZ; değişirse geçmiş etiketler yanlış sınıfa kayar.

SINIF BAZLI EŞİK — NEDEN?
    Bazı sınıflar diğerlerinden çok daha gürültülüdür. Örnek: olgunluk
    sınıfları ayrı bir veri setinden geldi ve orada olgunlaşmamış çilek yeşil
    görünüyor; model "yeşil yuvarlak kütle" ile çilek yaprağını karıştırıyor.
    Genel eşiği yükseltmek erken evre hastalık tespitlerini de kaybettirir.
    Sınıf bazlı eşik, sorunlu sınıfı tek başına sıkılaştırır.

    Bu bir GÖRÜNTÜLEME filtresidir: modeli düzeltmez, yanlış tespiti gizler.
    Kalıcı çözüm negatif örneklerle yeniden eğitimdir (bkz. README).
"""

import logging
import os
from pathlib import Path

import yaml

from app import config

logger = logging.getLogger(__name__)

def _yol(dosya: str, urun=None):
    from app import urunler
    return urunler.yapilandirma(urun, dosya)


# Varsayılan ürünün yolları (modül düzeyi API bunları kullanır)
KUTUK_YOLU = _yol('siniflar.yaml')
EGITIM_YAML = _yol('veri.yaml')

# Ortam değişkeniyle tek seferlik kapatma (Docker'da dosya düzenlemeden):
#   KAPALI_SINIFLAR="strawberry_unripe,strawberry_semi_ripe"
_KAPALI_ENV = {a.strip() for a in os.environ.get('KAPALI_SINIFLAR', '').split(',') if a.strip()}


def _yukle(yol=None) -> dict:
    yol = yol or KUTUK_YOLU
    if not yol.exists():
        return {}
    try:
        return yaml.safe_load(yol.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as e:
        logger.error(f'{yol} okunamadı: {e}')
        return {}


KUTUK = _yukle()

# Ürün başına kütük önbelleği. Çok bitkili kurulumda her ürünün sınıf listesi
# BAĞIMSIZDIR; ID'ler ürün içinde 0..n-1'dir, ürünler arası çakışma olmaz.
_urun_kutukleri = {}


def kutuk(urun=None) -> dict:
    from app import urunler
    ad = urunler.slug(urun) if urun else urunler.VARSAYILAN
    if ad not in _urun_kutukleri:
        _urun_kutukleri[ad] = _yukle(_yol('siniflar.yaml', ad))
    return _urun_kutukleri[ad]


def bosalt_onbellek():
    _urun_kutukleri.clear()


def bilgi(ad: str, urun=None) -> dict:
    return (kutuk(urun) if urun else KUTUK).get(ad) or {}


def esik(ad: str, urun=None) -> float:
    """Bu sınıf için kabul eşiği. Tanımlı değilse genel CONF_THRESHOLD."""
    d = bilgi(ad, urun).get('esik')
    try:
        return float(d) if d is not None else config.CONF_THRESHOLD
    except (TypeError, ValueError):
        return config.CONF_THRESHOLD


def aktif_mi(ad: str, urun=None) -> bool:
    """Kapalı sınıflar hiç gösterilmez (model yine üretir, arayüz eler)."""
    if ad in _KAPALI_ENV:
        return False
    return bilgi(ad, urun).get('aktif', True) is not False


def en_dusuk_esik() -> float:
    """Modele verilecek conf değeri.

    Sınıf eşikleri elemeyi SONRADAN yapar; bu yüzden model en düşük eşikle
    çalıştırılır, yoksa yüksek eşikli sınıf uğruna diğerleri kaybolurdu.
    """
    esikler = [esik(ad) for ad in KUTUK if aktif_mi(ad)]
    return min(esikler + [config.CONF_THRESHOLD])


def kabul_edilir_mi(ad: str, guven: float, urun=None) -> bool:
    return aktif_mi(ad, urun) and guven >= esik(ad, urun)


def grup(ad: str, urun=None) -> str:
    """hastalik | zararli | olgunluk | besin | diger — arayüzde gruplama.

    Grup, YAPILACAK İŞİ ayırır: hastalıkta fungisit/kültürel önlem,
    zararlıda sayım-eşik ve biyolojik mücadele, besinde gübreleme.
    Bu yüzden 'besin' ayrı bir gruptur — belirtisi hastalığa benzer
    (yaprakta sararma/leke) ama çözümü tamamen farklıdır.
    """
    return bilgi(ad, urun).get('grup', 'diger')


def egitimde_mi(ad: str, urun=None) -> bool:
    """Model bu sınıfı tanıyor mu?

    Kütüğe yeni sınıf eklemek onu MODELE ÖĞRETMEZ; yalnızca etiketlemede
    kullanılabilir hale getirir. Model ancak yeniden eğitimden sonra tanır.
    """
    return bilgi(ad, urun).get('egitimde', True) is not False


def id_haritasi() -> dict:
    """{id: ad} — etiketleme ekranının kullandığı liste.

    Eğitimdeki sınıflar strawberry_data.yaml'dan gelir (kaynak orasıdır).
    Kütükte `id` verilmiş ama henüz eğitilmemiş sınıflar da eklenir: böylece
    yeni bir zararlı için VERİ TOPLAMAYA hemen başlanabilir, model bir sonraki
    eğitimde öğrenir.
    """
    harita = {}
    if EGITIM_YAML.exists():
        try:
            cfg = yaml.safe_load(EGITIM_YAML.read_text(encoding='utf-8')) or {}
            isimler = cfg.get('names', {})
            if isinstance(isimler, list):
                harita = {i: ad for i, ad in enumerate(isimler)}
            else:
                harita = {int(k): v for k, v in isimler.items()}
        except (yaml.YAMLError, ValueError) as e:
            logger.error(f'strawberry_data.yaml okunamadı: {e}')

    for ad, d in KUTUK.items():
        kimlik = (d or {}).get('id')
        if kimlik is None:
            continue
        kimlik = int(kimlik)
        if kimlik in harita and harita[kimlik] != ad:
            logger.error(f'ID çakışması: {kimlik} hem {harita[kimlik]} hem {ad}. '
                         'Etiketler yanlış sınıfa kayar — configs/siniflar.yaml düzeltin.')
            continue
        harita[kimlik] = ad
    return dict(sorted(harita.items()))


def yeni_id() -> int:
    """Yeni sınıf için bir sonraki boş ID (kütüğe eklerken kullanılır)."""
    mevcut = id_haritasi()
    return (max(mevcut) + 1) if mevcut else 0
