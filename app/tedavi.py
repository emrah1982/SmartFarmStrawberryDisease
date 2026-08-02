"""Tedavi önerisi kütüğü ve ORGAN BAZLI çözümleme.

NEDEN AYRI MODÜL?
    Öneriler sınıf bazlıydı: "Gray Mold" için tek bir metin vardı ve
    belirtisi "Meyvede gri, tozlu küf tabakası" diye yazılıydı. Hiyerarşik
    boru hattı aynı sınıfı hem yaprakta hem meyvede bulabildiği için
    yapraktaki bulguya meyve metni gösteriliyordu — kullanıcı yanlış
    belirtiyi arıyor, yanlış işi yapıyordu.

    İki hastalık ayrı sınıf yapılamaz: model ikisini de "Gray Mold" olarak
    öğrenir (etiketler öyle). Ayrım TESPİT anında değil, SUNUM anında
    yapılmalı — burada.

YAPILANDIRMA BİÇİMİ
    Gray Mold:
      ad: Kurşuni Küf (Botrytis)
      aciliyet: yuksek                 # organ bilinmiyorsa geçerli
      belirti: ...                     # ortak/genel metin
      onlem: [...]
      organ:                           # İSTEĞE BAĞLI — yalnızca farklı olanı yaz
        leaf:
          aciliyet: orta
          belirti: Yaprakta ...
          onlem: [...]                 # verilirse ortak listeyi DEĞİŞTİRİR
        fruit:
          belirti: Meyvede ...

    Organ anahtarı yoksa davranış eskisiyle birebir aynıdır; mevcut
    yapılandırmaların hiçbiri bozulmaz.

BİRLEŞTİRME KURALI
    Organ bloğunda VERİLEN alanlar ortak alanları ezer, verilmeyenler
    ortaktan gelir. Listeler birleştirilmez, değiştirilir: yaprak için
    "meyveyi hasat edin" maddesi ortakta kalırsa yanlış olur; kısmi
    birleştirme sessizce alakasız tavsiye üretirdi.
"""

import logging

logger = logging.getLogger(__name__)

# Organ bloğunda ezilebilen alanlar. `ad` bilerek DIŞARIDA: görünen sınıf adı
# organa göre değişmemeli, yoksa aynı hastalık iki isimle anılır.
EZILEBILIR = ('aciliyet', 'belirti', 'etken', 'onlem', 'uzman')


def _oku(p) -> dict:
    import yaml
    try:
        return yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as e:
        logger.error(f'{p} okunamadı: {e}')
        return {}


def yukle(urun=None) -> dict:
    """Ortak + ürüne özgü tedavi metinleri. Aynı ad varsa ürün ezer."""
    from app import urunler
    birlesik = {}
    ortak = urunler.ortak_yapilandirma('tedavi_onerileri.yaml')
    if ortak is not None:
        birlesik.update(_oku(ortak))
    p = urunler.yapilandirma(urun, 'tedavi_onerileri.yaml')
    if p.exists():
        birlesik.update(_oku(p))
    return birlesik


def coz(kutuk: dict, sinif_adi: str, organ: str = '') -> dict:
    """Bu sınıf + organ için geçerli öneriyi döner.

    Organ boşsa veya o organ için özel metin yoksa ortak metin döner.
    """
    kayit = (kutuk or {}).get(sinif_adi)
    if not kayit:
        return {}

    ortak = {k: v for k, v in kayit.items() if k != 'organ'}
    if not organ:
        return ortak

    ozel = (kayit.get('organ') or {}).get(organ.lower())
    if not ozel:
        return ortak

    birlesik = dict(ortak)
    for alan in EZILEBILIR:
        if alan in ozel and ozel[alan] not in (None, '', []):
            birlesik[alan] = ozel[alan]
    return birlesik


def organa_ozel_mi(kutuk: dict, sinif_adi: str) -> bool:
    """Bu sınıfın organa göre değişen metni var mı? (arayüzde not için)"""
    return bool(((kutuk or {}).get(sinif_adi) or {}).get('organ'))
