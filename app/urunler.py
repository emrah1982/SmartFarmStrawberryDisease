"""Ürün (bitki türü) kapsamı — çok bitkili kurulumun temeli.

NEDEN GEREKLİ?
    Bugün yalnızca çilek var. İkinci bitki eklendiğinde en sinsi hata SINIF ID
    ÇAKIŞMASIDIR: domatesin "Leaf Spot"u çileğinkiyle aynı ID'yi alırsa model
    iki farklı hastalığı tek sınıf sanır; ayrı ID alırsa her model gereksiz
    yere diğer bitkinin sınıflarını taşır.

    Çözüm: her ürünün KENDİ sınıf kütüğü, KENDİ model dosyaları, KENDİ
    dataset'i olur. ID'ler ürün içinde 0..n-1'dir; ürünler arası çakışma
    kavramsal olarak imkânsızdır.

ÜRÜN NEREDEN BELİRLENİR?
    1. Açıkça verilirse (analiz çağrısındaki parametre)
    2. Seranın `urun` alanından  ← asıl kaynak, üretici zaten giriyor
    3. VARSAYILAN_URUN ortam değişkeni
    4. 'cilek'

    Bitki türünü GÖRÜNTÜDEN tespit etmek birincil yol DEĞİLDİR: kare tamamen
    yaprakla dolduğunda çilek/domates ayrımı güvenilmezdir. "Hangi seradasın"
    bilgisi bedava ve kesindir. Görüntüden tespit ileride yalnızca DOĞRULAMA
    için eklenebilir ("bu sera çilek kayıtlı ama görüntü domates gibi").

DİZİN DÜZENİ
    configs/urunler/<urun>/  modeller.yaml · siniflar.yaml · tedavi_onerileri.yaml
    models/<urun>/           organ.pt · leaf_disease.pt · ...
    datasets/<urun>/         organ_detection/ · leaf_disease/ · ...

GERİYE DÖNÜK UYUMLULUK
    Ürün klasörü yoksa eski (kapsamsız) yollara düşülür. Böylece bu yapı
    mevcut kurulumu bozmadan devreye girer.
"""

import logging
import os
import unicodedata
from pathlib import Path
from typing import List, Optional

from app import config

logger = logging.getLogger(__name__)

URUN_KOK = config.BASE_DIR / 'configs' / 'urunler'
MODEL_KOK = Path(config.MODEL_PATH).parent
DATASET_KOK = config.BASE_DIR / 'datasets'

VARSAYILAN = os.environ.get('VARSAYILAN_URUN', 'cilek').strip().lower() or 'cilek'


def slug(ad: str) -> str:
    """'Çilek' → 'cilek'. Sera kaydındaki serbest metni kapsam anahtarına çevirir.

    Türkçe karakterler ASCII'ye indirgenir; klasör adları ve ortam değişkenleri
    her yerde sorunsuz taşınsın diye.
    """
    if not ad:
        return VARSAYILAN
    tablo = str.maketrans('çğıöşüÇĞİÖŞÜ', 'cgiosuCGIOSU')
    metin = unicodedata.normalize('NFKC', ad).translate(tablo)
    temiz = ''.join(c if c.isalnum() else '_' for c in metin.lower()).strip('_')
    return temiz or VARSAYILAN


def dizin(urun: Optional[str] = None) -> Path:
    """Ürünün yapılandırma klasörü."""
    return URUN_KOK / slug(urun or VARSAYILAN)


def var_mi(urun: str) -> bool:
    return dizin(urun).is_dir()


def yapilandirma(urun: Optional[str], dosya: str) -> Path:
    """Ürüne ait yapılandırma dosyası; yoksa eski (kapsamsız) yola düşer.

    Geçiş sırasında her iki düzen de çalışsın diye: yeni kurulumda
    configs/urunler/cilek/siniflar.yaml, eskisinde configs/siniflar.yaml.
    """
    ozel = dizin(urun) / dosya
    if ozel.exists():
        return ozel
    return config.BASE_DIR / 'configs' / dosya


def model_dizini(urun: Optional[str] = None) -> Path:
    """Ürünün model klasörü; yoksa eski models/ köküne düşer."""
    ozel = MODEL_KOK / slug(urun or VARSAYILAN)
    return ozel if ozel.is_dir() else MODEL_KOK


def dataset_dizini(urun: Optional[str] = None) -> Path:
    ozel = DATASET_KOK / slug(urun or VARSAYILAN)
    return ozel if ozel.is_dir() else DATASET_KOK


# ═══════════════════════════════════════════════════════════════════════
# ORTAK KAPSAM — ürüne bağlı OLMAYAN varlıklar
#
# Ürün kapsamı hastalıklar için zorunludur: `Leaf Spot` çilekte
# *Mycosphaerella fragariae*, fındıkta *Piggotia coryli*'dir. Aynı ad,
# farklı etken, farklı ilaç — birleştirilirse yanlış tedavi önerilir.
#
# AMA BÖCEK TÜRÜ İÇİN DURUM TERSİDİR. Danaburnu (*Gryllotalpa*) hangi
# bitkinin yanında fotoğraflanırsa fotoğraflansın AYNI TÜRDÜR. Böcek
# teşhis akışı makro fotoğraftan CANLIYI tanır; bitkiyi hiç görmez.
# Ürün başına kopyalansaydı:
#   - aynı model her ürün klasörüne tekrar tekrar konurdu (20 MB × n)
#   - aynı dataset n kez saklanırdı
#   - yeni bitki eklenince böcek kütüğü elle kopyalanır, biri unutulur
#
# Bu yüzden ortak varlıklar KAPSAMSIZ kökte durur:
#   configs/ortak/   modeller.yaml · siniflar.yaml · tedavi_onerileri.yaml
#   models/          bocek_teshis.pt
#   datasets/        bocek_teshis/
#
# KURAL: bir varlık ancak "aynı gerçek nesneyi" gösteriyorsa ortaktır.
# Hastalık adı ortak olabilir ama HASTALIK ortak değildir.
# ═══════════════════════════════════════════════════════════════════════

ORTAK_KOK = URUN_KOK.parent / 'ortak'


def ortak_yapilandirma(dosya: str) -> Optional[Path]:
    """configs/ortak/<dosya> — yoksa None."""
    p = ORTAK_KOK / dosya
    return p if p.exists() else None


def seradan(sera) -> str:
    """Sera kaydından ürün kapsamını çıkarır (üretici zaten giriyor)."""
    return slug(getattr(sera, 'urun', None) if sera is not None else None)


def liste() -> List[dict]:
    """Kurulu ürünler — arayüzde seçim ve durum için."""
    out = []
    if URUN_KOK.is_dir():
        for d in sorted(URUN_KOK.iterdir()):
            if not d.is_dir():
                continue
            out.append({
                'ad': d.name,
                'yapilandirma': str(d),
                'model_dizini': str(MODEL_KOK / d.name),
                'model_var': (MODEL_KOK / d.name).is_dir(),
                'dataset_var': (DATASET_KOK / d.name).is_dir(),
                'varsayilan': d.name == VARSAYILAN,
            })
    if not out:
        # Henüz ürün klasörü yok: eski düzen tek ürün gibi davranır
        out.append({'ad': VARSAYILAN, 'yapilandirma': str(config.BASE_DIR / 'configs'),
                    'model_dizini': str(MODEL_KOK), 'model_var': MODEL_KOK.is_dir(),
                    'dataset_var': DATASET_KOK.is_dir(), 'varsayilan': True})
    return out
