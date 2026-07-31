"""Çok modelli mimarinin model deposu.

Her uzman model (organ, yaprak hastalığı, meyve hastalığı, olgunluk, zararlı)
ayrı bir ağırlık dosyasıdır. Bu modül onları kütükten okur, GEREKTİĞİNDE yükler
ve bellekte tutar.

NEDEN LAZY (gerektiğinde) YÜKLEME?
    Beş modeli birden belleğe almak hem RAM hem başlangıç süresi demektir.
    Görüntüde meyve yoksa meyve hastalığı modeli hiç yüklenmez.

NEDEN AYRI DOSYA?
    Yeni bir zararlı eklendiğinde yalnızca zararlı modeli yeniden eğitilir;
    hastalık modelleri dokunulmadan kalır. Tek modelde her ekleme, tüm
    sistemin yeniden eğitilmesini gerektiriyordu.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from app import config

logger = logging.getLogger(__name__)

def _kutuk_yolu(urun=None) -> Path:
    from app import urunler
    return urunler.yapilandirma(urun, 'modeller.yaml')


def model_dizini(urun=None) -> Path:
    from app import urunler
    return urunler.model_dizini(urun)


KUTUK_YOLU = _kutuk_yolu()
MODEL_DIZINI = model_dizini()


@dataclass
class ModelTanimi:
    ad: str
    dosya: str
    rol: str
    siniflar: List[str] = field(default_factory=list)
    tetik: List[str] = field(default_factory=list)
    esik: float = 0.25
    aktif: bool = True
    zorunlu: bool = False
    aciklama: str = ''
    urun: str = ''          # hangi ürünün modeli (boş = varsayılan)
    # Bu modelin ÇIKARIM çözünürlüğü. None ise config.IMGSZ kullanılır.
    #
    # NEDEN MODEL BAŞINA? Ölçüldü: organ modeli 1024'te bir serada 2 meyve,
    # 640'ta 3 meyve buluyor ve HER kutuda güven daha yüksek çıkıyor
    # (0.841/0.793/0.608 karşı 0.738/0.669/0.339). Tek genel imgsz bütün
    # modellere dayatılınca bazıları kaybediyor; her model kendi ölçülmüş
    # değerini taşımalı.
    imgsz: Optional[int] = None

    @property
    def yol(self) -> Path:
        return model_dizini(self.urun or None) / self.dosya

    @property
    def var(self) -> bool:
        return self.yol.exists()


def _kutugu_oku(urun=None) -> Dict[str, ModelTanimi]:
    yol = _kutuk_yolu(urun)
    if not yol.exists():
        logger.warning(f'Model kütüğü yok: {yol}')
        return {}
    try:
        ham = yaml.safe_load(yol.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as e:
        logger.error(f'configs/modeller.yaml okunamadı: {e}')
        return {}

    tanimlar = {}
    for ad, d in ham.items():
        d = d or {}
        tanimlar[ad] = ModelTanimi(
            ad=ad,
            dosya=d.get('dosya', f'{ad}.pt'),
            rol=d.get('rol', ad),
            siniflar=list(d.get('siniflar') or []),
            tetik=list(d.get('tetik') or []),
            esik=float(d.get('esik', config.CONF_THRESHOLD)),
            aktif=d.get('aktif', True) is not False,
            zorunlu=bool(d.get('zorunlu', False)),
            aciklama=(d.get('aciklama') or '').strip(),
            urun=urun or '',
            imgsz=int(d['imgsz']) if d.get('imgsz') else None,
        )
    return tanimlar


TANIMLAR: Dict[str, ModelTanimi] = _kutugu_oku()

# Ürün başına kütük — her bitkinin KENDİ model seti vardır.
_urun_tanimlari: Dict[str, Dict[str, ModelTanimi]] = {}


def tanimlar(urun=None) -> Dict[str, ModelTanimi]:
    from app import urunler
    if urun is None:
        return TANIMLAR
    ad = urunler.slug(urun)
    if ad not in _urun_tanimlari:
        _urun_tanimlari[ad] = _kutugu_oku(ad)
    return _urun_tanimlari[ad]

# Yüklenmiş modeller (ad → YOLO nesnesi). Süreç ömrü boyunca bellekte kalır.
_yuklu: Dict[str, object] = {}


def tanim(ad: str, urun=None) -> Optional[ModelTanimi]:
    return tanimlar(urun).get(ad)


def rol_ile(rol: str, urun=None) -> List[ModelTanimi]:
    """Bir role sahip AKTİF ve DOSYASI VAR olan modeller."""
    return [t for t in tanimlar(urun).values() if t.rol == rol and t.aktif and t.var]


def tetiklenen(organ: str, urun=None) -> List[ModelTanimi]:
    """Bu organ bulunduğunda çalışacak uzman modeller.

    Eşleşme büyük/küçük harf duyarsızdır: organ modeli sınıf adlarını
    dataset'teki biçimde ('Leaf') üretir, kütükte 'leaf' yazılıdır.
    """
    o = organ.lower()
    return [t for t in tanimlar(urun).values()
            if t.aktif and t.var and o in [x.lower() for x in t.tetik]]


def yukle(ad: str, urun=None):
    """Modeli (gerekiyorsa) yükler. Yoksa None döner — akış çökmemeli."""
    from app import urunler
    anahtar = f'{urunler.slug(urun) if urun else urunler.VARSAYILAN}/{ad}'
    t = tanimlar(urun).get(ad)
    if t is None or not t.aktif:
        return None
    if anahtar in _yuklu:
        return _yuklu[anahtar]
    if not t.var:
        return None
    try:
        from ultralytics import YOLO
        logger.info(f'Model yükleniyor: {anahtar} ({t.yol})')
        _yuklu[anahtar] = YOLO(str(t.yol))
        return _yuklu[anahtar]
    except Exception as e:
        logger.error(f'Model yüklenemedi ({t.ad}): {e}')
        return None


def bosalt():
    """Bellekteki modelleri bırakır (testlerde ve model değişiminde)."""
    _yuklu.clear()


def durum(urun=None) -> List[dict]:
    """Arayüzde gösterilecek model durumu: hangisi hazır, hangisi eksik."""
    out = []
    for t in tanimlar(urun).values():
        out.append({
            'ad': t.ad, 'rol': t.rol, 'dosya': t.dosya, 'var': t.var,
            'aktif': t.aktif, 'zorunlu': t.zorunlu, 'tetik': t.tetik,
            'siniflar': t.siniflar, 'aciklama': t.aciklama,
            'urun': t.urun or '', 'yol': str(t.yol),
            'yuklu': any(k.endswith('/' + t.ad) for k in _yuklu),
        })
    return out


def hiyerarsik_hazir(urun=None) -> bool:
    """Organ modeli var mı? Yoksa boru hattı mirasa düşer."""
    return any(t.var and t.aktif for t in tanimlar(urun).values() if t.rol == 'organ')


def eksikler(urun=None) -> List[str]:
    return [t.ad for t in tanimlar(urun).values()
            if t.aktif and not t.var and t.rol != 'miras']


def bosalt_kutuk():
    """Ürün kütüğü önbelleğini temizler (testlerde ve yapılandırma değişiminde)."""
    _urun_tanimlari.clear()
