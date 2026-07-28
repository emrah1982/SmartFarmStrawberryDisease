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

KUTUK_YOLU = config.BASE_DIR / 'configs' / 'modeller.yaml'
MODEL_DIZINI = Path(config.MODEL_PATH).parent


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

    @property
    def yol(self) -> Path:
        return MODEL_DIZINI / self.dosya

    @property
    def var(self) -> bool:
        return self.yol.exists()


def _kutugu_oku() -> Dict[str, ModelTanimi]:
    if not KUTUK_YOLU.exists():
        logger.warning(f'Model kütüğü yok: {KUTUK_YOLU}')
        return {}
    try:
        ham = yaml.safe_load(KUTUK_YOLU.read_text(encoding='utf-8')) or {}
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
        )
    return tanimlar


TANIMLAR: Dict[str, ModelTanimi] = _kutugu_oku()

# Yüklenmiş modeller (ad → YOLO nesnesi). Süreç ömrü boyunca bellekte kalır.
_yuklu: Dict[str, object] = {}


def tanim(ad: str) -> Optional[ModelTanimi]:
    return TANIMLAR.get(ad)


def rol_ile(rol: str) -> List[ModelTanimi]:
    """Bir role sahip AKTİF ve DOSYASI VAR olan modeller."""
    return [t for t in TANIMLAR.values() if t.rol == rol and t.aktif and t.var]


def tetiklenen(organ: str) -> List[ModelTanimi]:
    """Bu organ bulunduğunda çalışacak uzman modeller."""
    o = organ.lower()
    return [t for t in TANIMLAR.values()
            if t.aktif and t.var and o in [x.lower() for x in t.tetik]]


def yukle(ad: str):
    """Modeli (gerekiyorsa) yükler. Yoksa None döner — akış çökmemeli."""
    t = TANIMLAR.get(ad)
    if t is None or not t.aktif:
        return None
    if ad in _yuklu:
        return _yuklu[ad]
    if not t.var:
        return None
    try:
        from ultralytics import YOLO
        logger.info(f'Model yükleniyor: {t.ad} ({t.yol})')
        _yuklu[ad] = YOLO(str(t.yol))
        return _yuklu[ad]
    except Exception as e:
        logger.error(f'Model yüklenemedi ({t.ad}): {e}')
        return None


def bosalt():
    """Bellekteki modelleri bırakır (testlerde ve model değişiminde)."""
    _yuklu.clear()


def durum() -> List[dict]:
    """Arayüzde gösterilecek model durumu: hangisi hazır, hangisi eksik."""
    out = []
    for t in TANIMLAR.values():
        out.append({
            'ad': t.ad, 'rol': t.rol, 'dosya': t.dosya, 'var': t.var,
            'aktif': t.aktif, 'zorunlu': t.zorunlu, 'tetik': t.tetik,
            'siniflar': t.siniflar, 'aciklama': t.aciklama,
            'yuklu': t.ad in _yuklu,
        })
    return out


def hiyerarsik_hazir() -> bool:
    """Organ modeli var mı? Yoksa boru hattı mirasa düşer."""
    return any(t.var and t.aktif for t in TANIMLAR.values() if t.rol == 'organ')


def eksikler() -> List[str]:
    return [t.ad for t in TANIMLAR.values() if t.aktif and not t.var and t.rol != 'miras']
