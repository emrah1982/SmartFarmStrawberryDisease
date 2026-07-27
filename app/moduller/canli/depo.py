"""Canlı akıştan gelen karenin diske ve veritabanına yazılması.

Modülün TEK veritabanı temas noktası burasıdır. Kayıt biçimi çekirdekle
birebir aynıdır (Analiz + Tespit) — canlı kayıtlar da geçmişte, incelemede,
etiketlemede ve haritada diğerleriyle aynı şekilde görünür; ayrı bir "canlı
kayıt" kavramı yoktur, yalnızca kaynak_tip='canli' olur.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from app import config
from app.detector import Kutu, Sonuc
from app.moduller.canli import servis

logger = logging.getLogger(__name__)


def kare_kaydet(frame, kutular: List[Kutu], sera_id: Optional[int] = None,
                kaynak_ad: str = 'canli') -> Optional[int]:
    """Kareyi kaydeder, Analiz kaydı açar; kayıt id'sini döner.

    Hata durumunda None döner ve akış kesilmez — canlı izleme, kayıt
    yazılamadı diye durmamalı.
    """
    import cv2

    try:
        tekil = f'canli_{uuid.uuid4().hex[:12]}'
        orijinal = config.UPLOAD_DIR / f'{tekil}.jpg'
        cikti = config.RESULT_DIR / f'{tekil}.jpg'
        orijinal.parent.mkdir(parents=True, exist_ok=True)
        cikti.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(orijinal), frame)
        cv2.imwrite(str(cikti), servis.ciz(frame, kutular))

        sonuc = Sonuc(kutular=list(kutular), sonuc_yolu=str(cikti),
                      islenen_kare=1, sure_ms=0)

        from app import main as cekirdek                # döngüsel import olmasın
        with cekirdek.SessionLocal() as db:
            kayit = cekirdek._kaydet(
                sonuc, db, 'canli', kaynak_ad,
                orijinal.relative_to(config.STORAGE_DIR).as_posix(),
                sera_id=sera_id)
            return kayit.id
    except Exception as e:
        logger.warning(f'Canlı kare kaydedilemedi: {e}')
        return None


def seralar():
    """Sayfadaki sera seçimi için: [{'id':1,'tam_ad':'...'}].

    ORM nesnesi DÖNDÜRÜLMEZ: oturum kapandıktan sonra şablonda `sera.tam_ad`
    okunursa ilişkili üretici tembel yüklenmeye çalışır ve
    DetachedInstanceError verir. Gerekli alanlar oturum içindeyken düz veriye
    çevrilir.
    """
    from app import main as cekirdek
    from app import yetki
    try:
        with cekirdek.SessionLocal() as db:
            return [{'id': s.id, 'tam_ad': s.tam_ad}
                    for s in yetki.gorunur_seralar(db, yetki.aktif_kullanici(db))]
    except Exception as e:
        logger.warning(f'Sera listesi alınamadı: {e}')
        return []


def sonuc_yolu(kayit_id: int) -> str:
    """Kaydedilen görselin /media altındaki yolu (arayüzde küçük önizleme)."""
    from app import main as cekirdek
    from app.database import Analiz
    try:
        with cekirdek.SessionLocal() as db:
            a = db.get(Analiz, kayit_id)
            return a.sonuc_yolu if a else ''
    except Exception:
        return ''


__all__ = ['kare_kaydet', 'seralar', 'sonuc_yolu', 'Path']
