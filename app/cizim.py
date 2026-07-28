"""Tespit kutularının görüntü üzerine çizimi.

NEDEN ULTRALYTICS'İN r.plot() KULLANILMIYOR?
    r.plot() etiketleri EĞİTİMDEKİ İngilizce adla yazar: kullanıcı ekranda
    "Kurşuni Küf" görse bile kaydedilen görselin üzerinde "Gray Mold" yazar.
    Ayrıca kapalı/eşik altı sınıfları da çizer — arayüzde elenen bir tespit
    görselde görünmeye devam ederdi.

NEDEN cv2.putText DEĞİL?
    OpenCV'nin dahili yazı tipleri yalnızca ASCII çizer; "Olgunlaşmamış Çilek"
    ekranda "Olgunla?mam?? ?ilek" olur. Türkçe karakter için TrueType yazı
    tipi gerekir → Pillow kullanılır (zaten bağımlılık).
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sınıfa göre sabit renk (BGR) — aynı hastalık her görselde aynı renkte olsun
RENKLER = [
    (54, 67, 244), (156, 39, 176), (229, 136, 30), (139, 195, 74), (26, 82, 245),
    (171, 57, 57), (51, 202, 192), (65, 76, 109), (233, 30, 99), (76, 175, 80),
]

_YAZI_ADAYLARI = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',      # Docker (fonts-dejavu-core)
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/segoeuib.ttf',                             # Windows
    'C:/Windows/Fonts/arialbd.ttf',
]
_yazi_yolu = None
_yazi_arandi = False


def _yazi_tipi_yolu():
    global _yazi_yolu, _yazi_arandi
    if not _yazi_arandi:
        _yazi_arandi = True
        for y in _YAZI_ADAYLARI:
            if Path(y).exists():
                _yazi_yolu = y
                break
        if _yazi_yolu is None:
            logger.warning('TrueType yazı tipi bulunamadı; etiketler ASCII çizilecek '
                           '(Türkçe karakterler bozulabilir).')
    return _yazi_yolu


def _asciye_indir(metin: str) -> str:
    """Yazı tipi yoksa son çare: Türkçe harfleri ASCII karşılığına çevir."""
    tablo = str.maketrans('çğıöşüÇĞİÖŞÜ', 'cgiosuCGIOSU')
    return metin.translate(tablo)


def kutulari_ciz(frame, kutular, ad_cevir=None):
    """Kutuları ve etiketleri çizer, yeni görüntüyü döner.

    kutular   : x,y,w,h (0-1 normalize), sinif_id, sinif_adi, guven alanları
    ad_cevir  : ad → görünen ad (dil çevirisi). None ise ad olduğu gibi yazılır.
    """
    if frame is None:
        return frame
    ad_cevir = ad_cevir or (lambda a: a)
    y_boy, x_boy = frame.shape[:2]
    kalinlik = max(2, round(min(x_boy, y_boy) / 400))
    punto = max(14, round(min(x_boy, y_boy) / 38))

    kopya = frame.copy()
    kutu_bilgi = []
    for k in kutular:
        renk = RENKLER[int(getattr(k, 'sinif_id', 0)) % len(RENKLER)]
        x1 = int((k.x - k.w / 2) * x_boy)
        y1 = int((k.y - k.h / 2) * y_boy)
        x2 = int((k.x + k.w / 2) * x_boy)
        y2 = int((k.y + k.h / 2) * y_boy)
        cv2.rectangle(kopya, (x1, y1), (x2, y2), renk, kalinlik)
        kutu_bilgi.append((x1, y1, renk,
                           f'{ad_cevir(k.sinif_adi)} %{k.guven * 100:.0f}'))

    yol = _yazi_tipi_yolu()
    if yol is None:
        for x1, y1, renk, etiket in kutu_bilgi:
            cv2.putText(kopya, _asciye_indir(etiket), (x1, max(punto, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, punto / 34, renk, 2, cv2.LINE_AA)
        return kopya

    from PIL import Image, ImageDraw, ImageFont
    resim = Image.fromarray(cv2.cvtColor(kopya, cv2.COLOR_BGR2RGB))
    ciz = ImageDraw.Draw(resim)
    font = ImageFont.truetype(yol, punto)

    for x1, y1, renk, etiket in kutu_bilgi:
        rgb = (renk[2], renk[1], renk[0])
        kutu = ciz.textbbox((0, 0), etiket, font=font)
        g, y = kutu[2] - kutu[0], kutu[3] - kutu[1]
        ty = y1 - y - 6 if y1 - y - 6 > 0 else y1        # üstte yer yoksa içeri al
        ciz.rectangle([x1, ty, x1 + g + 8, ty + y + 6], fill=rgb)
        ciz.text((x1 + 4, ty + 3), etiket, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(resim), cv2.COLOR_RGB2BGR)


def sonuc_yaz(frame, kutular, cikti_yol: str, ad_cevir=None) -> bool:
    """Çizilmiş görüntüyü diske yazar."""
    try:
        return bool(cv2.imwrite(str(cikti_yol), kutulari_ciz(frame, kutular, ad_cevir)))
    except Exception as e:
        logger.warning(f'Sonuç görseli yazılamadı: {e}')
        return False


__all__ = ['kutulari_ciz', 'sonuc_yaz', 'RENKLER', 'List']
