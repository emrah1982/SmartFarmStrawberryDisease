"""Konum çıkarımı ve yaygınlık hesapları."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def exif_gps(yol: Path) -> Optional[Tuple[float, float, Optional[float]]]:
    """Fotoğrafın EXIF GPS bilgisini (enlem, boylam, yükseklik) döner.

    Telefonlarda konum servisi açıkken çekilen fotoğraflar ve drone görüntüleri
    bu bilgiyi taşır. Yoksa None döner — konum elle veya kameradan gelir.
    """
    try:
        from PIL import Image, ExifTags
    except ImportError:
        logger.warning('Pillow kurulu değil; EXIF GPS okunamıyor')
        return None

    try:
        with Image.open(yol) as im:
            exif = im.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if not gps_ifd:
                return None
            veri = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    except Exception as e:                        # bozuk/desteklenmeyen EXIF
        logger.debug(f'EXIF okunamadı ({yol.name}): {e}')
        return None

    def ondalik(dms, ref) -> Optional[float]:
        try:
            derece, dakika, saniye = (float(x) for x in dms)
        except (TypeError, ValueError):
            return None
        deger = derece + dakika / 60 + saniye / 3600
        if ref in ('S', 'W'):
            deger = -deger
        return deger

    enlem = ondalik(veri.get('GPSLatitude'), veri.get('GPSLatitudeRef'))
    boylam = ondalik(veri.get('GPSLongitude'), veri.get('GPSLongitudeRef'))
    if enlem is None or boylam is None:
        return None

    yukseklik = None
    if 'GPSAltitude' in veri:
        try:
            yukseklik = float(veri['GPSAltitude'])
            if veri.get('GPSAltitudeRef') in (1, b'\x01'):     # deniz seviyesi altı
                yukseklik = -yukseklik
        except (TypeError, ValueError):
            pass

    return enlem, boylam, yukseklik


# Hastalık sınıfları (olgunluk sınıfları yaygınlık hesabına girmez)
def hastalik_mi(sinif_adi: str) -> bool:
    return not sinif_adi.startswith('strawberry_')


def yaygınlık_hesapla(kayitlar) -> List[Dict]:
    """Konuma göre hastalık yaygınlığı.

    ÖNEMLİ: Yaygınlık ölçüsü kutu sayısı DEĞİL, 'enfekte görüntü oranı'dır.
    Tek bir yaprakta 16 leke olması o bölgeyi 16 kat sorunlu yapmaz; asıl
    soru "bu bölgede çekilen görüntülerin yüzde kaçında hastalık var".
    Kutu sayısı ayrıca şiddet göstergesi olarak raporlanır.
    """
    gruplar: Dict[str, Dict] = {}
    for a in kayitlar:
        k = getattr(a, 'konum', None)
        anahtar = k.etiket if k else 'konum yok'
        g = gruplar.setdefault(anahtar, {
            'konum': anahtar, 'blok': k.blok if k else '', 'sira': k.sira if k else '',
            'analiz': 0, 'enfekte': 0, 'kutu': 0, 'siniflar': {},
            'enlem': k.enlem if k and k.gps_var else None,
            'boylam': k.boylam if k and k.gps_var else None,
        })
        g['analiz'] += 1
        hastalikli = False
        for t in a.tespitler:
            if hastalik_mi(t.sinif_adi):
                hastalikli = True
                g['kutu'] += 1
                g['siniflar'][t.sinif_adi] = g['siniflar'].get(t.sinif_adi, 0) + 1
        if hastalikli:
            g['enfekte'] += 1

    sonuc = []
    for g in gruplar.values():
        g['oran'] = round(100 * g['enfekte'] / g['analiz']) if g['analiz'] else 0
        g['en_sik'] = max(g['siniflar'], key=g['siniflar'].get) if g['siniflar'] else '—'
        sonuc.append(g)
    return sorted(sonuc, key=lambda x: (-x['oran'], -x['kutu']))


def gps_noktalari(kayitlar) -> List[Dict]:
    """Harita için GPS noktaları (normalize edilmiş 0-1 koordinatlarla).

    Çevrimdışı çalışabilmek için harita karosu (tile) kullanılmaz; noktalar
    kendi sınırlayıcı kutularına göre ölçeklenip basit bir düzlemde çizilir.
    """
    noktalar = [a for a in kayitlar
                if getattr(a, 'konum', None) and a.konum.gps_var]
    if not noktalar:
        return []

    enlemler = [a.konum.enlem for a in noktalar]
    boylamlar = [a.konum.boylam for a in noktalar]
    en_az_e, en_cok_e = min(enlemler), max(enlemler)
    en_az_b, en_cok_b = min(boylamlar), max(boylamlar)
    fark_e = (en_cok_e - en_az_e) or 1e-9
    fark_b = (en_cok_b - en_az_b) or 1e-9

    cikti = []
    for a in noktalar:
        hastalik = sum(1 for t in a.tespitler if hastalik_mi(t.sinif_adi))
        cikti.append({
            'id': a.id,
            'x': round(100 * (a.konum.boylam - en_az_b) / fark_b, 2),
            'y': round(100 * (en_cok_e - a.konum.enlem) / fark_e, 2),  # kuzey yukarıda
            'enlem': a.konum.enlem, 'boylam': a.konum.boylam,
            'hastalik': hastalik,
            'ozet': a.ozet,
        })
    return cikti
