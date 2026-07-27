"""Konum modülü testleri (EXIF GPS, kamera konumu, yaygınlık hesabı)."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = tempfile.mkdtemp(prefix='konum_test_')
os.environ['STORAGE_DIR'] = _tmp
os.environ['DATABASE_URL'] = f'sqlite:///{Path(_tmp) / "konum.db"}'

from fastapi.testclient import TestClient          # noqa: E402

from app import main                                # noqa: E402
from app.detector import Kutu, Sonuc                # noqa: E402
from app.moduller.konum import servis               # noqa: E402


class SahteDetector:
    hazir = True

    def __init__(self, kutular=None):
        self._kutular = kutular if kutular is not None else [
            Kutu(0, 'Angular Leafspot', 0.8, .5, .5, .2, .2)]

    def goruntu(self, kaynak, cikti):
        Path(cikti).write_bytes(b'sahte')
        return Sonuc(kutular=list(self._kutular), sonuc_yolu=str(cikti), sure_ms=5)

    def kamera(self, url, cikti, kaynak_kaydet=None):
        Path(cikti).write_bytes(b'sahte')
        if kaynak_kaydet:
            Path(kaynak_kaydet).write_bytes(b'kare')
        return Sonuc(kutular=list(self._kutular), sonuc_yolu=str(cikti), sure_ms=5)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, 'detector', SahteDetector())
    with TestClient(main.app) as c:
        yield c


def gps_jpeg(enlem_dms=(36, 51, 11.52), boylam_dms=(30, 43, 17.28)) -> bytes:
    """EXIF GPS taşıyan gerçek bir JPEG üretir (telefon/drone fotoğrafı taklidi)."""
    import io as _io
    from PIL import ExifTags, Image
    from PIL.TiffImagePlugin import IFDRational

    im = Image.fromarray(np.full((60, 80, 3), 180, dtype='uint8'))
    exif = Image.Exif()

    def kesir(x):
        return IFDRational(int(round(x * 100)), 100)

    exif[ExifTags.IFD.GPSInfo] = {
        ExifTags.GPS.GPSLatitudeRef: 'N',
        ExifTags.GPS.GPSLatitude: tuple(kesir(v) for v in enlem_dms),
        ExifTags.GPS.GPSLongitudeRef: 'E',
        ExifTags.GPS.GPSLongitude: tuple(kesir(v) for v in boylam_dms),
    }
    tampon = _io.BytesIO()
    im.save(tampon, format='JPEG', exif=exif)
    return tampon.getvalue()


def test_exif_gps_okunur():
    yol = Path(_tmp) / 'gps_ornek.jpg'
    yol.write_bytes(gps_jpeg())
    sonuc = servis.exif_gps(yol)
    assert sonuc is not None
    enlem, boylam, _ = sonuc
    assert abs(enlem - 36.8532) < 0.001
    assert abs(boylam - 30.7215) < 0.001


def test_exif_olmayan_goruntu_sorun_cikarmaz():
    from PIL import Image
    yol = Path(_tmp) / 'gpssiz.jpg'
    Image.fromarray(np.zeros((20, 20, 3), dtype='uint8')).save(yol)
    assert servis.exif_gps(yol) is None
    assert servis.exif_gps(Path(_tmp) / 'olmayan.jpg') is None      # dosya yok


def test_yukleme_sirasinda_konum_otomatik_atanir(client):
    """Telefon/drone fotoğrafındaki GPS otomatik okunmalı."""
    from app.database import Analiz, SessionLocal
    client.post('/analiz/dosya',
                files={'dosyalar': ('drone.jpg', gps_jpeg(), 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        a = db.query(Analiz).filter(Analiz.kaynak_ad == 'drone.jpg').first()
        assert a.konum is not None, 'EXIF GPS konum kaydı oluşturmalı'
        assert a.konum.kaynak == 'exif'
        assert abs(a.konum.enlem - 36.8532) < 0.001


def test_kamera_konumu_analize_gecer(client):
    """Sabit kameradan gelen analiz kameranın konumunu devralmalı."""
    from app.database import Analiz, Kamera, SessionLocal
    client.post('/kameralar/ekle',
                data={'ad': 'Kuzey', 'url': 'rtsp://1.2.3.4/s',
                      'blok': 'A blok', 'sira': '3',
                      'enlem': '36.90000', 'boylam': '30.80000'},
                follow_redirects=True)
    with SessionLocal() as db:
        kid = db.query(Kamera).filter(Kamera.ad == 'Kuzey').first().id

    client.post('/analiz/kamera', data={'kamera_id': str(kid)}, follow_redirects=True)
    with SessionLocal() as db:
        a = (db.query(Analiz).filter(Analiz.kaynak_tip == 'kamera')
             .order_by(Analiz.id.desc()).first())
        assert a.konum is not None and a.konum.kaynak == 'kamera'
        assert a.konum.blok == 'A blok' and a.konum.sira == '3'
        assert abs(a.konum.enlem - 36.9) < 1e-6


def test_konum_elle_girilebilir(client):
    from app.database import Analiz, SessionLocal
    client.post('/analiz/dosya', files={'dosyalar': ('elle.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        aid = db.query(Analiz).filter(Analiz.kaynak_ad == 'elle.jpg').first().id

    client.post(f'/konum/kayit/{aid}', data={'blok': 'B blok', 'sira': '7'},
                follow_redirects=True)
    with SessionLocal() as db:
        a = db.get(Analiz, aid)
        assert a.konum.etiket == 'B blok / 7'
        assert a.konum.kaynak == 'elle'


def test_yayginlik_kutu_sayisi_degil_enfekte_oranini_olcer():
    """Tek yaprakta 16 leke, o bölgeyi 16 kat sorunlu yapmaz."""
    class SahteKutu:
        def __init__(self, ad):
            self.sinif_adi = ad

    class SahteKonum:
        def __init__(self, blok):
            self.blok, self.sira = blok, ''
            self.enlem = self.boylam = None
            self.gps_var = False

        @property
        def etiket(self):
            return self.blok

    class SahteAnaliz:
        def __init__(self, blok, kutu_sayisi, hastalik=True):
            self.konum = SahteKonum(blok)
            ad = 'Gray Mold' if hastalik else 'strawberry_ripe'
            self.tespitler = [SahteKutu(ad) for _ in range(kutu_sayisi)]

    kayitlar = [
        SahteAnaliz('A blok', 16),                     # tek görüntü, 16 leke
        SahteAnaliz('A blok', 0, hastalik=False),
        SahteAnaliz('A blok', 0, hastalik=False),
        SahteAnaliz('A blok', 0, hastalik=False),
        SahteAnaliz('B blok', 1),                      # 4 görüntünün 3'ü hastalıklı
        SahteAnaliz('B blok', 1),
        SahteAnaliz('B blok', 1),
        SahteAnaliz('B blok', 0, hastalik=False),
    ]
    hesap = servis.yaygınlık_hesapla(kayitlar)
    sonuc = {b['konum']: b for b in hesap}

    assert sonuc['A blok']['oran'] == 25, 'A blokta 4 görüntünün 1i enfekte'
    assert sonuc['B blok']['oran'] == 75, 'B blokta 4 görüntünün 3ü enfekte'
    assert sonuc['A blok']['kutu'] == 16          # şiddet ayrıca raporlanır
    assert hesap[0]['konum'] == 'B blok', 'yaygınlığa göre sıralanmalı'


def test_olgunluk_siniflari_hastalik_sayilmaz():
    assert servis.hastalik_mi('Gray Mold') is True
    assert servis.hastalik_mi('strawberry_ripe') is False


def test_yayginlik_sayfasi(client):
    client.post('/analiz/dosya',
                files={'dosyalar': ('harita.jpg', gps_jpeg(), 'image/jpeg')},
                follow_redirects=True)
    r = client.get('/konum/yayginlik')
    assert r.status_code == 200
    assert 'Hastalık Yaygınlığı' in r.text
    assert 'GPS dağılımı' in r.text               # koordinatlı kayıt var


def test_modul_menude_gorunur(client):
    r = client.get('/')
    assert '/konum/yayginlik' in r.text, 'modül menüde otomatik görünmeli'
