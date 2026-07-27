"""Web uygulaması testleri.

Gerçek YOLO modeli yerine sahte bir detector kullanılır: testler ultralytics
ve eğitilmiş model olmadan da çalışır, sadece uygulama mantığı sınanır.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Testler geçici bir depolama/DB kullansın — gerçek kayıtlara dokunulmaz
_tmp = tempfile.mkdtemp(prefix='cilek_test_')
os.environ['STORAGE_DIR'] = _tmp
os.environ['DATABASE_URL'] = f'sqlite:///{Path(_tmp) / "test.db"}'

from fastapi.testclient import TestClient          # noqa: E402

from app import config, main                        # noqa: E402
from app.detector import Kutu, Sonuc                # noqa: E402


class SahteDetector:
    """Gerçek modeli taklit eder; sabit tespitler üretir."""

    def __init__(self, kutular=None, hazir=True):
        self._kutular = kutular if kutular is not None else [
            Kutu(3, 'Gray Mold', 0.91, 0.5, 0.5, 0.2, 0.2),
            Kutu(4, 'Leaf Spot', 0.42, 0.2, 0.3, 0.1, 0.1),   # düşük güven
        ]
        self.hazir = hazir

    def _sonuc(self, cikti_yol, kare=1):
        Path(cikti_yol).write_bytes(b'sahte-jpeg')
        return Sonuc(kutular=list(self._kutular), sonuc_yolu=str(cikti_yol),
                     islenen_kare=kare, sure_ms=12)

    def goruntu(self, kaynak, cikti):
        return self._sonuc(cikti)

    def video(self, kaynak, cikti):
        return self._sonuc(cikti, kare=8)

    def kamera(self, url, cikti, kaynak_kaydet=None):
        if 'hatali' in url:
            raise RuntimeError('Kameraya bağlanılamadı')
        if kaynak_kaydet:
            Path(kaynak_kaydet).write_bytes(b'sahte-kare')
        return self._sonuc(cikti)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, 'detector', SahteDetector())
    with TestClient(main.app) as c:
        yield c


def test_anasayfa_acilir(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'Fotoğraf / Video Yükle' in r.text


def test_foto_analizi_kayit_olusturur(client):
    r = client.post('/analiz/dosya',
                    files={'dosyalar': ('bahce.jpg', b'sahte-goruntu', 'image/jpeg')},
                    follow_redirects=True)
    assert r.status_code == 200
    assert 'Gray Mold' in r.text or 'Kurşuni Küf' in r.text
    # Düşük güvenli tespit içerdiği için inceleme kuyruğuna düşmeli
    assert 'inceleme' in r.text.lower()


def test_tedavi_onerisi_gosterilir(client):
    r = client.post('/analiz/dosya',
                    files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                    follow_redirects=True)
    assert 'Kurşuni Küf' in r.text          # configs/tedavi_onerileri.yaml'dan
    assert 'ziraat mühendisi' in r.text     # sorumluluk notu


def test_video_analizi(client):
    r = client.post('/analiz/dosya',
                    files={'dosyalar': ('tarla.mp4', b'sahte-video', 'video/mp4')},
                    follow_redirects=True)
    assert r.status_code == 200
    assert 'kare örneklendi' in r.text


def test_desteklenmeyen_dosya_reddedilir(client):
    r = client.post('/analiz/dosya',
                    files={'dosyalar': ('rapor.pdf', b'%PDF', 'application/pdf')})
    assert r.status_code == 400


def test_kamera_hatasi_anlasilir_mesaj(client):
    r = client.post('/analiz/kamera', data={'url': 'rtsp://hatali/stream'})
    assert r.status_code == 502
    assert 'bağlanılamadı' in r.json()['detail']


def test_kamera_ekle_ve_analiz(client):
    client.post('/kameralar/ekle',
                data={'ad': 'Sera 1', 'url': 'rtsp://1.2.3.4/s', 'konum': 'Kuzey'},
                follow_redirects=True)
    liste = client.get('/kameralar')
    assert 'Sera 1' in liste.text

    r = client.post('/analiz/kamera', data={'kamera_id': '1'}, follow_redirects=True)
    assert r.status_code == 200
    assert 'Analiz #' in r.text


def test_tespit_yok_durumu(client, monkeypatch):
    monkeypatch.setattr(main, 'detector', SahteDetector(kutular=[]))
    r = client.post('/analiz/dosya',
                    files={'dosyalar': ('bos.jpg', b'x', 'image/jpeg')},
                    follow_redirects=True)
    assert 'Tespit yok' in r.text
    assert 'background' in r.text        # sağlıklı örnek açıklaması


def test_inceleme_kuyrugu_ve_disa_aktarim(client):
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    q = client.get('/inceleme')
    assert q.status_code == 200
    assert 'İnceleme Kuyruğu' in q.text

    r = client.post('/inceleme/disa-aktar', follow_redirects=True)
    assert r.status_code == 200
    disa = list(Path(config.EXPORT_DIR).glob('inceleme_*'))
    assert disa, 'dışa aktarım klasörü oluşmadı'
    etiketler = list((disa[0] / 'labels').glob('*.txt'))
    assert etiketler, 'ön-etiket dosyası yok'
    satir = etiketler[0].read_text(encoding='utf-8').strip().split('\n')[0]
    assert satir.split()[0] == '3'       # sinif_id korunmuş olmalı
    assert len(satir.split()) == 5       # YOLO formatı: cls x y w h


def test_gecmis_ve_panel(client):
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    g = client.get('/gecmis')
    assert g.status_code == 200
    assert 'Geçmiş Analizler' in g.text

    p = client.get('/panel')
    assert p.status_code == 200
    assert 'toplam analiz' in p.text


def test_gecmis_sinif_filtresi(client):
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    r = client.get('/gecmis?sinif=Gray Mold')
    assert r.status_code == 200
    bos = client.get('/gecmis?sinif=Olmayan Sinif')
    assert 'Kayıt bulunamadı' in bos.text


def test_model_yoksa_uyari(client, monkeypatch):
    monkeypatch.setattr(main, 'detector', SahteDetector(hazir=False))
    r = client.get('/')
    assert 'Model bulunamadı' in r.text
    y = client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')})
    assert y.status_code == 400
