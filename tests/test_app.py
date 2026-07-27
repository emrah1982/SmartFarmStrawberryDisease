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
    assert 'Görüntü Analizi' in r.text
    assert 'Fotoğraf Çek' in r.text and 'Video Çek' in r.text


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


def test_gorseller_tarayicidan_acilabilir(client):
    """Kaydedilen yollar URL olarak çalışmalı.

    Windows'ta relative_to() ters bölü üretir ('results\\x.jpg'); bu URL'de
    klasör ayracı sayılmadığı için görseller 404 verir. Yollar as_posix() ile
    saklanmalı — bu test o regresyonu yakalar.
    """
    r = client.post('/analiz/dosya',
                    files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                    follow_redirects=True)
    assert r.status_code == 200

    import re
    yollar = re.findall(r'src="(/media/[^"]+)"', r.text)
    assert yollar, 'sayfada görsel bağlantısı yok'
    for y in yollar:
        assert '\\' not in y, f'yolda ters bölü var: {y}'
        assert client.get(y).status_code == 200, f'görsel açılamadı: {y}'


# ─────────────────────────────────── Üretici → Sera → Kamera hiyerarşisi
def _isletme_kur(client, uretici='Ahmet Yılmaz', sera='Sera 1'):
    client.post('/ureticiler/ekle', data={'ad': uretici, 'telefon': '0532'},
                follow_redirects=True)
    from app.database import SessionLocal, Uretici
    with SessionLocal() as db:
        u = db.query(Uretici).filter(Uretici.ad == uretici).first()
    client.post('/seralar/ekle',
                data={'uretici_id': str(u.id), 'ad': sera, 'konum': 'Kuzey'},
                follow_redirects=True)
    from app.database import Sera
    with SessionLocal() as db:
        srr = db.query(Sera).filter(Sera.ad == sera, Sera.uretici_id == u.id).first()
    return u.id, srr.id


def test_uretici_sera_ekleme(client):
    uid, sid = _isletme_kur(client, 'Mehmet Demir', 'Batı Serası')
    r = client.get('/isletmeler')
    assert 'Mehmet Demir' in r.text
    assert 'Batı Serası' in r.text


def test_kamera_seraya_baglanir(client):
    uid, sid = _isletme_kur(client, 'Veli Kaya', 'Sera A')
    client.post('/kameralar/ekle',
                data={'ad': 'Giriş', 'url': 'rtsp://1.2.3.4/s',
                      'konum': '3. sıra', 'sera_id': str(sid)},
                follow_redirects=True)
    liste = client.get('/kameralar')
    # Kamera hangi serada, sera kime ait — hepsi görünmeli
    assert 'Veli Kaya — Sera A' in liste.text
    assert 'Giriş' in liste.text


def test_kamera_analizi_sera_bilgisi_tasir(client):
    uid, sid = _isletme_kur(client, 'Ayşe Şahin', 'Sera 2')
    client.post('/kameralar/ekle',
                data={'ad': 'Kuzey kamera', 'url': 'rtsp://5.6.7.8/s', 'sera_id': str(sid)},
                follow_redirects=True)
    from app.database import SessionLocal, Kamera
    with SessionLocal() as db:
        kid = db.query(Kamera).filter(Kamera.ad == 'Kuzey kamera').first().id

    r = client.post('/analiz/kamera', data={'kamera_id': str(kid)}, follow_redirects=True)
    assert r.status_code == 200
    assert 'Ayşe Şahin — Sera 2 / Kuzey kamera' in r.text

    with SessionLocal() as db:
        from app.database import Analiz
        a = db.query(Analiz).order_by(Analiz.id.desc()).first()
        assert a.sera_id == sid, 'kamera analizinde sera_id kameradan türetilmeli'


def test_telefon_yuklemesinde_sera_secilebilir(client):
    uid, sid = _isletme_kur(client, 'Fatma Öz', 'Sera 3')
    r = client.post('/analiz/dosya',
                    data={'sera_id': str(sid)},
                    files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                    follow_redirects=True)
    assert 'Fatma Öz — Sera 3' in r.text


def test_gecmis_sera_filtresi(client):
    uid1, sid1 = _isletme_kur(client, 'Üretici A', 'A-Sera')
    uid2, sid2 = _isletme_kur(client, 'Üretici B', 'B-Sera')
    client.post('/analiz/dosya', data={'sera_id': str(sid1)},
                files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')}, follow_redirects=True)

    sadece_a = client.get(f'/gecmis?sera_id={sid1}')
    assert 'A-Sera' in sadece_a.text
    sadece_b = client.get(f'/gecmis?sera_id={sid2}')
    assert 'Kayıt bulunamadı' in sadece_b.text

    uretici_a = client.get(f'/gecmis?uretici_id={uid1}')
    assert 'A-Sera' in uretici_a.text
    uretici_b = client.get(f'/gecmis?uretici_id={uid2}')
    assert 'Kayıt bulunamadı' in uretici_b.text


def test_panel_sera_bazli_ozet(client):
    uid, sid = _isletme_kur(client, 'Panel Üretici', 'Panel Sera')
    client.post('/analiz/dosya', data={'sera_id': str(sid)},
                files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')}, follow_redirects=True)
    p = client.get('/panel')
    assert 'Sera Bazlı Özet' in p.text
    assert 'Panel Üretici — Panel Sera' in p.text


# ─────────────────────────────────────────────── görüntü kalitesi (bulanıklık)
def test_bulaniklik_olcumu():
    """Laplacian varyansı keskin ve bulanık görüntüyü ayırt etmeli."""
    import numpy as np, cv2
    from app.detector import keskinlik_olc
    rng = np.random.default_rng(0)
    keskin = (rng.random((300, 300, 3)) * 255).astype('uint8')
    bulanik = cv2.GaussianBlur(keskin, (31, 31), 0)
    assert keskinlik_olc(keskin) > keskinlik_olc(bulanik) * 100


def test_kalite_notu_kullaniciya_gosterilir(client, monkeypatch):
    """Bulanık video uyarısı sonuç sayfasında görünmeli."""
    class BulanikDetector(SahteDetector):
        def video(self, kaynak, cikti):
            s = self._sonuc(cikti, kare=3)
            s.bulanik_kare = 7
            s.kalite_notu = ('10 karenin 7 tanesi bulanık olduğu için atlandı. '
                             'Yürürken çekimde hareket bulanıklığı olağandır.')
            return s
    monkeypatch.setattr(main, 'detector', BulanikDetector())
    r = client.post('/analiz/dosya',
                    files={'dosyalar': ('yururken.mp4', b'x', 'video/mp4')},
                    follow_redirects=True)
    assert 'Görüntü kalitesi' in r.text
    assert 'bulanık olduğu için atlandı' in r.text


def test_kalite_bilgisi_veritabanina_yazilir(client, monkeypatch):
    class BulanikDetector(SahteDetector):
        def goruntu(self, kaynak, cikti):
            s = self._sonuc(cikti)
            s.keskinlik = 12.5
            s.kalite_notu = 'Görüntü bulanık'
            return s
    monkeypatch.setattr(main, 'detector', BulanikDetector())
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    from app.database import Analiz, SessionLocal
    with SessionLocal() as db:
        a = db.query(Analiz).order_by(Analiz.id.desc()).first()
        assert a.keskinlik == 12.5
        assert 'bulanık' in a.kalite_notu


def test_cekim_rehberi_gosteriliyor(client):
    """Kullanıcı görüntü kalitesi konusunda bilgilendirilmeli."""
    r = client.get('/')
    assert 'İyi görüntü için' in r.text
    assert 'Yürürken video' in r.text
    assert 'Bulanık kareler otomatik atlanır' in r.text
    # Cihaz farkı açıklaması (dizüstünde galeri açılması kafa karıştırmasın)
    assert 'kamera uygulamasını açar' in r.text
