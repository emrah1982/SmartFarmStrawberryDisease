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
    paket = Path(config.INCELEME_DIR)
    assert paket.exists(), 'inceleme paketi oluşmadı'
    assert (paket / 'data.yaml').exists()
    etiketler = list((paket / 'labels').glob('*.txt'))
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
    assert 'benzersiz görüntü' in p.text


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


# ────────────────────────────────────────── elle etiketleme (Roboflow benzeri)
def test_etiketleme_sayfasi_acilir(client):
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    r = client.get('/kayit/1/etiketle')
    assert r.status_code == 200
    assert 'Etiketleme' in r.text
    # Sınıf listesi eğitimdeki ID düzeniyle gelmeli
    assert 'Angular Leafspot' in r.text and 'strawberry_unripe' in r.text
    assert 'tuval' in r.text          # canvas


def test_etiketler_kaydedilir_ve_tespitler_degisir(client):
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    from app.database import Analiz, SessionLocal
    with SessionLocal() as db:
        aid = db.query(Analiz).order_by(Analiz.id.desc()).first().id

    r = client.post(f'/api/kayit/{aid}/etiketler', json={'kutular': [
        {'sinif_id': 0, 'x': 0.5, 'y': 0.5, 'w': 0.2, 'h': 0.2},
        {'sinif_id': 4, 'x': 0.2, 'y': 0.3, 'w': 0.1, 'h': 0.1},
    ]})
    assert r.status_code == 200 and r.json()['kutu'] == 2

    with SessionLocal() as db:
        a = db.get(Analiz, aid)
        assert a.tespit_sayisi == 2
        assert a.elle_etiketlendi and a.incelendi
        adlar = sorted(t.sinif_adi for t in a.tespitler)
        assert adlar == ['Angular Leafspot', 'Leaf Spot']
        assert all(t.guven == 1.0 for t in a.tespitler)


def test_gecersiz_etiket_reddedilir(client):
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    from app.database import Analiz, SessionLocal
    with SessionLocal() as db:
        aid = db.query(Analiz).order_by(Analiz.id.desc()).first().id

    kotu_sinif = client.post(f'/api/kayit/{aid}/etiketler',
                             json={'kutular': [{'sinif_id': 99, 'x': .5, 'y': .5, 'w': .1, 'h': .1}]})
    assert kotu_sinif.status_code == 400

    kotu_koord = client.post(f'/api/kayit/{aid}/etiketler',
                             json={'kutular': [{'sinif_id': 0, 'x': 1.5, 'y': .5, 'w': .1, 'h': .1}]})
    assert kotu_koord.status_code == 400


def test_bos_etiket_background_ornegi_olur(client):
    """Hastalık yoksa hiç kutu bırakılmaz — bu geçerli bir eğitim örneğidir."""
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                follow_redirects=True)
    from app.database import Analiz, SessionLocal
    with SessionLocal() as db:
        aid = db.query(Analiz).order_by(Analiz.id.desc()).first().id
    r = client.post(f'/api/kayit/{aid}/etiketler', json={'kutular': []})
    assert r.status_code == 200
    with SessionLocal() as db:
        a = db.get(Analiz, aid)
        assert a.tespit_sayisi == 0 and a.elle_etiketlendi


def test_egitim_formatinda_disa_aktarim(client):
    """Çıktı merge_datasets.py'nin beklediği yapıda olmalı: images/, labels/, data.yaml"""
    import yaml as _y
    uid, sid = _isletme_kur(client, 'Etiket Üretici', 'E-Sera')
    client.post('/analiz/dosya', data={'sera_id': str(sid)},
                files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')}, follow_redirects=True)
    from app.database import Analiz, SessionLocal
    with SessionLocal() as db:
        aid = db.query(Analiz).order_by(Analiz.id.desc()).first().id
    client.post(f'/api/kayit/{aid}/etiketler',
                json={'kutular': [{'sinif_id': 3, 'x': .4, 'y': .4, 'w': .2, 'h': .2}]})

    r = client.post('/inceleme/egitime-hazirla', follow_redirects=True)
    assert r.status_code == 200

    # Tek birikimli klasör: her aktarımda yeni klasör AÇILMAMALI
    d = Path(config.EGITIM_DIR)
    assert d.exists(), 'egitim_verisi klasörü oluşmadı'
    assert not list(Path(config.STORAGE_DIR).glob('exports/egitim_*')),         'tarihli egitim_* klasörü açılmamalı — tek havuz kullanılıyor'
    assert (d / 'data.yaml').exists(), 'merge_datasets.py data.yaml şart koşar'
    cfg = _y.safe_load((d / 'data.yaml').read_text(encoding='utf-8'))
    assert cfg['nc'] == 10 and cfg['names'][3] == 'Gray Mold'

    goruntuler = list((d / 'images').glob('*'))
    etiketler = list((d / 'labels').glob('*.txt'))
    assert goruntuler and etiketler
    # Dosya adı sera ile başlamalı (grup bazlı split için).
    # Aktarım bekleyen TÜM etiketli kayıtları alır, bu yüzden aramak gerekir.
    bizimki = [g for g in goruntuler if g.name.startswith('E-Sera_')]
    assert bizimki, f'sera adıyla başlayan dosya yok: {[g.name for g in goruntuler]}'
    satir = (d / 'labels' / f'{bizimki[0].stem}.txt').read_text(encoding='utf-8').strip()
    assert satir.split()[0] == '3' and len(satir.split()) == 5

    # Aynı kayıt iki kez aktarılmamalı
    tekrar = client.post('/inceleme/egitime-hazirla')
    assert tekrar.status_code == 400


# ─────────────────────────── aynı görüntünün birden çok kez etiketlenmesi
def test_ayni_goruntu_havuzda_kopya_olusturmaz(client):
    """Aynı fotoğraf iki kez yüklenip iki kez etiketlenirse havuzda TEK dosya olmalı.

    Aksi halde eğitim verisinde aynı görüntü iki kez, üstelik çelişen
    etiketlerle bulunur; split sırasında train/val'e birden düşerek
    veri sızıntısına da yol açar.
    """
    from app.database import Analiz, SessionLocal
    icerik = b'AYNI-GORUNTU-BAYTLARI'

    # Aynı içerik, iki ayrı yükleme
    client.post('/analiz/dosya', files={'dosyalar': ('ilk.jpg', icerik, 'image/jpeg')},
                follow_redirects=True)
    client.post('/analiz/dosya', files={'dosyalar': ('ikinci.jpg', icerik, 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        kayitlar = (db.query(Analiz).filter(Analiz.kaynak_ad.in_(['ilk.jpg', 'ikinci.jpg']))
                    .order_by(Analiz.id).all())
        assert len(kayitlar) == 2
        assert kayitlar[0].dosya_hash == kayitlar[1].dosya_hash, 'hash aynı olmalı'
        ilk_id, ikinci_id = kayitlar[0].id, kayitlar[1].id

    # İkisini de farklı şekilde etiketle
    client.post(f'/api/kayit/{ilk_id}/etiketler',
                json={'kutular': [{'sinif_id': 0, 'x': .3, 'y': .3, 'w': .1, 'h': .1}]})
    client.post(f'/api/kayit/{ikinci_id}/etiketler',
                json={'kutular': [{'sinif_id': 4, 'x': .6, 'y': .6, 'w': .2, 'h': .2},
                                  {'sinif_id': 4, 'x': .2, 'y': .2, 'w': .1, 'h': .1}]})

    client.post('/inceleme/egitime-hazirla', follow_redirects=True)

    havuz = Path(config.EGITIM_DIR) / 'images'
    ayni_olanlar = [f for f in havuz.glob('*') if f.read_bytes() == icerik]
    assert len(ayni_olanlar) == 1, f'aynı görüntü {len(ayni_olanlar)} kez var: {ayni_olanlar}'

    # EN SON etiketlenen sürüm geçerli olmalı (2 kutu, sınıf 4)
    etiket = (Path(config.EGITIM_DIR) / 'labels' / f'{ayni_olanlar[0].stem}.txt')
    satirlar = [l for l in etiket.read_text(encoding='utf-8').strip().splitlines() if l]
    assert len(satirlar) == 2, f'son etiket geçerli olmalı, bulunan: {satirlar}'
    assert all(l.split()[0] == '4' for l in satirlar)


def test_tekrar_yukleme_kullaniciya_bildirilir(client):
    icerik = b'TEKRAR-EDEN-GORUNTU'
    client.post('/analiz/dosya', files={'dosyalar': ('a.jpg', icerik, 'image/jpeg')},
                follow_redirects=True)
    r = client.post('/analiz/dosya', files={'dosyalar': ('b.jpg', icerik, 'image/jpeg')},
                    follow_redirects=True)
    assert 'daha önce de analiz edilmiş' in r.text


def test_paket_eskimis_kalmaz(client):
    """Ham paket her aktarımda yeniden üretilmeli — eski etiket kalmamalı.

    Kullanıcının yaşadığı durum: kayıt aktarıldıktan SONRA etiketlenince
    klasördeki etiket dosyası veritabanıyla çelişiyordu (1 kutu vs 17 kutu).
    """
    from app.database import Analiz, SessionLocal
    client.post('/analiz/dosya', files={'dosyalar': ('p.jpg', b'PAKET-TESTI', 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        a = db.query(Analiz).filter(Analiz.kaynak_ad == 'p.jpg').first()
        aid, h = a.id, a.dosya_hash

    client.post('/inceleme/disa-aktar', follow_redirects=True)
    etiket = Path(config.INCELEME_DIR) / 'labels' / f'atanmamis_{h}.txt'
    assert etiket.exists(), 'dosya adı <sera>_<hash> olmalı'
    ilk = len([x for x in etiket.read_text(encoding='utf-8').splitlines() if x])

    # Kayıt etiketlenip kuyruktan çıkınca paket yeniden üretilmeli
    client.post(f'/api/kayit/{aid}/etiketler', json={'kutular': [
        {'sinif_id': 0, 'x': .3, 'y': .3, 'w': .1, 'h': .1},
        {'sinif_id': 0, 'x': .6, 'y': .6, 'w': .1, 'h': .1},
        {'sinif_id': 4, 'x': .8, 'y': .2, 'w': .1, 'h': .1},
    ]})
    client.post('/analiz/dosya', files={'dosyalar': ('q.jpg', b'IKINCI', 'image/jpeg')},
                follow_redirects=True)
    client.post('/inceleme/disa-aktar', follow_redirects=True)

    # Etiketlenen kayıt kuyruktan çıktığı için paketten de kalkmalı (eskimiş kalmaz)
    assert not etiket.exists(), 'etiketlenmiş kayıt pakette eskimiş halde kalmamalı'
    assert ilk >= 0


# ─────────────────────────────── etiketlenmiş kayıtları görüntüleme
def test_etiketlenenler_sayfasi(client):
    """Kullanıcı ne etiketlediğini görebilmeli."""
    from app.database import Analiz, SessionLocal
    client.post('/analiz/dosya', files={'dosyalar': ('gorunur.jpg', b'GORUNUR', 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        aid = db.query(Analiz).filter(Analiz.kaynak_ad == 'gorunur.jpg').first().id

    bos = client.get('/etiketlenenler')
    assert bos.status_code == 200

    client.post(f'/api/kayit/{aid}/etiketler', json={'kutular': [
        {'sinif_id': 0, 'x': .3, 'y': .3, 'w': .1, 'h': .1},
        {'sinif_id': 0, 'x': .6, 'y': .6, 'w': .1, 'h': .1},
    ]})

    r = client.get('/etiketlenenler')
    assert r.status_code == 200
    assert f'/kayit/{aid}/etiket-onizleme.jpg' in r.text, 'önizleme görseli sayfada olmalı'
    assert 'Angular Leafspot' in r.text          # sınıf dağılımı
    assert 'toplam kutu' in r.text


def test_etiket_onizlemesi_dinamik(client):
    """Önizleme dosyaya yazılmamalı; her istekte veritabanından üretilmeli."""
    import cv2, numpy as np
    from app.database import Analiz, SessionLocal

    # Gerçek bir JPEG gerekiyor (cv2 okuyabilsin)
    kare = np.full((200, 300, 3), 200, dtype='uint8')
    ok, tampon = cv2.imencode('.jpg', kare)
    assert ok
    client.post('/analiz/dosya',
                files={'dosyalar': ('onizleme.jpg', tampon.tobytes(), 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        aid = db.query(Analiz).filter(Analiz.kaynak_ad == 'onizleme.jpg').first().id

    client.post(f'/api/kayit/{aid}/etiketler',
                json={'kutular': [{'sinif_id': 3, 'x': .5, 'y': .5, 'w': .4, 'h': .4}]})
    ilk = client.get(f'/kayit/{aid}/etiket-onizleme.jpg')
    assert ilk.status_code == 200 and ilk.headers['content-type'] == 'image/jpeg'
    assert ilk.headers.get('cache-control') == 'no-store'

    # Etiket değişince önizleme de değişmeli (dosya önbelleği yok)
    client.post(f'/api/kayit/{aid}/etiketler', json={'kutular': []})
    sonra = client.get(f'/kayit/{aid}/etiket-onizleme.jpg')
    assert sonra.status_code == 200
    assert sonra.content != ilk.content, 'önizleme güncel veritabanından üretilmeli'


# ───────────────────────────────────────────── panel istatistik doğruluğu
def test_panel_ayni_goruntuyu_iki_kez_saymaz(client):
    """Aynı fotoğraf iki kez yüklenirse istatistikte BİR görüntü sayılmalı.

    Kullanıcının fark ettiği durum: tek bir resmin kutuları, resim tekrar
    yüklendiği için grafikte iki kez görünüyordu.
    """
    from app.database import Analiz, SessionLocal
    icerik = b'PANEL-TEKRAR-TESTI'
    client.post('/analiz/dosya', files={'dosyalar': ('p1.jpg', icerik, 'image/jpeg')},
                follow_redirects=True)
    client.post('/analiz/dosya', files={'dosyalar': ('p2.jpg', icerik, 'image/jpeg')},
                follow_redirects=True)

    with SessionLocal() as db:
        kayitlar = db.query(Analiz).filter(Analiz.kaynak_ad.in_(['p1.jpg', 'p2.jpg'])).all()
        assert len(kayitlar) == 2
        idler = sorted(a.id for a in kayitlar)

    # İkisini de etiketle: ilkine 1, ikincisine 3 kutu
    client.post(f'/api/kayit/{idler[0]}/etiketler',
                json={'kutular': [{'sinif_id': 2, 'x': .5, 'y': .5, 'w': .1, 'h': .1}]})
    client.post(f'/api/kayit/{idler[1]}/etiketler', json={'kutular': [
        {'sinif_id': 2, 'x': .2, 'y': .2, 'w': .1, 'h': .1},
        {'sinif_id': 2, 'x': .4, 'y': .4, 'w': .1, 'h': .1},
        {'sinif_id': 2, 'x': .6, 'y': .6, 'w': .1, 'h': .1},
    ]})

    r = client.get('/panel')
    assert r.status_code == 200
    # Tekrar uyarısı görünmeli
    assert 'daha önce yüklenmiş bir görüntüye' in r.text
    # Blossom Blight yalnızca SON kayıttan (3 kutu) sayılmalı, 1+3=4 değil
    import re
    satir = re.search(r'Blossom Blight.*?class="sayi">(\d+)<', r.text, re.S)
    assert satir, 'sınıf satırı bulunamadı'
    assert satir.group(1) == '3', f'beklenen 3, bulunan {satir.group(1)}'


def test_panel_model_ve_elle_ayrimi(client):
    """Model tespitleri ile elle etiketler ayrı raporlanmalı."""
    from app.database import Analiz, SessionLocal
    client.post('/analiz/dosya', files={'dosyalar': ('m.jpg', b'MODEL-KAYDI', 'image/jpeg')},
                follow_redirects=True)   # SahteDetector: Gray Mold + Leaf Spot
    client.post('/analiz/dosya', files={'dosyalar': ('e.jpg', b'ELLE-KAYDI', 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        eid = db.query(Analiz).filter(Analiz.kaynak_ad == 'e.jpg').first().id
    client.post(f'/api/kayit/{eid}/etiketler',
                json={'kutular': [{'sinif_id': 1, 'x': .5, 'y': .5, 'w': .2, 'h': .2}]})

    r = client.get('/panel')
    assert 'Model Tespitleri' in r.text and 'Elle Etiketlenenler' in r.text
    model_bolum = r.text.split('Model Tespitleri')[1].split('Elle Etiketlenenler')[0]
    elle_bolum = r.text.split('Elle Etiketlenenler')[1]
    assert 'Gray Mold' in model_bolum          # model tahmini
    assert 'Anthracnose' in elle_bolum         # elle etiket
    assert 'Anthracnose' not in model_bolum, 'elle etiket model bölümünde görünmemeli'


# ─────────────────────────────────────────────────── kalıcı silme
def test_kayit_kalici_silinir(client):
    """Kayıt silinince veritabanı satırı VE diskteki tüm kopyaları gitmeli."""
    from app.database import Analiz, SessionLocal
    client.post('/analiz/dosya', files={'dosyalar': ('silinecek.jpg', b'SIL-BENI', 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        a = db.query(Analiz).filter(Analiz.kaynak_ad == 'silinecek.jpg').first()
        aid, dosya, sonuc, h = a.id, a.dosya_yolu, a.sonuc_yolu, a.dosya_hash

    client.post(f'/api/kayit/{aid}/etiketler',
                json={'kutular': [{'sinif_id': 0, 'x': .5, 'y': .5, 'w': .2, 'h': .2}]})
    client.post('/inceleme/egitime-hazirla', follow_redirects=True)

    havuz_img = list((Path(config.EGITIM_DIR) / 'images').glob(f'atanmamis_{h}.*'))
    havuz_lbl = Path(config.EGITIM_DIR) / 'labels' / f'atanmamis_{h}.txt'
    assert havuz_img and havuz_lbl.exists(), 'önce havuzda olmalı'
    assert (Path(config.STORAGE_DIR) / dosya).exists()

    r = client.post(f'/kayit/{aid}/sil', follow_redirects=True)
    assert r.status_code == 200

    with SessionLocal() as db:
        assert db.get(Analiz, aid) is None, 'veritabanı kaydı silinmeli'
        from app.database import Tespit
        assert db.query(Tespit).filter(Tespit.analiz_id == aid).count() == 0,             'tespitler de silinmeli'
    assert not (Path(config.STORAGE_DIR) / dosya).exists(), 'orijinal görüntü silinmeli'
    if sonuc:
        assert not (Path(config.STORAGE_DIR) / sonuc).exists(), 'sonuç görseli silinmeli'
    assert not any(f.exists() for f in havuz_img), 'havuzdaki görüntü silinmeli'
    assert not havuz_lbl.exists(), 'havuzdaki etiket silinmeli'


def test_silme_ayni_goruntunun_diger_kaydini_korur(client):
    """Aynı görüntünün etiketli başka kaydı varsa havuz dosyası KALMALI."""
    from app.database import Analiz, SessionLocal
    icerik = b'PAYLASILAN-GORUNTU'
    client.post('/analiz/dosya', files={'dosyalar': ('x1.jpg', icerik, 'image/jpeg')},
                follow_redirects=True)
    client.post('/analiz/dosya', files={'dosyalar': ('x2.jpg', icerik, 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        kayitlar = sorted(db.query(Analiz).filter(Analiz.kaynak_ad.in_(['x1.jpg', 'x2.jpg'])).all(),
                          key=lambda a: a.id)
        id1, id2, h = kayitlar[0].id, kayitlar[1].id, kayitlar[0].dosya_hash

    for i in (id1, id2):
        client.post(f'/api/kayit/{i}/etiketler',
                    json={'kutular': [{'sinif_id': 0, 'x': .5, 'y': .5, 'w': .1, 'h': .1}]})
    client.post('/inceleme/egitime-hazirla', follow_redirects=True)

    havuz_lbl = Path(config.EGITIM_DIR) / 'labels' / f'atanmamis_{h}.txt'
    assert havuz_lbl.exists()

    client.post(f'/kayit/{id1}/sil', follow_redirects=True)
    assert havuz_lbl.exists(), 'diğer kayıt hâlâ etiketli olduğu için havuz dosyası kalmalı'

    client.post(f'/kayit/{id2}/sil', follow_redirects=True)
    assert not havuz_lbl.exists(), 'son kayıt da silinince havuz dosyası gitmeli'


def test_silme_onay_uyarisi_arayuzde(client):
    """Kullanıcı ne silineceğini görmeden onaylamamalı."""
    from app.database import Analiz, SessionLocal
    client.post('/analiz/dosya', files={'dosyalar': ('u.jpg', b'UYARI', 'image/jpeg')},
                follow_redirects=True)
    with SessionLocal() as db:
        aid = db.query(Analiz).filter(Analiz.kaynak_ad == 'u.jpg').first().id
    client.post(f'/api/kayit/{aid}/etiketler',
                json={'kutular': [{'sinif_id': 0, 'x': .5, 'y': .5, 'w': .1, 'h': .1}]})

    r = client.get('/etiketlenenler')
    assert 'silOnayi' in r.text and 'GERİ ALINAMAZ' in r.text
    assert 'Kalıcı sil' in r.text
