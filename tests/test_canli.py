"""Canlı tespit modülü testleri.

Kamera gerektirmez: karar mantığı saf fonksiyon olduğu için doğrudan, uç
noktalar ise sahte dedektör + sahte JPEG ile sınanır.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.detector import Kutu, Sonuc
from app.moduller.canli import servis


# ───────────────────────────────────────────────── otomatik kayıt kararı
def _kutu(sinif_id=3, guven=0.9):
    return Kutu(sinif_id, 'Gray Mold', guven, 0.5, 0.5, 0.2, 0.2)


def test_tek_karelik_tespit_kaydedilmez():
    """Gürültü kayda geçmemeli: kararlılık şartı sağlanmadan kayıt açılmaz."""
    k = servis.KayitKarari(kararlilik_kare=3, guven_esigi=0.6, bekleme_sn=10)
    assert k.degerlendir([_kutu()], simdi=0) is None
    assert k.degerlendir([_kutu()], simdi=1) is None
    assert k.degerlendir([_kutu()], simdi=2) is not None      # 3. karede kaydeder


def test_dusuk_guven_sayilmaz():
    k = servis.KayitKarari(kararlilik_kare=2, guven_esigi=0.6, bekleme_sn=10)
    for t in range(5):
        assert k.degerlendir([_kutu(guven=0.4)], simdi=t) is None


def test_ardisik_olmayan_kareler_sayaci_sifirlar():
    """'Üst üste' şartı gerçekten ardışık kareleri ifade etmeli."""
    k = servis.KayitKarari(kararlilik_kare=3, guven_esigi=0.6, bekleme_sn=10)
    k.degerlendir([_kutu()], simdi=0)
    k.degerlendir([], simdi=1)                                # bulgu kayboldu
    k.degerlendir([_kutu()], simdi=2)
    assert k.degerlendir([_kutu()], simdi=3) is None          # sayaç sıfırlandı


def test_bekleme_suresi_ayni_bulguyu_tekrar_kaydetmez():
    k = servis.KayitKarari(kararlilik_kare=1, guven_esigi=0.6, bekleme_sn=20)
    assert k.degerlendir([_kutu()], simdi=100) is not None
    assert k.degerlendir([_kutu()], simdi=110) is None        # bekleme dolmadı
    assert k.degerlendir([_kutu()], simdi=125) is not None    # doldu


def test_farkli_sinif_kendi_sayacini_tutar():
    k = servis.KayitKarari(kararlilik_kare=2, guven_esigi=0.6, bekleme_sn=10)
    k.degerlendir([_kutu(sinif_id=3)], simdi=0)
    assert k.degerlendir([_kutu(sinif_id=5)], simdi=1) is None
    assert k.degerlendir([_kutu(sinif_id=5)], simdi=2) is not None


# ───────────────────────────────────────────────────────── uç noktalar
class SahteCanliDetector:
    hazir = True

    def __init__(self, kutular=None):
        self.kutular = kutular if kutular is not None else [_kutu()]
        self.cagri = 0

    def kare(self, frame, imgsz=None):
        self.cagri += 1
        return Sonuc(kutular=list(self.kutular), islenen_kare=1, sure_ms=7)


def _jpeg(genislik=64, yukseklik=48, gurultulu=True):
    """Gerçek JPEG üretir; gürültülü olan bulanıklık eşiğini geçer."""
    import cv2
    if gurultulu:
        rng = np.random.default_rng(0)
        kare = rng.integers(0, 255, (yukseklik, genislik, 3), dtype=np.uint8)
    else:
        kare = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)   # düz = bulanık
    return cv2.imencode('.jpg', kare)[1].tobytes()


@pytest.fixture
def client(monkeypatch):
    # Depolama ve veritabanı conftest.py tarafından geçici dizine yönlendirildi;
    # burada config'e dokunmak /media bağlamasıyla tutarsızlık yaratırdı.
    from app import main
    monkeypatch.setattr(main, 'detector', SahteCanliDetector())
    with TestClient(main.app) as c:
        yield c


def test_canli_sayfasi_acilir(client):
    r = client.get('/canli')
    assert r.status_code == 200
    assert 'Canlı Tespit' in r.text
    assert '/statik/canli/izle.js' in r.text       # bileşenler yükleniyor mu


def test_modul_statik_dosyalari_sunuluyor(client):
    for dosya in ('kamera.js', 'akis.js', 'cizim.js', 'izle.js'):
        r = client.get(f'/statik/canli/{dosya}')
        assert r.status_code == 200, dosya


def test_websocket_kare_gonderince_kutu_doner(client):
    with client.websocket_connect('/canli/ws') as ws:
        ws.send_bytes(_jpeg())
        yanit = ws.receive_json()
    assert yanit['tip'] == 'sonuc'
    assert yanit['kutular'][0]['ad'] == 'Gray Mold'
    assert 0 <= yanit['kutular'][0]['x'] <= 1      # normalize koordinat
    assert yanit['kayit_id'] is None               # tek kare: henüz kararlı değil


def test_websocket_elle_kaydet_kayit_acar(client):
    with client.websocket_connect('/canli/ws') as ws:
        ws.send_json({'tip': 'kaydet'})
        ws.send_bytes(_jpeg())
        yanit = ws.receive_json()
    assert yanit['kayit_id'], 'elle kaydet isteği kayıt açmalı'
    assert yanit['kayit_tipi'] == 'elle'

    # Kayıt çekirdekle aynı biçimde: geçmişte ve kayıt sayfasında görünür
    assert client.get(f"/kayit/{yanit['kayit_id']}").status_code == 200


def test_websocket_kararli_bulgu_otomatik_kaydedilir(client):
    from app.moduller.canli import ayarlar
    with client.websocket_connect('/canli/ws') as ws:
        kayit = None
        for _ in range(ayarlar.KARARLILIK_KARE):
            ws.send_bytes(_jpeg())
            kayit = ws.receive_json()
    assert kayit['kayit_id'], 'üst üste görülen bulgu otomatik kaydedilmeli'
    assert kayit['kayit_tipi'] == 'otomatik'


def test_bulanik_kare_modele_verilmez(client):
    from app import main
    from app.moduller.canli import ayarlar
    if not ayarlar.BULANIKLIK_ESIGI:
        pytest.skip('bulanıklık kontrolü kapalı')
    with client.websocket_connect('/canli/ws') as ws:
        ws.send_bytes(_jpeg(gurultulu=False))       # düz görüntü = keskinlik ~0
        yanit = ws.receive_json()
    assert yanit['bulanik'] is True
    assert yanit['kutular'] == []
    assert main.detector.cagri == 0, 'bulanık kare modele gönderilmemeli'


def test_bozuk_kare_akisi_dusurmez(client):
    with client.websocket_connect('/canli/ws') as ws:
        ws.send_bytes(b'bu-jpeg-degil')
        hata = ws.receive_json()
        assert hata['tip'] == 'hata'
        ws.send_bytes(_jpeg())                      # bağlantı hâlâ ayakta
        assert ws.receive_json()['tip'] == 'sonuc'


def test_rest_yedegi_ayni_yaniti_verir(client):
    """WebSocket engelli ağlarda kullanılan yol aynı sonucu üretmeli."""
    r = client.post('/canli/kare',
                    files={'kare': ('k.jpg', _jpeg(), 'image/jpeg')},
                    data={'oturum': 'test1'})
    assert r.status_code == 200
    veri = r.json()
    assert veri['tip'] == 'sonuc' and veri['kutular']


def test_rest_yedegi_oturum_bazli_kararlilik_tutar(client):
    from app.moduller.canli import ayarlar
    son = None
    for _ in range(ayarlar.KARARLILIK_KARE):
        son = client.post('/canli/kare',
                          files={'kare': ('k.jpg', _jpeg(), 'image/jpeg')},
                          data={'oturum': 'test2'}).json()
    assert son['kayit_id'], 'REST yedeğinde de otomatik kayıt çalışmalı'


def test_canli_menude_ana_grupta(client):
    r = client.get('/')
    assert '/canli' in r.text
