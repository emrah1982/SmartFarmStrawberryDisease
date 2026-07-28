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
    # kayit_tipi hangi kuralın kaydettiğini söyler (kaynak_ad'a da yazılır)
    assert kayit['kayit_tipi'] == 'akilli'


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


# ─────────────────────────────────────────────────────── sertifika kurulumu
def test_sertifika_sayfasi_acilir(client):
    """Telefonda çıkan güven uyarısının çözümü uygulama içinden anlatılmalı."""
    r = client.get('/canli/sertifika')
    assert r.status_code == 200
    assert 'localhost' in r.text                     # bilgisayarda gerek yok bilgisi


def test_sertifika_yoksa_404(client, monkeypatch):
    from app import config
    monkeypatch.setattr(config, 'SSL_CERT', '')
    assert client.get('/canli/sertifika.crt').status_code == 404


def test_sertifika_indirilebilir(client, monkeypatch, tmp_path):
    sahte = tmp_path / 'sunucu.crt'
    sahte.write_text('-----BEGIN CERTIFICATE-----\nsahte\n-----END CERTIFICATE-----')
    from app import config
    monkeypatch.setattr(config, 'SSL_CERT', str(sahte))

    r = client.get('/canli/sertifika.crt')
    assert r.status_code == 200
    # Android bu içerik türünü görünce "sertifika kur" ekranını açar
    assert r.headers['content-type'] == 'application/x-x509-ca-cert'
    assert 'BEGIN CERTIFICATE' in r.text


def test_ozel_anahtar_asla_sunulmaz(client):
    """Özel anahtar hiçbir yoldan indirilememeli."""
    for yol in ('/canli/sertifika.key', '/canli/sunucu.key', '/statik/canli/sunucu.key'):
        assert client.get(yol).status_code == 404, yol


# ──────────────────────────────────────────────── neyin kaydedildigi
def test_tespit_edilemeyen_kare_elle_kaydedilebilir(client, monkeypatch):
    """Modelin KACIRDIGI kare de saklanabilmeli.

    Bunlar sürekli iyileştirme için en değerli örneklerdir: kullanıcı hastalığı
    görüyor ama model göremiyorsa, o kare etiketlenip eğitime katılmalı.
    """
    from app import main
    monkeypatch.setattr(main, 'detector', SahteCanliDetector(kutular=[]))

    with client.websocket_connect('/canli/ws') as ws:
        ws.send_json({'tip': 'kaydet'})
        ws.send_bytes(_jpeg())
        yanit = ws.receive_json()

    assert yanit['kutular'] == []
    assert yanit['kayit_id'], 'tespit yokken de elle kayıt açılabilmeli'

    # Tespit içermeyen kayıt inceleme kuyruğuna düşmeli
    from app.database import Analiz, SessionLocal
    with SessionLocal() as db:
        a = db.get(Analiz, yanit['kayit_id'])
        assert a.tespit_sayisi == 0
        assert a.inceleme_gerekli is True


def test_tespit_edilemeyen_kare_otomatik_kaydedilmez(client, monkeypatch):
    """Boş kareler kendiliğinden kaydedilmemeli — depolama dolar."""
    from app import main
    monkeypatch.setattr(main, 'detector', SahteCanliDetector(kutular=[]))
    with client.websocket_connect('/canli/ws') as ws:
        for _ in range(6):
            ws.send_bytes(_jpeg())
            assert ws.receive_json()['kayit_id'] is None


def test_kaydedilen_kare_dosyalari_diske_yazilir(client):
    """Kayıt varsa hem orijinal hem kutulanmış görsel diskte olmalı."""
    from app import config
    from app.database import Analiz, SessionLocal

    with client.websocket_connect('/canli/ws') as ws:
        ws.send_json({'tip': 'kaydet'})
        ws.send_bytes(_jpeg())
        kayit_id = ws.receive_json()['kayit_id']

    with SessionLocal() as db:
        a = db.get(Analiz, kayit_id)
    assert (config.STORAGE_DIR / a.dosya_yolu).exists(), 'orijinal kare yok'
    assert (config.STORAGE_DIR / a.sonuc_yolu).exists(), 'kutulanmış görsel yok'
    assert a.kaynak_tip == 'canli'


def test_kaydedilmeyen_kareler_diske_yazilmaz(client):
    """Canlı akış video/kare biriktirmemeli — yalnızca kaydedilen anlar kalır."""
    from app import config
    once = len(list((config.STORAGE_DIR / 'uploads').glob('*')))
    with client.websocket_connect('/canli/ws') as ws:
        for _ in range(2):                      # kararlılık eşiğinin altında
            ws.send_bytes(_jpeg())
            ws.receive_json()
    sonra = len(list((config.STORAGE_DIR / 'uploads').glob('*')))
    assert sonra == once, 'kaydedilmeyen kareler diske yazılmamalı'


# ──────────────────────────────────────────────────────── kayit modlari
def test_tespitli_mod_her_tespitli_kareyi_kaydeder():
    o = servis.OturumKaydi(mod='tespitli', azami=100, aralik=0)
    assert o.kaydedilsin_mi([_kutu()], simdi=0) is True
    assert o.kaydedilsin_mi([_kutu()], simdi=1) is True     # kararlılık beklemez
    assert o.kaydedilsin_mi([], simdi=2) is False           # tespit yoksa kaydetmez


def test_hepsi_modu_tespit_olmayani_da_kaydeder():
    """Modelin kaçırdığı kareler eğitim için en değerli örneklerdir."""
    o = servis.OturumKaydi(mod='hepsi', azami=100, aralik=0)
    assert o.kaydedilsin_mi([], simdi=0) is True


def test_mod_araligi_ayni_saniyede_yigilmayi_onler():
    o = servis.OturumKaydi(mod='hepsi', azami=100, aralik=1.0)
    assert o.kaydedilsin_mi([], simdi=10.0) is True
    assert o.kaydedilsin_mi([], simdi=10.5) is False        # aralık dolmadı
    assert o.kaydedilsin_mi([], simdi=11.1) is True


def test_oturum_siniri_diski_korur():
    o = servis.OturumKaydi(mod='hepsi', azami=3, aralik=0)
    for i in range(3):
        assert o.kaydedilsin_mi([], simdi=i) is True
    assert o.doldu is True
    assert o.kaydedilsin_mi([], simdi=9) is False, 'sınırdan sonra kayıt olmamalı'
    assert o.elle() is False, 'sınır elle kayıt için de geçerli'


def test_gecersiz_mod_varsayilana_duser():
    from app.moduller.canli import ayarlar
    assert servis.OturumKaydi(mod='saçma').mod == ayarlar.VARSAYILAN_MOD


def test_websocket_mod_degistirilebilir(client):
    """Kullanıcı arayüzden modu seçince sunucu ona göre kaydetmeli."""
    with client.websocket_connect('/canli/ws') as ws:
        ws.send_json({'tip': 'ayar', 'mod': 'tespitli'})
        ws.send_bytes(_jpeg())
        yanit = ws.receive_json()
    assert yanit['kayit_id'], 'tespitli modda ilk kare kaydedilmeli'
    assert yanit['kayit_tipi'] == 'tespitli'
    assert yanit['sayac'] == 1


def test_websocket_sayaci_bildirir(client):
    with client.websocket_connect('/canli/ws') as ws:
        ws.send_bytes(_jpeg())
        yanit = ws.receive_json()
    assert 'sayac' in yanit and 'doldu' in yanit    # arayüz sınırı gösterebilsin
