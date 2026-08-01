"""Böcek teşhis modülü — ayrı akış, kapalı küme uyarısı.

NEDEN TEST?
    Bu modelin en tehlikeli yanı KAPALI KÜME olmasıdır: yalnızca 6 tür bilir
    ve "bilmiyorum" diyemez. Kullanıcı yaprak biti fotoğrafı çekerse
    "Toprak Larvası %70" cevabı alır. Buna güvenip yanlış mücadele yaparsa
    hem para hem zaman kaybeder.

    Arayüzün işi bu sınırı GİZLEMEMEK: tek cevap yerine aday listesi,
    bilinen türlerin açık listesi ve kararsızlık uyarısı. Bu güvenceler
    şablon metnine dayandığı için testle sabitlenmiştir.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, main, modeller
from app.database import SessionLocal
from app.moduller.bocek import servis
from app.moduller.bocek.modeller import BocekKaydi

KOK = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def temiz():
    modeller.bosalt_kutuk()
    modeller.bosalt()
    yield
    modeller.bosalt_kutuk()
    modeller.bosalt()


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def db(client):
    """client'tan SONRA açılır: tablolar uygulama başlarken kurulur."""
    o = SessionLocal()
    try:
        yield o
    finally:
        o.close()


class SahteKutu:
    def __init__(self, cid, guven, xywhn=(0.5, 0.5, 0.2, 0.2)):
        self.cls = [cid]
        self.conf = [guven]
        self.xywhn = [list(xywhn)]


class SahteSonuc:
    def __init__(self, kutular, adlar):
        self.boxes = kutular
        self.names = adlar


class SahteModel:
    """Ultralytics YOLO nesnesinin çağrı arayüzünü taklit eder."""

    ADLAR = {0: 'Army Worm', 1: 'Black Cutworm', 2: 'Grub',
             3: 'Mole Cricket', 4: 'Peach Borer', 5: 'Spider Mites'}

    def __init__(self, kutular):
        self._kutular = kutular

    def __call__(self, goruntu, **kw):
        esik = kw.get('conf', 0)
        return [SahteSonuc([k for k in self._kutular if k.conf[0] >= esik],
                           self.ADLAR)]


def _model_kur(monkeypatch, kutular):
    monkeypatch.setattr(modeller, 'yukle',
                        lambda ad, urun=None: SahteModel(kutular))


class TestTani:
    def test_adaylar_guvene_gore_siralanir(self, monkeypatch):
        _model_kur(monkeypatch, [SahteKutu(2, 0.30), SahteKutu(3, 0.82),
                                 SahteKutu(0, 0.55)])
        s = servis.tani(np.zeros((10, 10, 3), np.uint8))
        assert [a.ad for a in s.adaylar] == ['Mole Cricket', 'Army Worm', 'Grub']
        assert s.en_iyi.yuzde == 82

    def test_en_fazla_uc_aday(self, monkeypatch):
        _model_kur(monkeypatch, [SahteKutu(i, 0.9 - i * 0.1) for i in range(6)])
        assert len(servis.tani(np.zeros((10, 10, 3), np.uint8)).adaylar) == 3

    def test_ayni_tur_tekrar_etmez_en_yuksek_alinir(self, monkeypatch):
        """Birkaç birey varsa soru 'kaç tane' değil 'bu ne'."""
        _model_kur(monkeypatch, [SahteKutu(2, 0.40), SahteKutu(2, 0.75),
                                 SahteKutu(2, 0.60)])
        s = servis.tani(np.zeros((10, 10, 3), np.uint8))
        assert len(s.adaylar) == 1
        assert s.adaylar[0].yuzde == 75

    def test_yakin_adaylar_KARARSIZ_isaretlenir(self, monkeypatch):
        """Tek cevap göstermek burada yanıltıcı olur."""
        _model_kur(monkeypatch, [SahteKutu(2, 0.52), SahteKutu(3, 0.48)])
        assert servis.tani(np.zeros((10, 10, 3), np.uint8)).kararsiz is True

    def test_belirgin_fark_kararsiz_degil(self, monkeypatch):
        _model_kur(monkeypatch, [SahteKutu(2, 0.88), SahteKutu(3, 0.25)])
        assert servis.tani(np.zeros((10, 10, 3), np.uint8)).kararsiz is False

    def test_dusuk_guvenli_tespit_gosterilmez(self, monkeypatch):
        _model_kur(monkeypatch, [SahteKutu(2, 0.05)])
        assert servis.tani(np.zeros((10, 10, 3), np.uint8)).adaylar == []

    def test_model_yoksa_hata_doner_cokmez(self, monkeypatch):
        monkeypatch.setattr(modeller, 'yukle', lambda ad, urun=None: None)
        s = servis.tani(np.zeros((10, 10, 3), np.uint8))
        assert not s.bulundu and s.hata

    def test_model_patlarsa_akis_kesilmez(self, monkeypatch):
        class Patlayan:
            def __call__(self, *a, **k):
                raise RuntimeError('cuda yok')

        monkeypatch.setattr(modeller, 'yukle', lambda ad, urun=None: Patlayan())
        s = servis.tani(np.zeros((10, 10, 3), np.uint8))
        assert not s.bulundu and 'RuntimeError' in s.hata


class TestYedekTani:
    """Bitki analizi boş dönünce çalışan tamamlayıcı teşhis.

    NEDEN? Kullanıcı yaprağındaki tırtılın fotoğrafını yükleyip
    "hastalık veya meyve tespit edilmedi" cevabı aldı. Doğru cevaptı —
    karede çilek organı yok — ama işe yaramazdı: görüntüde apaçık bir
    zararlı vardı ve sistem onu tanıyabilecek modele sahipti.

    Ama dikkatli olmak gerek: model KAPALI KÜMEDİR, bulanık bir duvar
    fotoğrafına da kendinden emin bir cevap verebilir. Kullanıcı bu
    fotoğrafı böcek sorusuyla YÜKLEMEDİĞİ için çıta daha yüksek olmalı.
    """

    def test_yuksek_guvende_oneri_doner(self, monkeypatch):
        _model_kur(monkeypatch, [SahteKutu(0, 0.92)])
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        b = servis.yedek_tani(np.zeros((10, 10, 3), np.uint8))
        assert b['ad'] == 'Army Worm'
        assert b['guven'] == pytest.approx(0.92)

    def test_dusuk_guvende_SESSIZ_kalir(self, monkeypatch):
        """Kullanıcı böcek sormadı; zayıf tahminle onu yanıltmayalım."""
        _model_kur(monkeypatch, [SahteKutu(0, 0.40)])
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        assert servis.yedek_tani(np.zeros((10, 10, 3), np.uint8)) == {}

    def test_esik_normal_teshisten_YUKSEK(self):
        """Kendiliğinden çıkan öneri, açıkça sorulan teşhisten sıkı olmalı."""
        assert servis.YEDEK_EN_DUSUK_GUVEN > servis.EN_DUSUK_GUVEN

    def test_model_yokken_sessiz(self, monkeypatch):
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: False)
        assert servis.yedek_tani(np.zeros((10, 10, 3), np.uint8)) == {}

    def test_bocek_yoksa_sessiz(self, monkeypatch):
        _model_kur(monkeypatch, [])
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        assert servis.yedek_tani(np.zeros((10, 10, 3), np.uint8)) == {}

    def test_adaylar_da_dondurulur(self, monkeypatch):
        """Arayüz tek cevap değil dağılım gösterir — kapalı küme uyarısı."""
        _model_kur(monkeypatch, [SahteKutu(0, 0.80), SahteKutu(2, 0.30)])
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        b = servis.yedek_tani(np.zeros((10, 10, 3), np.uint8))
        assert len(b['adaylar']) == 2

    def test_kutu_konumu_dondurulur(self, monkeypatch):
        """Kullanıcı böceğin karenin NERESİNDE olduğunu görmeli."""
        _model_kur(monkeypatch, [SahteKutu(0, 0.90, (0.4, 0.55, 0.3, 0.2))])
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        b = servis.yedek_tani(np.zeros((10, 10, 3), np.uint8))
        assert b['kutu']['x'] == pytest.approx(0.4)
        assert b['kutu']['w'] == pytest.approx(0.3)


class TestKutuCizimi:
    def test_kutu_goruntuye_cizilir(self):
        f = np.zeros((200, 200, 3), np.uint8)
        bulgu = {'ad': 'Army Worm', 'guven': 0.9,
                 'kutu': {'x': 0.5, 'y': 0.5, 'w': 0.4, 'h': 0.4, 'sinif_id': 0}}
        ciz = servis.kutuyu_ciz(f, bulgu)
        assert ciz is not None
        assert ciz.shape == f.shape
        assert ciz.any(), 'görüntüye hiçbir şey çizilmemiş'

    def test_kutusuz_bulguda_goruntu_degismez(self):
        f = np.zeros((50, 50, 3), np.uint8)
        assert servis.kutuyu_ciz(f, {'ad': 'X', 'guven': 0.9, 'kutu': None}) is f

    def test_bos_bulgu_cokmez(self):
        f = np.zeros((50, 50, 3), np.uint8)
        assert servis.kutuyu_ciz(f, {}) is f
        assert servis.kutuyu_ciz(f, None) is f

    def test_teshis_sayfasi_gorseline_de_cizilir(self, monkeypatch):
        """Modülün KENDİ sayfası da kutuyu göstermeli — yalnızca yedek
        akışa eklemek eksik kalıyordu."""
        _model_kur(monkeypatch, [SahteKutu(0, 0.9, (0.5, 0.5, 0.4, 0.4))])
        s = servis.tani(np.zeros((200, 200, 3), np.uint8))
        ciz = servis.adaylari_ciz(np.zeros((200, 200, 3), np.uint8), s)
        assert ciz.any(), 'teşhis sayfasında kutu çizilmemiş'

    def test_HER_birey_cizilir(self, monkeypatch):
        """GERÇEK HATA: 3 tırtıllı karede tek kutu çiziliyordu.

        Etiket dosyasında 3 kutu, model 2 buluyor, arayüze 1 ulaşıyordu —
        çünkü tür başına tek kutu tutuluyordu. Zararlıda SAYI da tarımsal
        bilgidir: 1 birey ile 20 birey aynı şey değildir.
        """
        _model_kur(monkeypatch, [SahteKutu(0, 0.9, (0.3, 0.3, 0.2, 0.2)),
                                 SahteKutu(0, 0.7, (0.7, 0.7, 0.2, 0.2))])
        s = servis.tani(np.zeros((200, 200, 3), np.uint8))
        assert len(s.kutular) == 2, 'her birey saklanmalı'
        assert len(s.adaylar) == 1, 'tür listesi tek satır olmalı (aynı tür)'
        assert s.adet('Army Worm') == 2

        ciz = servis.adaylari_ciz(np.zeros((200, 200, 3), np.uint8), s)
        assert ciz[40:80, 40:80].any(), 'birinci birey çizilmemiş'
        assert ciz[150:190, 150:190].any(), 'ikinci birey çizilmemiş'

    def test_farkli_turler_de_hepsi_cizilir(self, monkeypatch):
        _model_kur(monkeypatch, [SahteKutu(0, 0.9, (0.3, 0.3, 0.2, 0.2)),
                                 SahteKutu(2, 0.7, (0.7, 0.7, 0.2, 0.2))])
        s = servis.tani(np.zeros((200, 200, 3), np.uint8))
        assert len(s.kutular) == 2 and len(s.adaylar) == 2
        ciz = servis.adaylari_ciz(np.zeros((200, 200, 3), np.uint8), s)
        assert ciz[40:80, 40:80].any() and ciz[150:190, 150:190].any()

    def test_eski_kayit_tek_kutu_bicimi_calisir(self):
        """Geçmiş kayıtlarda `kutular` yok, yalnızca `kutu` vardı."""
        f = np.zeros((200, 200, 3), np.uint8)
        eski = {'ad': 'Army Worm', 'guven': 0.9,
                'kutu': {'x': 0.5, 'y': 0.5, 'w': 0.4, 'h': 0.4, 'sinif_id': 0}}
        assert servis.kutuyu_ciz(f, eski).any()

    def test_kutusuz_sonucta_gorsel_degismez(self, monkeypatch):
        _model_kur(monkeypatch, [])
        s = servis.tani(np.zeros((50, 50, 3), np.uint8))
        f = np.zeros((50, 50, 3), np.uint8)
        assert servis.adaylari_ciz(f, s) is f


class TestYedekAkis:
    """Uçtan uca: tespit yoksa öneri çıkar, tespit varsa ÇIKMAZ."""

    @staticmethod
    def _detector(kutular, iz=None):
        from tests.test_app import SahteDetector

        class D(SahteDetector):
            def _sonuc(self, cikti_yol, kare=1):
                s = super()._sonuc(cikti_yol, kare)
                s.iz = dict(iz or {})
                return s
        return D(kutular)

    def test_tespit_yokken_oneri_gosterilir(self, monkeypatch):
        from app import main
        monkeypatch.setattr(main, 'detector', self._detector([]))
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        _model_kur(monkeypatch, [SahteKutu(0, 0.92)])
        with TestClient(main.app) as c:
            r = c.post('/analiz/dosya',
                       files={'dosyalar': ('a.jpg', _kucuk_jpeg(), 'image/jpeg')},
                       follow_redirects=True)
        assert 'Böcek olabilir mi' in r.text
        assert 'bilmiyorum' in r.text, 'kapalı küme uyarısı görünmeli'

    def test_tespit_VARKEN_oneri_gosterilmez(self, monkeypatch):
        """İki cevap birden kullanıcıyı hangisinin asıl olduğu konusunda
        kararsız bırakır."""
        from app import main
        from app.detector import Kutu
        monkeypatch.setattr(main, 'detector',
                            self._detector([Kutu(3, 'Gray Mold', 0.9, .5, .5, .2, .2)]))
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        _model_kur(monkeypatch, [SahteKutu(0, 0.99)])
        with TestClient(main.app) as c:
            r = c.post('/analiz/dosya',
                       files={'dosyalar': ('a.jpg', _kucuk_jpeg(), 'image/jpeg')},
                       follow_redirects=True)
        assert 'Böcek olabilir mi' not in r.text

    def test_bocek_modeli_yokken_sayfa_bozulmaz(self, monkeypatch):
        from app import main
        monkeypatch.setattr(main, 'detector', self._detector([]))
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: False)
        with TestClient(main.app) as c:
            r = c.post('/analiz/dosya',
                       files={'dosyalar': ('a.jpg', _kucuk_jpeg(), 'image/jpeg')},
                       follow_redirects=True)
        assert r.status_code == 200
        assert 'Böcek olabilir mi' not in r.text


class TestKutukleTutarli:
    def test_taniyabildikleri_kutukten_gelir(self):
        t = modeller.tanim('bocek_teshis', 'cilek')
        assert servis.taniyabildikleri('cilek') == t.siniflar

    def test_alti_tur(self):
        assert len(servis.taniyabildikleri('cilek')) == 6

    def test_oneri_ortak_metni_doner(self):
        """Makro fotoğrafta böceğin organı bilinmez → organ bloğu kullanılmaz."""
        from app import tedavi
        o = servis.oneri('Spider Mites', 'cilek')
        assert o, 'kırmızı örümceğin önerisi olmalı'
        ortak = tedavi.coz(tedavi.yukle('cilek'), 'Spider Mites')
        assert o['belirti'] == ortak['belirti']


class TestSayfa:
    def test_sayfa_acilir(self, client):
        r = client.get('/bocek')
        assert r.status_code == 200
        assert 'Böcek Teşhis' in r.text

    def test_menude_gorunur(self, client):
        assert '/bocek' in client.get('/').text

    def test_ayri_akis_oldugu_yazili(self, client):
        """Kullanıcı hangi fotoğrafı nereye vereceğini karıştırmamalı."""
        r = client.get('/bocek')
        assert 'ayrı bir akış' in r.text

    def test_model_yokken_kurulum_yonergesi(self, client):
        """Model kurulu değilken sayfa boş kalmamalı, ne yapılacağını söylemeli."""
        r = client.get('/bocek')
        if not servis.hazir():
            assert 'model_kur.py bocek_teshis' in r.text
            assert "EGITILECEK = 'bocek_teshis'" in r.text

    def test_bos_dosya_hata_verir(self, client):
        r = client.post('/bocek/tani', files={'dosya': ('a.jpg', b'', 'image/jpeg')})
        assert r.status_code == 200
        assert 'boş' in r.text

    def test_bozuk_dosya_cokmez(self, client):
        r = client.post('/bocek/tani',
                        files={'dosya': ('a.jpg', b'bu-jpeg-degil', 'image/jpeg')})
        assert r.status_code == 200
        assert 'okunamadı' in r.text


class TestKapaliKumeUyarisi:
    """Modelin bilmediği türe de cevap ürettiği AÇIKÇA yazmalı."""

    def test_bilinen_turler_listeleniyor(self, client, monkeypatch):
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        r = client.get('/bocek')
        for t in servis.taniyabildikleri('cilek'):
            assert t in r.text, f'{t} listede yok'

    def test_kapali_kume_uyarisi_var(self, client, monkeypatch):
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        r = client.get('/bocek')
        assert 'Listede olmayan bir böcek için de cevap üretir' in r.text
        assert 'bilmiyorum' in r.text

    def test_saha_taramasi_degil_uyarisi(self, client, monkeypatch):
        """Kullanıcı bu sayfayı tarla taraması sanmamalı."""
        monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
        assert 'Tarlada zararlı taraması bu sayfa değildir' in client.get('/bocek').text


class TestBoruHattinaGirmez:
    def test_hicbir_organ_tetiklemez(self):
        for organ in ('Leaf', 'Fruit', 'Flower'):
            adlar = [t.ad for t in modeller.tetiklenen(organ, 'cilek')]
            assert 'bocek_teshis' not in adlar

    def test_analiz_tablosuna_yazmaz(self):
        """Bu akış bitki analizi değil tür sorgusudur.

        Analiz tablosuna yazılsaydı hastalık istatistiklerine ve yaygınlık
        haritasına karışır, "şu serada 12 tespit" sayısı anlamını yitirirdi.
        Kendi tablosu var: bocek_kayitlari.
        """
        kaynak = (KOK / 'app' / 'moduller' / 'bocek' / 'rotalar.py').read_text(
            encoding='utf-8')
        assert 'Analiz(' not in kaynak
        assert 'Tespit(' not in kaynak

    def test_kendi_tablosunu_kullanir(self):
        from app.moduller.bocek.modeller import BocekKaydi
        assert BocekKaydi.__tablename__ == 'bocek_kayitlari'


# ═════════════════════════════════════════════════════ kayıt ve doğrulama
def _tani_ettir(client, monkeypatch, kutular):
    """Bir teşhis yapıp kaydı döndürür."""
    _model_kur(monkeypatch, kutular)
    monkeypatch.setattr(servis, 'hazir', lambda urun=None: True)
    r = client.post('/bocek/tani',
                    files={'dosya': ('a.jpg', _kucuk_jpeg(), 'image/jpeg')})
    assert r.status_code == 200
    return r


def _kucuk_jpeg() -> bytes:
    """Gerçek, çözülebilir bir JPEG — cv2.imdecode None dönmemeli."""
    ok, tampon = cv2.imencode('.jpg', np.full((32, 32, 3), 128, np.uint8))
    assert ok
    return tampon.tobytes()


class TestKayit:
    def test_teshis_kaydediliyor(self, client, monkeypatch, db):
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        assert k is not None
        assert k.tur == 'Mole Cricket'
        assert round(k.guven, 2) == 0.82
        assert k.gorsel, 'fotoğraf yolu kaydedilmeli'

    def test_adaylar_da_saklaniyor(self, client, monkeypatch, db):
        """Sonradan 'model neyi karıştırmış' sorusu ancak böyle cevaplanır."""
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.55), SahteKutu(2, 0.45)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        assert [a['ad'] for a in k.adaylar] == ['Mole Cricket', 'Grub']
        assert k.kararsiz is True

    def test_bocek_bulunamasa_da_kaydediliyor(self, client, monkeypatch, db):
        """'Model bir şey göremedi' bilgisi de modeli değerlendirmek için gerekli."""
        _tani_ettir(client, monkeypatch, [])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        assert k is not None and k.tur == ''

    def test_sonuc_sayfasinda_dogrulama_sorulur(self, client, monkeypatch):
        """Böcek hâlâ elinizdeyken sorulmalı; geçmişe bırakılırsa kimse
        dönüp işaretlemez ve isabet ölçülemez."""
        r = _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        assert 'Bu teşhis doğru mu?' in r.text
        assert 'listede_yok' in r.text


class TestDogrulama:
    def test_dogru_isaretlenir(self, client, monkeypatch, db):
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        client.post(f'/bocek/kayit/{k.id}/dogrula', data={'dogrulama': 'dogru'},
                    follow_redirects=False)
        db.expire_all()
        assert db.get(BocekKaydi, k.id).dogrulama == 'dogru'

    def test_yanlis_isaretinde_dogru_tur_saklanir(self, client, monkeypatch, db):
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        client.post(f'/bocek/kayit/{k.id}/dogrula',
                    data={'dogrulama': 'yanlis', 'dogru_tur': 'Yaprak Biti'},
                    follow_redirects=False)
        db.expire_all()
        g = db.get(BocekKaydi, k.id)
        assert g.dogru_tur == 'Yaprak Biti'
        assert g.gecerli_tur == 'Yaprak Biti'
        assert g.tur == 'Mole Cricket', 'modelin cevabı KORUNMALI (isabet ölçümü)'

    def test_dogru_isaretinde_tur_alani_temizlenir(self, client, monkeypatch, db):
        """'doğru' seçilince eski 'doğru tür' metni kalmamalı."""
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        client.post(f'/bocek/kayit/{k.id}/dogrula',
                    data={'dogrulama': 'yanlis', 'dogru_tur': 'X'},
                    follow_redirects=False)
        client.post(f'/bocek/kayit/{k.id}/dogrula', data={'dogrulama': 'dogru'},
                    follow_redirects=False)
        db.expire_all()
        assert db.get(BocekKaydi, k.id).dogru_tur == ''

    def test_gecersiz_deger_yazilmaz(self, client, monkeypatch, db):
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        client.post(f'/bocek/kayit/{k.id}/dogrula',
                    data={'dogrulama': 'saçma_değer'}, follow_redirects=False)
        db.expire_all()
        assert db.get(BocekKaydi, k.id).dogrulama == ''

    def test_olmayan_kayit_cokmez(self, client):
        r = client.post('/bocek/kayit/999999/dogrula',
                        data={'dogrulama': 'dogru'}, follow_redirects=False)
        assert r.status_code == 303


class TestIsabetOzeti:
    def test_oran_yalnizca_degerlendirilenlerden(self):
        """'listede yok' isabete katılmamalı: model yanılmadı, soru dışıydı."""
        from app.moduller.bocek.modeller import isabet

        class K:
            def __init__(self, d):
                self.dogrulama = d

        o = isabet([K('dogru'), K('dogru'), K('dogru'), K('yanlis'),
                    K('listede_yok'), K('')])
        assert o['toplam'] == 6
        assert o['oran'] == 75.0        # 3 doğru / 4 değerlendirilen
        assert o['dogrulanmamis'] == 1

    def test_hic_degerlendirilmemisse_oran_yok(self):
        from app.moduller.bocek.modeller import isabet

        class K:
            dogrulama = ''

        assert isabet([K(), K()])['oran'] is None

    def test_gecmis_sayfasi_acilir(self, client):
        r = client.get('/bocek/gecmis')
        assert r.status_code == 200
        assert 'Sahadaki isabet' in r.text

    def test_sekmeler_her_iki_sayfada(self, client):
        for yol in ('/bocek', '/bocek/gecmis'):
            t = client.get(yol).text
            assert '/bocek/gecmis' in t and 'Teşhis' in t

    def test_gecmis_isabeti_suzgecten_etkilenmez(self, client, monkeypatch, db):
        """Özet TÜM kayıtlardan hesaplanmalı, süzülmüş listeden değil.

        'yanlış' süzgecinde isabet %0 görünseydi kullanıcı modelin hiç
        tutturamadığını sanırdı.
        """
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        client.post(f'/bocek/kayit/{k.id}/dogrula', data={'dogrulama': 'dogru'},
                    follow_redirects=False)

        db.expire_all()
        from app.moduller.bocek.modeller import isabet
        beklenen = isabet(db.query(BocekKaydi).all())
        assert beklenen['dogru'] >= 1

        # Süzgeç hiçbir kayıt döndürmese bile özet aynı sayıları göstermeli
        r = client.get('/bocek/gecmis?dogrulama=yanlis')
        assert f'>{beklenen["toplam"]}<' in r.text, 'toplam süzgece göre değişmiş'
        assert f'>{beklenen["dogru"]}<' in r.text, 'doğru sayısı süzgece göre değişmiş'


class TestSilme:
    def test_kayit_ve_dosya_silinir(self, client, monkeypatch, db):
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        yol = config.STORAGE_DIR / k.gorsel
        assert yol.exists()
        kimlik = k.id
        client.post(f'/bocek/kayit/{kimlik}/sil', follow_redirects=False)
        db.expire_all()
        assert db.get(BocekKaydi, kimlik) is None
        assert not yol.exists()

    def test_dosya_yoksa_da_kayit_silinir(self, client, monkeypatch, db):
        _tani_ettir(client, monkeypatch, [SahteKutu(3, 0.82)])
        k = db.query(BocekKaydi).order_by(BocekKaydi.id.desc()).first()
        (config.STORAGE_DIR / k.gorsel).unlink()
        kimlik = k.id
        client.post(f'/bocek/kayit/{kimlik}/sil', follow_redirects=False)
        db.expire_all()
        assert db.get(BocekKaydi, kimlik) is None
