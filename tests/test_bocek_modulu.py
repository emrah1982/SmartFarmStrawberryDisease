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

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, modeller
from app.moduller.bocek import servis

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


class SahteKutu:
    def __init__(self, cid, guven):
        self.cls = [cid]
        self.conf = [guven]


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
        """
        kaynak = (KOK / 'app' / 'moduller' / 'bocek' / 'rotalar.py').read_text(
            encoding='utf-8')
        assert 'Analiz(' not in kaynak
        assert 'Tespit(' not in kaynak
