"""Hiyerarşik boru hattı: organ → ROI → uzman model → birleştirme.

Bu mimarinin varlık sebebi somut bir hataydı: olgunluk sınıfları YAPRAKLARI
"olgunlaşmamış çilek" sanıyordu ve iki grubun güven aralıkları üst üste
bindiği için ayıran eşik yoktu. Hiyerarşide olgunluk modeli yalnızca meyve
ROI'si görür — hata yapısal olarak imkânsızlaşır. Testler bunu sabitler.
"""

import numpy as np
import pytest

from app import modeller, pipeline
from app.detector import Kutu


# ───────────────────────────────────────────── koordinat dönüşümü
def test_roi_kutusu_orijinal_koordinata_donusur():
    """Uzman model kırpıntıya göre kutu verir; geri dönüştürülmezse tespit
    görüntünün yanlış yerinde görünür — sessiz ve fark edilmesi zor hata."""
    # 1000x1000 görüntünün (200,100) noktasından 400x300 ROI kırpıldı
    # ROI'nin tam ortasında, ROI'nin yarısı kadar bir kutu bulundu
    k = Kutu(0, 'Leaf Spot', 0.9, x=0.5, y=0.5, w=0.5, h=0.5)
    y = pipeline.roi_kutusunu_cevir(k, 400, 300, 200, 100, 1000, 1000)

    # merkez: 200 + 0.5*400 = 400 → 0.40 ;  100 + 0.5*300 = 250 → 0.25
    assert y.x == pytest.approx(0.40)
    assert y.y == pytest.approx(0.25)
    # boyut: 0.5*400 = 200 → 0.20 ;  0.5*300 = 150 → 0.15
    assert y.w == pytest.approx(0.20)
    assert y.h == pytest.approx(0.15)


def test_donusum_sinif_ve_guveni_korur():
    k = Kutu(3, 'Gray Mold', 0.77, 0.5, 0.5, 0.2, 0.2)
    y = pipeline.roi_kutusunu_cevir(k, 200, 200, 0, 0, 400, 400)
    assert (y.sinif_id, y.sinif_adi, y.guven) == (3, 'Gray Mold', 0.77)


def test_tam_goruntu_roi_ise_donusum_kimliktir():
    """ROI = tüm görüntü ise kutu aynen kalmalı."""
    k = Kutu(0, 'x', 0.5, 0.3, 0.7, 0.1, 0.2)
    y = pipeline.roi_kutusunu_cevir(k, 640, 480, 0, 0, 640, 480)
    for a, b in ((y.x, k.x), (y.y, k.y), (y.w, k.w), (y.h, k.h)):
        assert a == pytest.approx(b)


def test_kose_roi_de_dogru_donusur():
    """Sağ alt köşeden kırpılan ROI'deki kutu, görüntünün sağ altında kalmalı."""
    k = Kutu(0, 'x', 0.5, x=0.9, y=0.9, w=0.1, h=0.1)
    y = pipeline.roi_kutusunu_cevir(k, 100, 100, 900, 900, 1000, 1000)
    assert y.x == pytest.approx(0.99)
    assert y.y == pytest.approx(0.99)


# ───────────────────────────────────────────────────── ROI kırpma
def _kare(g=1000, y=1000):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (y, g, 3), dtype=np.uint8)


def test_kirpma_pay_birakir():
    """Lezyon çoğu zaman organ kutusunun kenarındadır; paysız kırpma böler."""
    frame = _kare()
    kutu = Kutu(0, 'fruit', 0.9, x=0.5, y=0.5, w=0.2, h=0.2)
    roi, ox, oy = pipeline._kirp(frame, kutu)
    # paysız 200x200 olurdu; %12 payla daha büyük
    assert roi.shape[1] > 200 and roi.shape[0] > 200


def test_cok_kucuk_roi_atlanir():
    """Birkaç piksellik ROI'de uzman model anlamlı çalışmaz."""
    frame = _kare()
    assert pipeline._kirp(frame, Kutu(0, 'fruit', 0.9, 0.5, 0.5, 0.01, 0.01)) is None


def test_kirpma_goruntu_disina_tasmaz():
    frame = _kare(400, 300)
    roi, ox, oy = pipeline._kirp(frame, Kutu(0, 'leaf', 0.9, x=0.95, y=0.95, w=0.3, h=0.3))
    assert ox >= 0 and oy >= 0
    assert ox + roi.shape[1] <= 400
    assert oy + roi.shape[0] <= 300


# ─────────────────────────────────────── yönlendirme (asıl kazanım)
class SahteModel:
    """Verilen kutuları döndüren sahte YOLO."""

    def __init__(self, kutular):
        self.kutular = kutular
        self.cagri = 0

    def __call__(self, goruntu, **kw):
        self.cagri += 1

        class R:
            names = {i: a for i, (a, *_) in enumerate(self.kutular)}
            boxes = []
        r = R()
        for i, (ad, guven, x, y, w, h) in enumerate(self.kutular):
            # Ultralytics tensör döner; .tolist() çağrısı bu yüzden taklit edilir
            class Dizi(list):
                def tolist(self):
                    return list(self)

            class B:
                cls = [i]
                conf = [guven]
                xywhn = [Dizi([x, y, w, h])]
            r.boxes.append(B())
        return [r]


@pytest.fixture
def sahte_kurulum(monkeypatch):
    """organ + olgunluk + yaprak hastalığı modellerini taklit eder."""
    yuklenen = {}

    def sahte_yukle(ad, urun=None):
        return yuklenen.get(ad)

    monkeypatch.setattr(modeller, 'yukle', sahte_yukle)
    return yuklenen


def _tanim(ad, rol, tetik=(), esik=0.2):
    return modeller.ModelTanimi(ad=ad, dosya=f'{ad}.pt', rol=rol,
                                tetik=list(tetik), esik=esik)


def test_olgunluk_modeli_yaprak_roisinde_calistirilmaz(monkeypatch, sahte_kurulum):
    """MİMARİNİN ASIL KAZANIMI.

    Görüntüde yalnızca yaprak varsa olgunluk modeli HİÇ çağrılmamalı; yani
    bir yaprağı 'olgunlaşmamış çilek' diye işaretlemesi imkânsız olmalı.
    """
    organ = SahteModel([('leaf', 0.9, 0.5, 0.5, 0.4, 0.4)])
    olgunluk = SahteModel([('strawberry_unripe', 0.95, 0.5, 0.5, 0.5, 0.5)])
    yaprak = SahteModel([('Leaf Spot', 0.8, 0.5, 0.5, 0.3, 0.3)])
    sahte_kurulum.update({'organ': organ, 'olgunluk': olgunluk,
                          'yaprak_hastalik': yaprak})

    monkeypatch.setattr(modeller, 'rol_ile',
                        lambda rol, urun=None: [_tanim('organ', 'organ')] if rol == 'organ' else [])
    monkeypatch.setattr(modeller, 'tetiklenen', lambda organ_ad, urun=None: (
        [_tanim('yaprak_hastalik', 'yaprak_hastalik', ['leaf'])]
        if organ_ad == 'leaf' else
        [_tanim('olgunluk', 'olgunluk', ['fruit'])]))

    kutular, iz = pipeline.calistir(_kare())

    assert olgunluk.cagri == 0, 'olgunluk modeli yaprakta ÇALIŞMAMALI'
    assert yaprak.cagri == 1
    assert 'olgunluk' not in iz.calisan_modeller
    assert [k.sinif_adi for k in kutular] == ['Leaf Spot']


def test_meyve_bulununca_olgunluk_calisir(monkeypatch, sahte_kurulum):
    organ = SahteModel([('fruit', 0.9, 0.5, 0.5, 0.4, 0.4)])
    olgunluk = SahteModel([('strawberry_ripe', 0.9, 0.5, 0.5, 0.5, 0.5)])
    sahte_kurulum.update({'organ': organ, 'olgunluk': olgunluk})
    monkeypatch.setattr(modeller, 'rol_ile',
                        lambda rol, urun=None: [_tanim('organ', 'organ')] if rol == 'organ' else [])
    monkeypatch.setattr(modeller, 'tetiklenen', lambda o, urun=None: (
        [_tanim('olgunluk', 'olgunluk', ['fruit'])] if o == 'fruit' else []))

    kutular, iz = pipeline.calistir(_kare())
    assert olgunluk.cagri == 1
    assert [k.sinif_adi for k in kutular] == ['strawberry_ripe']
    assert iz.roi_sayisi == 1


def test_organ_modeli_yoksa_mirasa_duser(monkeypatch, sahte_kurulum):
    """Kademeli geçiş: uzman modeller hazır değilken sistem çalışmaya devam eder."""
    miras = SahteModel([('Gray Mold', 0.8, 0.5, 0.5, 0.3, 0.3)])
    sahte_kurulum['miras'] = miras
    monkeypatch.setattr(modeller, 'rol_ile', lambda rol, urun=None: [])
    monkeypatch.setattr(modeller, 'tanim', lambda ad, urun=None: _tanim('miras', 'miras'))

    kutular, iz = pipeline.calistir(_kare())
    assert miras.cagri == 1
    assert iz.miras_kullanildi is True
    assert [k.sinif_adi for k in kutular] == ['Gray Mold']


def test_uzman_model_eksikse_mirasla_tamamlanir(monkeypatch, sahte_kurulum):
    """Organ var ama uzman model dosyası yoksa tespit kaybolmamalı."""
    organ = SahteModel([('fruit', 0.9, 0.5, 0.5, 0.4, 0.4)])
    miras = SahteModel([('Anthracnose Fruit Rot', 0.7, 0.5, 0.5, 0.2, 0.2)])
    sahte_kurulum.update({'organ': organ, 'miras': miras})   # olgunluk YOK
    monkeypatch.setattr(modeller, 'rol_ile',
                        lambda rol, urun=None: [_tanim('organ', 'organ')] if rol == 'organ' else [])
    monkeypatch.setattr(modeller, 'tetiklenen', lambda o, urun=None: [])
    monkeypatch.setattr(modeller, 'tanim', lambda ad, urun=None: _tanim('miras', 'miras'))

    kutular, iz = pipeline.calistir(_kare())
    assert iz.miras_kullanildi is True
    assert len(kutular) == 1


def test_iz_ozeti_okunabilir(monkeypatch, sahte_kurulum):
    organ = SahteModel([('fruit', 0.9, 0.3, 0.3, 0.2, 0.2),
                        ('leaf', 0.8, 0.7, 0.7, 0.2, 0.2)])
    olgunluk = SahteModel([('strawberry_ripe', 0.9, 0.5, 0.5, 0.4, 0.4)])
    sahte_kurulum.update({'organ': organ, 'olgunluk': olgunluk})
    monkeypatch.setattr(modeller, 'rol_ile',
                        lambda rol, urun=None: [_tanim('organ', 'organ')] if rol == 'organ' else [])
    monkeypatch.setattr(modeller, 'tetiklenen', lambda o, urun=None: (
        [_tanim('olgunluk', 'olgunluk', ['fruit'])] if o == 'fruit' else []))

    _, iz = pipeline.calistir(_kare())
    ozet = iz.ozet()
    assert 'organ' in ozet and 'ROI' in ozet
    assert len(iz.organlar) == 2
