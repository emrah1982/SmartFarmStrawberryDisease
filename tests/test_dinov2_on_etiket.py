"""dinov2_on_etiket.py — omurgadan BAĞIMSIZ mantık.

torch/transformers bu depoda kurulu değil (modeller Docker/Colab'da
çalışır). Bu yüzden testler DINOv2'yi hiç çağırmaz; sahte patch gömüsü
üretip geometriyi, prototip çıkarımını ve kutu üretimini doğrular.

Doğrulanan asıl şey: patch ızgarasından normalize görüntü koordinatına
geçiş DOĞRU mu. Burada bir kayma olsa aday kutular sistematik olarak
kaymış olur ve etiketleyici bunu fark etmeden düzeltmeye çalışır.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

KOK = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'dinov2_on_etiket', KOK / 'scripts' / 'dinov2_on_etiket.py')
d2 = importlib.util.module_from_spec(_spec)
sys.modules['dinov2_on_etiket'] = d2
_spec.loader.exec_module(d2)

IZGARA = 20
BOYUT = 8


def _bos_gomu(izgara=IZGARA, boyut=BOYUT):
    """Arka plan: hepsi aynı yöne bakan birim vektörler."""
    g = np.zeros((izgara, izgara, boyut))
    g[..., 0] = 1.0
    return g


def _bolge_koy(g, y0, y1, x0, x1, eksen):
    """Verilen patch dikdörtgenine ayrı bir yön yerleştirir."""
    g[y0:y1, x0:x1] = 0.0
    g[y0:y1, x0:x1, eksen] = 1.0
    return g


def test_kutu_patchleri_dogru_dilimi_verir():
    # Karenin sol üst çeyreği: cx=cy=0.25, w=h=0.5
    y0, y1, x0, x1 = d2.kutu_patchleri((0.25, 0.25, 0.5, 0.5), IZGARA)
    assert (y0, y1, x0, x1) == (0, 10, 0, 10)


def test_kutu_patchleri_sinirlari_asmaz():
    y0, y1, x0, x1 = d2.kutu_patchleri((0.95, 0.95, 0.4, 0.4), IZGARA)
    assert 0 <= y0 < y1 <= IZGARA
    assert 0 <= x0 < x1 <= IZGARA


def test_bilesen_kutusu_dogru_yerde_cikar():
    """Bilinen bir dikdörtgen maske → aynı yerde normalize kutu."""
    maske = np.zeros((IZGARA, IZGARA), dtype=bool)
    maske[4:8, 10:16] = True          # y 4-8, x 10-16
    (cx, cy, w, h, n), = d2.bilesen_kutulari(maske)
    assert n == 24
    assert cx == pytest.approx((10 + 16) / 2 / IZGARA)
    assert cy == pytest.approx((4 + 8) / 2 / IZGARA)
    assert w == pytest.approx(6 / IZGARA)
    assert h == pytest.approx(4 / IZGARA)


def test_ayrik_iki_bolge_iki_kutu_verir():
    maske = np.zeros((IZGARA, IZGARA), dtype=bool)
    maske[2:6, 2:6] = True
    maske[12:17, 12:17] = True
    assert len(d2.bilesen_kutulari(maske)) == 2


def test_cok_kucuk_bilesen_elenir():
    maske = np.zeros((IZGARA, IZGARA), dtype=bool)
    maske[5, 5] = True                 # tek patch — gürültü
    assert d2.bilesen_kutulari(maske) == []


def test_prototip_kutu_icindeki_patchlerden_kurulur():
    g = _bos_gomu()
    _bolge_koy(g, 0, 10, 0, 10, eksen=1)     # sol üst çeyrek: yön 1
    gomuler = g[None]
    etiketler = [[(0, (0.25, 0.25, 0.5, 0.5))]]
    proto, adet = d2.prototip_kur(gomuler, etiketler, sinif_sayisi=2)

    assert adet[0] == 100
    assert adet[1] == 0
    assert proto[0][1] == pytest.approx(1.0)     # yön 1'i öğrendi
    assert np.allclose(proto[1], 0)              # tohumsuz sınıf boş kaldı


def test_tohumsuz_sinif_rapora_sifir_dusor():
    g = _bos_gomu()
    proto, adet = d2.prototip_kur(g[None], [[]], sinif_sayisi=3)
    assert list(adet) == [0, 0, 0]


def test_patch_ata_esigin_altini_arka_plan_sayar():
    g = _bos_gomu()
    _bolge_koy(g, 0, 5, 0, 5, eksen=1)
    proto = np.zeros((1, BOYUT))
    proto[0, 1] = 1.0                            # yalnızca yön 1'i tanır
    atama, skor = d2.patch_ata(g, proto, en_az_benzerlik=0.55)
    assert (atama[:5, :5] == 0).all()            # bölge sınıfa atandı
    assert (atama[10:, 10:] == -1).all()         # arka plan elendi


def test_prototipli_kip_dogru_sinifta_dogru_kutu_uretir():
    g = _bos_gomu()
    _bolge_koy(g, 4, 8, 10, 16, eksen=1)         # sınıf 0 bölgesi
    _bolge_koy(g, 12, 18, 2, 6, eksen=2)         # sınıf 1 bölgesi
    proto = np.zeros((2, BOYUT))
    proto[0, 1] = 1.0
    proto[1, 2] = 1.0

    adaylar = d2.adaylari_uret(g, proto, en_az_benzerlik=0.55)
    bulunan = {sid: (cx, cy, w, h) for sid, cx, cy, w, h, _ in adaylar}
    assert set(bulunan) == {0, 1}

    cx, cy, w, h = bulunan[0]
    assert cx == pytest.approx(13 / IZGARA)
    assert cy == pytest.approx(6 / IZGARA)
    cx, cy, w, h = bulunan[1]
    assert cx == pytest.approx(4 / IZGARA)
    assert cy == pytest.approx(15 / IZGARA)


def test_tohumsuz_kip_sinif_atamaz():
    """Prototip yoksa bütün kutular ID 0 ve bu 'sınıfsız' demektir."""
    g = _bos_gomu()
    _bolge_koy(g, 6, 14, 6, 14, eksen=1)         # ortada bir nesne
    adaylar = d2.adaylari_uret(g, prototipler=None)
    assert adaylar, 'ön plan bulunamadı'
    assert {sid for sid, *_ in adaylar} == {0}


def test_on_plan_pca_kenari_arka_plan_sayar():
    """Nesne ortada, kenar arka plan — işaret düzeltmesi çalışmalı."""
    g = _bos_gomu()
    _bolge_koy(g, 7, 13, 7, 13, eksen=1)
    maske = d2.on_plan_pca(g)
    assert maske[10, 10], 'ortadaki nesne ön plan olmalı'
    assert not maske[0, 0], 'köşe arka plan olmalı'


def test_etiket_oku_bozuk_satiri_atlar(tmp_path):
    p = tmp_path / 'a.txt'
    p.write_text('0 0.5 0.5 0.2 0.2\nbozuk\n1 0.1 0.1 0.1 0.1\n',
                 encoding='utf-8')
    k = d2.etiket_oku(p)
    assert len(k) == 2
    assert k[0][0] == 0 and k[1][0] == 1


def test_olmayan_etiket_dosyasi_bos_doner(tmp_path):
    assert d2.etiket_oku(tmp_path / 'yok.txt') == []


def test_organ_siniflari_kutukten_gelir():
    """Sınıf sırası modeller.yaml'dan okunmalı, betikte sabit olmamalı."""
    s = d2.organ_siniflari('findik')
    assert s == ['Leaf', 'Nut', 'Husk', 'Branch', 'Flower']
    assert d2.organ_siniflari('olmayan_urun') == []
