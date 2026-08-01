"""sifir_atis_siniflandir.py + dinov2_egit.py — model gerektirmeyen mantık.

torch/transformers/CLIP bu depoda kurulu değil. Test edilen şey ölçüm
katmanı: doğruluk, precision/recall/f1, karışıklık matrisi ve TABAN
ÇİZGİSİ. Taban çizgisi kritiktir — bir yöntemin "%80 doğruluk" demesi,
verinin %80'i zaten tek sınıfsa hiçbir şey ifade etmez.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

KOK = Path(__file__).resolve().parents[1]


def _yukle(ad):
    s = importlib.util.spec_from_file_location(ad, KOK / 'scripts' / f'{ad}.py')
    m = importlib.util.module_from_spec(s)
    sys.modules[ad] = m
    s.loader.exec_module(m)
    return m


sa = _yukle('sifir_atis_siniflandir')
de = _yukle('dinov2_egit')

SINIFLAR = ['diseased_cluster', 'healthy_cluster']


# ───────────────────────────────────────────────── ölçüm katmanı

def test_kusursuz_tahmin_dogruluk_bir():
    g = ['diseased_cluster'] * 3 + ['healthy_cluster'] * 2
    o = sa.olcumler(g, list(g), SINIFLAR)
    assert o['accuracy'] == 1.0
    for s in SINIFLAR:
        assert o['siniflar'][s]['f1'] == 1.0


def test_hep_cogunluk_diyen_yontem_yakalanir():
    """Dengesiz veride 'hep çoğunluk' yüksek accuracy verir — f1 vermez."""
    g = ['diseased_cluster'] * 9 + ['healthy_cluster'] * 1
    t = ['diseased_cluster'] * 10
    o = sa.olcumler(g, t, SINIFLAR)
    assert o['accuracy'] == pytest.approx(0.9)
    # Ama azınlık sınıfı tamamen kaçırıldı:
    assert o['siniflar']['healthy_cluster']['recall'] == 0.0
    assert o['siniflar']['healthy_cluster']['f1'] == 0.0


def test_taban_cizgisi_cogunluk_oranini_verir():
    g = ['diseased_cluster'] * 9 + ['healthy_cluster'] * 1
    ad, oran = sa.taban_cizgisi(g)
    assert ad == 'diseased_cluster'
    assert oran == pytest.approx(0.9)


def test_karisiklik_matrisi_satir_gercek_sutun_tahmin():
    g = ['diseased_cluster', 'diseased_cluster', 'healthy_cluster']
    t = ['diseased_cluster', 'healthy_cluster', 'healthy_cluster']
    m = sa.karisiklik(g, t, SINIFLAR)
    assert m[0, 0] == 1        # diseased -> diseased
    assert m[0, 1] == 1        # diseased -> healthy (hata)
    assert m[1, 1] == 1        # healthy  -> healthy
    assert m[1, 0] == 0


def test_precision_recall_ayri_hesaplanir():
    # diseased'i asiri tahmin eden model: recall yuksek, precision dusuk
    g = ['diseased_cluster'] * 2 + ['healthy_cluster'] * 4
    t = ['diseased_cluster'] * 5 + ['healthy_cluster'] * 1
    o = sa.olcumler(g, t, SINIFLAR)['siniflar']['diseased_cluster']
    assert o['recall'] == pytest.approx(1.0)
    assert o['precision'] == pytest.approx(2 / 5)


def test_bos_sinif_sifira_bolunmez():
    g = ['diseased_cluster'] * 3
    t = ['diseased_cluster'] * 3
    o = sa.olcumler(g, t, SINIFLAR)
    assert o['siniflar']['healthy_cluster']['f1'] == 0.0


# ───────────────────────────────────────────────── prototip

def test_prototip_kendi_sinifini_bulur():
    tohum = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    tohum /= np.linalg.norm(tohum, axis=1, keepdims=True)
    etiket = ['diseased_cluster', 'diseased_cluster',
              'healthy_cluster', 'healthy_cluster']
    proto = sa.prototip_kur(tohum, etiket, SINIFLAR)

    sinav = np.array([[1.0, 0.05], [0.05, 1.0]])
    sinav /= np.linalg.norm(sinav, axis=1, keepdims=True)
    tahmin, _ = sa.prototip_siniflandir(sinav, proto, SINIFLAR)
    assert tahmin == ['diseased_cluster', 'healthy_cluster']


def test_tohumsuz_sinifin_prototipi_sifir_kalir():
    tohum = np.array([[1.0, 0.0]])
    proto = sa.prototip_kur(tohum, ['diseased_cluster'], SINIFLAR)
    assert np.allclose(proto[1], 0)


# ───────────────────────────────────────────────── dinov2_egit veri katmanı

@pytest.fixture
def veri(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    adet = {'train': (30, 20), 'val': (6, 4), 'test': (7, 3)}
    for bolum, (nd, nh) in adet.items():
        for sinif, n in (('diseased_cluster', nd), ('healthy_cluster', nh)):
            d = tmp_path / bolum / sinif
            d.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(
                    rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
                ).save(d / f'{i}.jpeg')
    return tmp_path


def test_sinif_sirasi_alfabetiktir(veri):
    """Sıra modele gömülür; alfabetik olmalı ki HF ile uyuşsun."""
    assert de.siniflari_bul(veri) == ['diseased_cluster', 'healthy_cluster']


def test_bolum_sayimi_val_klasorunu_okur(veri):
    say = de.bolum_sayimi(veri, de.siniflari_bul(veri))
    assert set(say) == {'train', 'val', 'test'}
    assert say['train']['diseased_cluster'] == 30
    assert sum(say['test'].values()) == 10


def test_valid_adli_klasor_bulunamaz(tmp_path):
    """'valid' sozlesmeye aykiri; sessizce kabul EDILMEMELI."""
    from PIL import Image
    for sinif in ('diseased_cluster', 'healthy_cluster'):
        for bolum in ('train', 'valid'):
            d = tmp_path / bolum / sinif
            d.mkdir(parents=True)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                d / 'a.jpeg')
    say = de.bolum_sayimi(tmp_path, de.siniflari_bul(tmp_path))
    assert 'val' not in say, "'valid' klasoru 'val' sayilmamali"


def test_taban_cizgisi_test_bolumunden_hesaplanir(veri):
    say = de.bolum_sayimi(veri, de.siniflari_bul(veri))
    t = de.taban_cizgisi(say, 'test')
    assert t['sinif'] == 'diseased_cluster'
    assert t['dogruluk'] == pytest.approx(0.7)


def test_rapor_taban_cizgisini_yazar(veri):
    siniflar = de.siniflari_bul(veri)
    say = de.bolum_sayimi(veri, siniflar)
    metin = de.rapor_metni(veri, siniflar, say, de.taban_cizgisi(say))
    assert 'TABAN' in metin.upper() or 'Taban' in metin
    assert '0.7000' in metin
    assert 'üst sınır' in metin          # sizinti uyarisi korunmali


# ───────────────────────────────────────────────── otomatik_etiketle
#
# Iki yaygin hatayi engelleyen denetimler. Ikisi de SESSIZ hatadir:
#   - COCO modeli yaprak bulamaz ama hata da vermez, bos sonuc doner
#   - Kararsiz argmax yuksek guvenli gorunur

oe = _yukle('otomatik_etiketle')


def test_coco_modeli_bulucu_olarak_reddedilir():
    coco = ['person', 'bicycle', 'car', 'dog', 'potted plant', 'chair',
            'bottle', 'tv']
    uygun, gerekce = oe.bulucu_siniflarini_denetle(coco)
    assert uygun is False
    assert 'COCO' in gerekce


def test_organ_modeli_bulucu_olarak_kabul_edilir():
    uygun, gerekce = oe.bulucu_siniflarini_denetle(['Flower', 'Fruit', 'Leaf'])
    assert uygun is True
    assert 'leaf' in gerekce


def test_findik_organ_siniflari_da_kabul_edilir():
    uygun, _ = oe.bulucu_siniflarini_denetle(
        ['Leaf', 'Nut', 'Husk', 'Branch', 'Flower'])
    assert uygun is True


def test_bos_sinif_listesi_reddedilir():
    uygun, gerekce = oe.bulucu_siniflarini_denetle([])
    assert uygun is False
    assert 'boş' in gerekce


def test_esik_altindaki_skor_bilinmeyen_olur():
    ad, guven, _ = oe.karar_ver(np.array([0.31, 0.20]), SINIFLAR, esik=0.50)
    assert ad == oe.BILINMEYEN
    assert guven == pytest.approx(0.31)


def test_kararsiz_argmax_bilinmeyen_olur():
    """Iki skor birbirine cok yakinsa secim guvenilmez."""
    ad, _, not_ = oe.karar_ver(np.array([0.92, 0.90]), SINIFLAR,
                               esik=0.50, ayrim_esigi=0.05)
    assert ad == oe.BILINMEYEN
    assert 'yakın' in not_


def test_net_ayrimda_sinif_atanir():
    ad, guven, not_ = oe.karar_ver(np.array([0.92, 0.31]), SINIFLAR,
                                   esik=0.50, ayrim_esigi=0.05)
    assert ad == 'diseased_cluster'
    assert guven == pytest.approx(0.92)
    assert not_ == ''


def test_kutu_normalize_merkez_ve_boyut_verir():
    # 200x100 goruntude (50,20)-(150,60) kutusu
    cx, cy, w, h = oe.kutu_normalize((50, 20, 150, 60), 200, 100)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.4)
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.4)


def test_kirpma_pay_birakir():
    from PIL import Image
    im = Image.new('RGB', (100, 100))
    kr = oe.kirp(im, (0.5, 0.5, 0.2, 0.2), pay=0.0)
    assert kr.size == (20, 20)
    kr2 = oe.kirp(im, (0.5, 0.5, 0.2, 0.2), pay=0.50)
    assert kr2.size == (30, 30)      # %50 pay -> 20 -> 30


def test_kenardaki_kutu_goruntu_disina_tasmaz():
    from PIL import Image
    im = Image.new('RGB', (100, 100))
    kr = oe.kirp(im, (0.02, 0.02, 0.2, 0.2), pay=0.5)
    assert kr is not None
    assert kr.size[0] <= 100 and kr.size[1] <= 100


def test_sifir_alanli_kutu_none_doner():
    from PIL import Image
    im = Image.new('RGB', (100, 100))
    assert oe.kirp(im, (0.5, 0.5, 0.0, 0.0), pay=0.0) is None
