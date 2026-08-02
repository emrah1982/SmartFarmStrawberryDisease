"""on_etiket_gdino.py — model gerektirmeyen temizlik katmani.

Mevcut on-etiketlerde olculen uc hata bu katmani dogurdu:
  - kadrajin %64'unu kaplayan sabit kutu
  - iki dev yatay serit 'leaf' diye isaretlenmis
  - ic ice, ayni nesneyi gosteren kutular
"""
import importlib.util
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location(
    'on_etiket_gdino', KOK / 'scripts' / 'on_etiket_gdino.py')
gd = importlib.util.module_from_spec(_s)
sys.modules['on_etiket_gdino'] = gd
_s.loader.exec_module(gd)


def test_iou_ayni_kutuda_bir():
    k = (0.1, 0.1, 0.5, 0.5)
    assert gd.iou(k, k) == pytest.approx(1.0)


def test_iou_ayrik_kutuda_sifir():
    assert gd.iou((0, 0, 0.2, 0.2), (0.5, 0.5, 0.9, 0.9)) == 0.0


def test_icerme_tam_kapsamada_bir():
    kucuk = (0.4, 0.4, 0.5, 0.5)
    buyuk = (0.1, 0.1, 0.9, 0.9)
    assert gd.icerme(kucuk, buyuk) == pytest.approx(1.0)


def test_tum_kareyi_kaplayan_kutu_atilir():
    """Kadrajin TAMAMI bir organ ornegi degildir."""
    kutular = [((0.0, 0.0, 1.0, 1.0), 0.9, 'Leaf')]        # %100
    tutulan, atilan = gd.temizle(kutular)
    assert tutulan == []
    assert 'alan çok büyük' in atilan[0][3]


def test_mesru_yakin_cekim_kutusu_ATILMAZ():
    """Alan esigi dejenere kutuyu mesru olandan AYIRAMAZ - olculdu.

    21.106 gercek saha organ kutusunda Leaf p90 = %66, max = %98.9.
    Yakin cekimde tek yaprak kareyi doldurur. %64'u yakalayan bir esik
    gercek kutularin %17.3'unu de atardi. Bu test, esigin daraltilip
    mesru kutulari elemeye baslamasini engeller.
    """
    p90_yaprak = (0.09, 0.09, 0.90, 0.90)          # ~%66 alan
    tutulan, _ = gd.temizle([(p90_yaprak, 0.9, 'Leaf')])
    assert len(tutulan) == 1, 'mesru buyuk yaprak kutusu atilmamali'


def test_cok_kucuk_kutu_atilir():
    tutulan, atilan = gd.temizle([((0.5, 0.5, 0.51, 0.51), 0.9, 'Leaf')])
    assert tutulan == []
    assert 'alan çok küçük' in atilan[0][3]


def test_nms_ayni_nesnenin_kopyasini_siler():
    a = ((0.2, 0.2, 0.5, 0.5), 0.9, 'Leaf')
    b = ((0.21, 0.21, 0.51, 0.51), 0.7, 'Leaf')     # neredeyse ayni
    tutulan, atilan = gd.temizle([a, b])
    assert len(tutulan) == 1
    assert tutulan[0][1] == 0.9                      # yuksek skor kalir


def test_farkli_siniflar_nms_ile_birbirini_silmez():
    a = ((0.2, 0.2, 0.5, 0.5), 0.9, 'Leaf')
    b = ((0.21, 0.21, 0.51, 0.51), 0.7, 'Husk')
    tutulan, _ = gd.temizle([a, b])
    assert len(tutulan) == 2


def test_ic_ice_kutu_ayiklanir():
    """NMS bunu KACIRIR: alanlar cok farkli oldugu icin IoU dusuk kalir."""
    buyuk = ((0.10, 0.10, 0.70, 0.70), 0.9, 'Leaf')
    kucuk = ((0.30, 0.30, 0.45, 0.45), 0.8, 'Leaf')
    assert gd.iou(buyuk[0], kucuk[0]) < gd.NMS_IOU     # NMS yakalayamaz
    assert gd.icerme(kucuk[0], buyuk[0]) > gd.ICERME_ESIGI
    tutulan, atilan = gd.temizle([buyuk, kucuk])
    assert len(tutulan) == 1
    assert any('iç içe' in x[3] for x in atilan)


def test_ayrik_iki_yaprak_ikisi_de_kalir():
    a = ((0.05, 0.05, 0.30, 0.30), 0.8, 'Leaf')
    b = ((0.60, 0.60, 0.90, 0.90), 0.7, 'Leaf')
    tutulan, _ = gd.temizle([a, b])
    assert len(tutulan) == 2


def test_yolo_satiri_merkez_ve_boyut_yazar():
    s = gd.yolo_satiri((0.2, 0.3, 0.6, 0.7), 2)
    p = s.split()
    assert p[0] == '2'
    assert float(p[1]) == pytest.approx(0.4)
    assert float(p[2]) == pytest.approx(0.5)
    assert float(p[3]) == pytest.approx(0.4)
    assert float(p[4]) == pytest.approx(0.4)


def test_saglik_etiketi_dosya_adindan_okunur():
    assert gd.saglik_etiketi('00001_Diseased_1458.jpg') == 'diseased'
    assert gd.saglik_etiketi('Healthy (12).jpeg') == 'healthy'
    assert gd.saglik_etiketi('IMG_0725.jpg') is None


def test_prompt_eslemesi_kutukteki_siniflara_gider():
    """Prompt kutukte olmayan organ uretmemeli."""
    kutuk = gd.organ_siniflari('findik')
    assert kutuk == ['Leaf', 'Nut', 'Husk', 'Branch', 'Flower']
    assert set(gd.ORGAN_PROMPT.values()) <= set(kutuk)


def test_cluster_promptu_husk_e_gider():
    """Gorunen organ zuruftur; findik kabugu onun icindedir."""
    assert gd.ORGAN_PROMPT['green hazelnut cluster'] == 'Husk'
