"""Kaynak dataset'i master şemaya yerinde çevirme testleri.

Sınıf ID çakışması bu projedeki en sinsi hatadır: yanlış eşlenen bir ID
sessizce yanlış sınıfa eğitim yaptırır, sonuç ancak eğitim bittikten sonra
fark edilir. Bu yüzden dönüşüm kuralları testle sabitlenir.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import merge_datasets as md


MASTER = ['Angular Leafspot', 'Anthracnose Fruit Rot', 'Blossom Blight', 'Gray Mold',
          'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf',
          'strawberry_ripe', 'strawberry_semi_ripe', 'strawberry_unripe']


@pytest.fixture
def sahte_dataset(tmp_path):
    """Kaynakta Leaf Spot id 6, bizde id 4 — gerçek çakışma senaryosu."""
    kok = tmp_path / 'kaynak'
    (kok / 'train' / 'labels').mkdir(parents=True)
    (kok / 'data.yaml').write_text(yaml.dump({
        'nc': 4,
        'names': ['Gray Mold', 'Healthy-Leaf -Strawberry', 'Mulch', 'Leaf Spot'],
    }), encoding='utf-8')
    (kok / 'train' / 'labels' / 'a.txt').write_text(
        '0 0.5 0.5 0.2 0.2\n3 0.3 0.3 0.1 0.1\n', encoding='utf-8')
    (kok / 'train' / 'labels' / 'b.txt').write_text(
        '1 0.5 0.5 0.4 0.4\n2 0.1 0.1 0.1 0.1\n', encoding='utf-8')

    master = tmp_path / 'master.yaml'
    master.write_text(yaml.dump({'nc': len(MASTER),
                                 'names': dict(enumerate(MASTER))}), encoding='utf-8')
    return kok, master


def _oku(kok, ad):
    return (kok / 'train' / 'labels' / ad).read_text(encoding='utf-8').strip().splitlines()


def test_id_cakismasi_duzeltilir(sahte_dataset):
    """Kaynaktaki id 3 (Leaf Spot) master'da id 4 olmalı."""
    kok, master = sahte_dataset
    assert md.yerinde_uyarla(str(kok), str(master), None,
                             ['healthy leaf strawberry', 'mulch']) is True
    satirlar = _oku(kok, 'a.txt')
    assert satirlar[0].split()[0] == '3', 'Gray Mold id 3 kalmalı'
    assert satirlar[1].split()[0] == '4', 'Leaf Spot 3 → 4 olmalı'


def test_koordinatlar_korunur(sahte_dataset):
    kok, master = sahte_dataset
    md.yerinde_uyarla(str(kok), str(master), None, ['healthy leaf strawberry', 'mulch'])
    assert _oku(kok, 'a.txt')[0].split()[1:] == ['0.5', '0.5', '0.2', '0.2']


def test_atilan_siniflar_background_yapar(sahte_dataset):
    """Sadece atılan sınıf içeren dosya boşalmalı — görüntü background örneği olur."""
    kok, master = sahte_dataset
    md.yerinde_uyarla(str(kok), str(master), None, ['healthy leaf strawberry', 'mulch'])
    assert _oku(kok, 'b.txt') == [], 'healthy+mulch içeren dosya boşalmalı'
    assert (kok / 'train' / 'labels' / 'b.txt').exists(), 'dosya silinmemeli'


def test_yedek_alinir(sahte_dataset):
    kok, master = sahte_dataset
    md.yerinde_uyarla(str(kok), str(master), None, ['healthy leaf strawberry', 'mulch'])
    yedek = kok / 'train' / 'labels_orijinal' / 'a.txt'
    assert yedek.exists()
    assert yedek.read_text(encoding='utf-8').startswith('0 '), 'yedek ORİJİNAL olmalı'


def test_iki_kez_calistirinca_idler_kaymaz(sahte_dataset):
    """En kritik davranış: betik yanlışlıkla iki kez çalışırsa veri bozulmamalı."""
    kok, master = sahte_dataset
    md.yerinde_uyarla(str(kok), str(master), None, ['healthy leaf strawberry', 'mulch'])
    ilk = _oku(kok, 'a.txt')
    md.yerinde_uyarla(str(kok), str(master), None, ['healthy leaf strawberry', 'mulch'])
    assert _oku(kok, 'a.txt') == ilk, 'ikinci çalıştırma ID kaydırmamalı'


def test_kuru_calistirma_dosyaya_yazmaz(sahte_dataset):
    kok, master = sahte_dataset
    once = (kok / 'train' / 'labels' / 'a.txt').read_text(encoding='utf-8')
    md.yerinde_uyarla(str(kok), str(master), None,
                      ['healthy leaf strawberry', 'mulch'], kuru=True)
    assert (kok / 'train' / 'labels' / 'a.txt').read_text(encoding='utf-8') == once
    assert not (kok / 'train' / 'labels_orijinal').exists()


def test_data_yaml_master_semaya_gecer(sahte_dataset):
    kok, master = sahte_dataset
    md.yerinde_uyarla(str(kok), str(master), None, ['healthy leaf strawberry', 'mulch'])
    veri = yaml.safe_load((kok / 'data.yaml').read_text(encoding='utf-8'))
    assert veri['nc'] == len(MASTER)
    assert veri['names'][4] == 'Leaf Spot'


def test_bilinmeyen_sinif_sessizce_gecmez(tmp_path):
    """Eşlenemeyen sınıf hata vermeli; sessizce yanlış ID'ye düşerse eğitim bozulur."""
    kok = tmp_path / 'k'
    (kok / 'train' / 'labels').mkdir(parents=True)
    (kok / 'data.yaml').write_text(yaml.dump({'nc': 1, 'names': ['Uzayli Hastalik']}),
                                   encoding='utf-8')
    master = tmp_path / 'm.yaml'
    master.write_text(yaml.dump({'nc': len(MASTER), 'names': dict(enumerate(MASTER))}),
                      encoding='utf-8')
    with pytest.raises(ValueError, match='Eşlenemeyen'):
        md.yerinde_uyarla(str(kok), str(master), None, [])
