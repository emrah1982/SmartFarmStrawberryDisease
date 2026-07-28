"""Uzman dataset türetme: sınıf seçimi, ID yeniden numaralama, background sınırı."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import dataset_ayir as da

MASTER = {0: 'Angular Leafspot', 1: 'Anthracnose Fruit Rot', 2: 'Blossom Blight',
          3: 'Gray Mold', 4: 'Leaf Spot', 5: 'Powdery Mildew Fruit',
          6: 'Powdery Mildew Leaf', 7: 'strawberry_ripe',
          8: 'strawberry_semi_ripe', 9: 'strawberry_unripe'}


def _kaynak(tmp_path, kayitlar):
    """kayitlar: {dosya_adi: [(sinif_id, x,y,w,h), ...]}"""
    d = tmp_path / 'kaynak' / 'ornek' / 'train'
    (d / 'images').mkdir(parents=True)
    (d / 'labels').mkdir(parents=True)
    for ad, kutular in kayitlar.items():
        (d / 'images' / f'{ad}.jpg').write_bytes(b'sahte')
        satirlar = [f'{c} {x} {y} {w} {h}' for c, x, y, w, h in kutular]
        (d / 'labels' / f'{ad}.txt').write_text(chr(10).join(satirlar), encoding='utf-8')
    return tmp_path / 'kaynak'


def test_sadece_ilgili_siniflar_alinir(tmp_path):
    """Yaprak dataset'ine meyve sınıfı sızmamalı."""
    kaynak = _kaynak(tmp_path, {
        'a': [(4, 0.5, 0.5, 0.2, 0.2),    # Leaf Spot  -> alinir
              (7, 0.3, 0.3, 0.1, 0.1)],   # ripe       -> atilir
    })
    da.ayir(kaynak, tmp_path / 'hedef', MASTER, 0.0, kuru=False)
    etiket = (tmp_path / 'hedef' / 'leaf_disease' / 'train' / 'labels')
    icerik = list(etiket.glob('*.txt'))[0].read_text(encoding='utf-8').strip()
    assert len(icerik.splitlines()) == 1, 'yalnizca yaprak sinifi kalmali'


def test_idler_sifirdan_yeniden_numaralanir(tmp_path):
    """Her dataset bağımsız olmalı: ID'ler 0..n-1 aralığına çekilir."""
    kaynak = _kaynak(tmp_path, {'a': [(9, 0.5, 0.5, 0.2, 0.2)]})  # unripe = master 9
    da.ayir(kaynak, tmp_path / 'hedef', MASTER, 0.0, kuru=False)
    d = tmp_path / 'hedef' / 'fruit_ripeness'
    satir = list((d / 'train' / 'labels').glob('*.txt'))[0].read_text(encoding='utf-8')
    assert satir.split()[0] == '0', 'unripe olgunluk datasetinde ID 0 olmali'
    veri = yaml.safe_load((d / 'data.yaml').read_text(encoding='utf-8'))
    assert veri['names'][0] == 'strawberry_unripe'
    assert veri['nc'] == 3


def test_koordinatlar_korunur(tmp_path):
    kaynak = _kaynak(tmp_path, {'a': [(4, 0.25, 0.75, 0.1, 0.2)]})
    da.ayir(kaynak, tmp_path / 'hedef', MASTER, 0.0, kuru=False)
    satir = list((tmp_path / 'hedef' / 'leaf_disease' / 'train' / 'labels')
                 .glob('*.txt'))[0].read_text(encoding='utf-8').split()
    assert satir[1:5] == ['0.25', '0.75', '0.1', '0.2']


def test_gray_mold_iki_datasette_de_bulunur():
    """Kurşuni küf hem yaprakta hem meyvede görülür."""
    assert 'Gray Mold' in da.AYRIM['leaf_disease']
    assert 'Gray Mold' in da.AYRIM['fruit_disease']


def test_background_orani_sinirlanir(tmp_path):
    """Background örnekleri veriyi boğmamalı."""
    kayitlar = {f'ic{i}': [(4, 0.5, 0.5, 0.2, 0.2)] for i in range(10)}
    kayitlar.update({f'bos{i}': [(7, 0.5, 0.5, 0.2, 0.2)] for i in range(100)})
    kaynak = _kaynak(tmp_path, kayitlar)
    da.ayir(kaynak, tmp_path / 'hedef', MASTER, 0.20, kuru=False)
    gorseller = list((tmp_path / 'hedef' / 'leaf_disease' / 'train' / 'images').glob('*'))
    assert len(gorseller) == 12, '10 icerikli + 2 background (oran 0.20)'


def test_kuru_calistirma_yazmaz(tmp_path):
    kaynak = _kaynak(tmp_path, {'a': [(4, 0.5, 0.5, 0.2, 0.2)]})
    da.ayir(kaynak, tmp_path / 'hedef', MASTER, 0.0, kuru=True)
    assert not (tmp_path / 'hedef').exists()


def test_background_etiketi_bos_dosya(tmp_path):
    """Background: görüntü silinmez, etiketi boşalır."""
    kaynak = _kaynak(tmp_path, {'ic': [(4, 0.5, 0.5, 0.2, 0.2)],
                                'bos': [(7, 0.5, 0.5, 0.2, 0.2)]})
    da.ayir(kaynak, tmp_path / 'hedef', MASTER, 1.0, kuru=False)
    etiketler = sorted((tmp_path / 'hedef' / 'leaf_disease' / 'train' / 'labels').glob('*.txt'))
    icerikler = [f.read_text(encoding='utf-8').strip() for f in etiketler]
    assert '' in icerikler, 'background icin bos etiket dosyasi olmali'
    assert len(etiketler) == 2
