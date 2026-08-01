"""siniflandirma_paketi.py — sınıflandırma paketi standarda çevirme.

Tespit paketinden iki yapısal farkı korumak için yazıldı:
  1. Bölüm klasörü 'val' olmalı, 'valid' DEĞİL. Sınıflandırmada data.yaml
     yoktur; klasör adı doğrudan sözleşmedir ve 'valid' yazılırsa
     Ultralytics doğrulama bölümünü sessizce atlar.
  2. Sızıntı, artırım kopyası yerine algısal hash kümesiyle önlenir.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

KOK = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'siniflandirma_paketi', KOK / 'scripts' / 'siniflandirma_paketi.py')
sp = importlib.util.module_from_spec(_spec)
sys.modules['siniflandirma_paketi'] = sp
_spec.loader.exec_module(sp)


def _goruntu(yol: Path, tohum: int, boyut=64):
    rng = np.random.default_rng(tohum)
    veri = rng.integers(0, 255, (boyut, boyut, 3), dtype=np.uint8)
    yol.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(veri).save(yol)


@pytest.fixture
def duz_paket(tmp_path):
    """Sınıfı DOSYA ADINDA taşıyan düz klasör — 286646.zip düzeni."""
    for i in range(1, 21):
        _goruntu(tmp_path / f'Diseased ({i}).jpeg', i)
    for i in range(1, 21):
        _goruntu(tmp_path / f'Healthy ({i}).jpeg', 100 + i)
    return tmp_path


ESLESME = {'Diseased': 'diseased_cluster', 'Healthy': 'healthy_cluster'}
DESEN = r'^([A-Za-z_ ]+?)\s*\('


def _siniflar(kok):
    g = sp.goruntuleri_topla(kok)
    return g, [sp.sinifi_coz(x, kok, DESEN, ESLESME) for x in g]


def test_sinif_dosya_adindan_cozulur(duz_paket):
    _, s = _siniflar(duz_paket)
    assert set(s) == {'diseased_cluster', 'healthy_cluster'}
    assert s.count('diseased_cluster') == 20


def test_sinif_klasorden_de_cozulur(tmp_path):
    """Klasörlü paket geldiğinde üst klasör adı kazanır."""
    _goruntu(tmp_path / 'Healthy' / 'a.jpeg', 1)
    _goruntu(tmp_path / 'Diseased' / 'b.jpeg', 2)
    g, s = _siniflar(tmp_path)
    assert sorted(s) == ['diseased_cluster', 'healthy_cluster']


def test_bolum_adi_val_olmali_valid_degil(duz_paket):
    """Sınıflandırmada data.yaml yok; 'valid' sessizce atlanır."""
    g, s = _siniflar(duz_paket)
    kume = list(range(len(g)))
    bolum = sp.bol(kume, s, (0.70, 0.15, 0.15), tohum=0)
    assert set(bolum) <= {'train', 'val', 'test'}
    assert 'valid' not in set(bolum)


def test_yakin_kopyalar_ayni_bolmede_kalir(tmp_path):
    """Aynı görüntünün kopyaları bölmelere DAĞILMAMALI."""
    for i in range(9):
        _goruntu(tmp_path / f'Healthy ({i}).jpeg', i)
    # 6 tanesi tek bir görüntünün birebir kopyası
    ham = (tmp_path / 'Healthy (0).jpeg').read_bytes()
    for i in range(20, 26):
        (tmp_path / f'Healthy ({i}).jpeg').write_bytes(ham)

    g, s = _siniflar(tmp_path)
    bit = np.array([sp.dhash(x.read_bytes()) for x in g])
    kume = sp.kopya_kumeleri(np.packbits(bit, axis=1), sp.KOPYA_ESIGI)
    bolum = sp.bol(kume, s, (0.70, 0.15, 0.15), tohum=0)

    kopya_kume = {kume[i] for i, x in enumerate(g)
                  if x.read_bytes() == ham}
    assert len(kopya_kume) == 1, 'birebir kopyalar tek kümede olmalı'
    bolmeler = {bolum[i] for i, k in enumerate(kume) if k in kopya_kume}
    assert len(bolmeler) == 1, 'bir küme tek bölmede kalmalı'


def test_kume_hicbir_bolme_ciftinde_paylasilmaz(duz_paket):
    g, s = _siniflar(duz_paket)
    bit = np.array([sp.dhash(x.read_bytes()) for x in g])
    kume = sp.kopya_kumeleri(np.packbits(bit, axis=1), sp.KOPYA_ESIGI)
    bolum = sp.bol(kume, s, (0.70, 0.15, 0.15), tohum=0)

    kumeler = {b: {kume[i] for i, x in enumerate(bolum) if x == b}
               for b in ('train', 'val', 'test')}
    assert not (kumeler['train'] & kumeler['val'])
    assert not (kumeler['train'] & kumeler['test'])
    assert not (kumeler['val'] & kumeler['test'])


def test_her_sinif_her_bolmede_temsil_edilir(duz_paket):
    g, s = _siniflar(duz_paket)
    bolum = sp.bol(list(range(len(g))), s, (0.70, 0.15, 0.15), tohum=0)
    for b in ('train', 'val', 'test'):
        bulunan = {s[i] for i, x in enumerate(bolum) if x == b}
        assert bulunan == {'diseased_cluster', 'healthy_cluster'}, \
            f'{b} bölmesinde eksik sınıf var: {bulunan}'


def test_dhash_ayni_goruntude_ayni_farkli_goruntude_farkli(tmp_path):
    _goruntu(tmp_path / 'a.jpeg', 1)
    _goruntu(tmp_path / 'b.jpeg', 2)
    a = sp.dhash((tmp_path / 'a.jpeg').read_bytes())
    a2 = sp.dhash((tmp_path / 'a.jpeg').read_bytes())
    b = sp.dhash((tmp_path / 'b.jpeg').read_bytes())
    assert (a == a2).all()
    assert (a != b).sum() > sp.KOPYA_ESIGI
