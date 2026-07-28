"""Epoch önerisi: eğriden plato/doyma tespiti.

"200 verelim, ezberlerse durur" yaklaşımı yanlış karar verdirir; öneri bu yüzden
ölçüme dayanır ve kuralları testle sabitlenir.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from epoch_oner import ANLAMLI_KAZANC, coz, egri_oku, oner


def _duz_egri(n, tepe_epoch, tepe=0.75):
    """tepe_epoch'a kadar yükselen, sonra sabit kalan eğri."""
    egri = []
    for e in range(1, n + 1):
        v = tepe * min(1.0, e / tepe_epoch)
        egri.append((e, round(v, 4)))
    return egri


def test_plato_tespit_edilir():
    d = coz(_duz_egri(200, 100))
    assert d['platoda'] is True
    assert d['sonda_hala_iyilesiyor'] is False


def test_sonda_iyilesme_tespit_edilir():
    """Son epoch'a kadar yükselen eğri 'hâlâ iyileşiyor' sayılmalı."""
    egri = [(e, round(0.4 + e * 0.004, 4)) for e in range(1, 51)]
    d = coz(egri)
    assert d['sonda_hala_iyilesiyor'] is True
    assert d['platoda'] is False


def test_argmax_sonda_ama_kazanc_yoksa_plato():
    """Gürültü yüzünden en iyi son epochta çıkabilir; bu iyileşme değildir."""
    egri = _duz_egri(200, 100)
    egri[-1] = (200, egri[-1][1] + 0.0001)      # ölçüm gürültüsü kadar artış
    d = coz(egri)
    assert d['en_iyi_epoch'] == 200
    assert d['sonda_hala_iyilesiyor'] is False, 'gürültü iyileşme sayılmamalı'
    assert d['platoda'] is True


def test_doyma_noktasi_bulunur():
    d = coz(_duz_egri(200, 100))
    assert 90 <= d['esikler'][0.98] <= 100


def test_en_uzun_duraklama_olculur():
    egri = [(1, 0.50), (2, 0.60), (3, 0.60), (4, 0.60), (5, 0.60), (6, 0.70)]
    assert coz(egri)['en_uzun_duraklama'] == 3


def test_patience_duraklamadan_buyuk_onerilir():
    """patience, geçici duraklamadan kısa olursa eğitim erken kesilir."""
    d = coz(_duz_egri(200, 100))
    o = oner([('x', d)], ince_ayar=True)
    assert o['patience'] > d['en_uzun_duraklama']


def test_ince_ayar_sifirdan_kisa_onerir():
    d = coz(_duz_egri(200, 100))
    assert oner([('x', d)], ince_ayar=True)['epochs'] < oner([('x', d)], False)['epochs']


def test_gecmis_yoksa_makul_varsayilan():
    o = oner([], ince_ayar=False)
    assert o['epochs'] >= 100 and o['patience'] >= 20


def test_gercek_csv_okunur(tmp_path):
    csv = tmp_path / 'results.csv'
    satirlar = ['epoch,metrics/mAP50-95(B)', '1,0.10', '2,0.20', '3,0.25']
    csv.write_text(chr(10).join(satirlar) + chr(10), encoding='utf-8')
    assert egri_oku(csv) == [(1, 0.10), (2, 0.20), (3, 0.25)]


def test_bilinmeyen_sutunlu_csv_bos_doner(tmp_path):
    csv = tmp_path / 'results.csv'
    csv.write_text('epoch,loss' + chr(10) + '1,0.5' + chr(10), encoding='utf-8')
    assert egri_oku(csv) == []
