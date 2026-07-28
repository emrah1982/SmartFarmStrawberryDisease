"""Model karşılaştırma raporunun karar mantığı.

Bu rapor "yeni modeli dağıtayım mı" sorusuna cevap verir. Yanlış karar ya
iyileşmeyi çöpe atar ya da gerilemiş bir modeli sahaya sokar; bu yüzden eşikler
ve gerileme tespiti testle sabitlenir.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from model_karsilastir import ONEMLI_FARK, rapor


def _sonuc(model, genel, siniflar):
    """{'mAP50-95': x} sözlüklerine çevirir."""
    return {
        'model': model,
        'genel': {'mAP50': genel, 'mAP50-95': genel, 'precision': genel, 'recall': genel},
        'sinif': {ad: {'mAP50': v, 'mAP50-95': v, 'precision': v, 'recall': v}
                  for ad, v in siniflar.items()},
    }


def test_genel_iyilesme_ve_gerileme_yoksa_dagitilabilir(capsys):
    eski = _sonuc('eski.pt', 0.50, {'Gray Mold': 0.50, 'Leaf Spot': 0.40})
    yeni = _sonuc('yeni.pt', 0.58, {'Gray Mold': 0.57, 'Leaf Spot': 0.49})
    s = rapor(eski, yeni)
    assert s['gerileyen'] == []
    assert set(s['iyilesen']) == {'Gray Mold', 'Leaf Spot'}
    assert 'Dağıtıma alınabilir' in capsys.readouterr().out


def test_gerileyen_sinif_raporlanir(capsys):
    """Toplam artarken tek sınıf gerileyebilir — ortalama bunu gizler."""
    eski = _sonuc('eski.pt', 0.50, {'Gray Mold': 0.50, 'Anthracnose Fruit Rot': 0.60})
    yeni = _sonuc('yeni.pt', 0.55, {'Gray Mold': 0.70, 'Anthracnose Fruit Rot': 0.40})
    s = rapor(eski, yeni)
    assert s['gerileyen'] == ['Anthracnose Fruit Rot']
    cikti = capsys.readouterr().out
    assert 'GERİLEDİ' in cikti
    assert 'forgetting' in cikti, 'sebep açıklanmalı'
    assert 'bazı sınıflar' in cikti, 'karar bu durumu uyarmalı'


def test_kotulesen_model_reddedilir(capsys):
    eski = _sonuc('eski.pt', 0.60, {'Gray Mold': 0.60})
    yeni = _sonuc('yeni.pt', 0.50, {'Gray Mold': 0.50})
    rapor(eski, yeni)
    assert 'Eski modeli koruyun' in capsys.readouterr().out


def test_gurultu_farki_degisim_sayilmaz(capsys):
    """Aynı model iki kez ölçülse bile küçük oynamalar olur."""
    kucuk = ONEMLI_FARK / 2
    eski = _sonuc('eski.pt', 0.50, {'Gray Mold': 0.50})
    yeni = _sonuc('yeni.pt', 0.50 + kucuk, {'Gray Mold': 0.50 + kucuk})
    s = rapor(eski, yeni)
    assert s['gerileyen'] == [] and s['iyilesen'] == []
    assert 'Anlamlı fark yok' in capsys.readouterr().out


def test_yalnizca_bir_modelde_olan_sinif_cokmez(capsys):
    """Sınıf listesi değiştiyse rapor yine de üretilmeli."""
    eski = _sonuc('eski.pt', 0.50, {'Gray Mold': 0.50})
    yeni = _sonuc('yeni.pt', 0.55, {'Gray Mold': 0.56, 'Spider Mites': 0.30})
    rapor(eski, yeni)
    assert 'yalnızca bir modelde var' in capsys.readouterr().out


def test_olcut_secilebilir(capsys):
    eski = {'model': 'e', 'genel': {'mAP50': 0.90, 'mAP50-95': 0.40},
            'sinif': {'Gray Mold': {'mAP50': 0.90, 'mAP50-95': 0.40}}}
    yeni = {'model': 'y', 'genel': {'mAP50': 0.91, 'mAP50-95': 0.55},
            'sinif': {'Gray Mold': {'mAP50': 0.91, 'mAP50-95': 0.55}}}
    s = rapor(eski, yeni, olcut='mAP50')
    assert abs(s['genel_fark'] - 0.01) < 1e-9
    assert s['olcut'] == 'mAP50'
