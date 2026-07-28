"""Etiket onarımı: taşan kutuları kırpma, bozuk kutuları atma.

Bu hata sessizdir: etiket dosyasındaki her sayı 0-1 aralığında olduğu için
hiçbir doğrulayıcı uyarmaz, ama kutu görüntünün dışına taşar ve model kayık
hedeflerle eğitilir. Kurallar bu yüzden testle sabitlenir.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from etiket_temizle import dosya_temizle, kirp


def _kose(x, y, w, h):
    """Kutunun köşeleri — kırpma sonrası sınır kontrolü için."""
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


# ────────────────────────────────────────────────────────────── kırpma
def test_saglam_kutu_degismez():
    x, y, w, h = kirp(0.5, 0.5, 0.2, 0.2)
    assert (x, y, w, h) == pytest.approx((0.5, 0.5, 0.2, 0.2))


def test_sol_kenardan_tasan_kirpilir():
    x, y, w, h = kirp(0.1, 0.5, 0.4, 0.2)          # sol kenar -0.1
    x1, y1, x2, y2 = _kose(x, y, w, h)
    assert x1 >= 0 and x2 <= 1
    assert abs(x2 - 0.3) < 1e-9, 'sağ kenar korunmalı'
    assert abs(w - 0.3) < 1e-9, 'yalnızca taşan kısım kesilmeli'


def test_dort_kenardan_tasan_kirpilir():
    assert kirp(0.5, 0.5, 1.6, 1.6) == pytest.approx((0.5, 0.5, 1.0, 1.0)),         'tüm görüntüye oturmalı'


def test_merkez_yeniden_hesaplanir():
    """Kırpma merkezi kaydırmalı; yoksa kutu görünen nesneden kayık kalır."""
    x, _, _, _ = kirp(0.0, 0.5, 0.4, 0.2)          # yarısı dışarıda
    assert abs(x - 0.1) < 1e-9


def test_tamamen_disarida_olan_atilir():
    assert kirp(-0.5, 0.5, 0.2, 0.2) is None


def test_kirpinca_cizgiye_donen_atilir():
    """Kenara teğet kutu kırpılınca sıfır alana düşer — öğretecek bir şeyi yok."""
    assert kirp(0.0005, 0.5, 0.001, 0.2) is None


def test_kirpma_tekrarlanabilir():
    """Betik iki kez çalışırsa veri bozulmamalı."""
    bir = kirp(0.1, 0.5, 0.4, 0.2)
    assert kirp(*bir) == pytest.approx(bir)


# ─────────────────────────────────────────────────────── dosya düzeyi
def test_bozuk_kutular_atilir():
    satirlar = [
        '3 0.5 0.5 0.2 0.2',      # sağlam
        '3 0.5 0.5 0.0 0.2',      # genişlik sıfır
        '3 0.5 0.5 0.2 0.0',      # yükseklik sıfır
        '3 0.5',                  # eksik alan
        'bozuk satir',
    ]
    yeni, kirpilan, atilan = dosya_temizle(satirlar)
    assert len(yeni) == 1
    assert kirpilan == 0
    assert atilan == 4            # 2 sıfır boyut + eksik alan + metin satırı


def test_sinif_id_korunur():
    yeni, _, _ = dosya_temizle(['7 0.1 0.5 0.4 0.2'])
    assert yeni[0].split()[0] == '7', 'kırpma sınıfı değiştirmemeli'


def test_saglam_dosya_degismeden_gecer():
    satirlar = ['0 0.500000 0.500000 0.200000 0.200000']
    yeni, kirpilan, atilan = dosya_temizle(satirlar)
    assert (kirpilan, atilan) == (0, 0)
    assert [tuple(float(v) for v in s.split()[1:]) for s in yeni] ==            [tuple(float(v) for v in s.split()[1:]) for s in satirlar]


def test_tum_kutular_kirpma_sonrasi_sinir_icinde():
    satirlar = ['0 0.05 0.05 0.4 0.4', '1 0.95 0.95 0.4 0.4', '2 0.5 0.5 2.0 2.0']
    yeni, kirpilan, _ = dosya_temizle(satirlar)
    assert kirpilan == 3
    for s in yeni:
        x, y, w, h = (float(v) for v in s.split()[1:5])
        x1, y1, x2, y2 = _kose(x, y, w, h)
        assert -1e-6 <= x1 and x2 <= 1 + 1e-6
        assert -1e-6 <= y1 and y2 <= 1 + 1e-6
