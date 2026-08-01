"""harici_paket_duzelt.py — arka plana alma ve sızıntı denetimi.

'Healthy' gibi sınıflar bu projede uzman modele SINIF olarak verilmez;
sağlıklı durumu organ modelinden türetilir (docs/MIMARI.md). Ama görüntü
atılmaz — etiketsiz görüntü negatif örnektir. Bu testler o davranışı korur.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'harici_paket_duzelt', KOK / 'scripts' / 'harici_paket_duzelt.py')
hpd = importlib.util.module_from_spec(_spec)
sys.modules['harici_paket_duzelt'] = hpd
_spec.loader.exec_module(hpd)


@pytest.fixture
def paket(tmp_path):
    """3 sınıflı küçük paket: 0=Leaf Spot, 1=Healthy Leaf, 2=Blight."""
    etiketler = {
        'a': ['0 0.5 0.5 0.2 0.2', '2 0.1 0.1 0.1 0.1'],
        'b': ['1 0.5 0.5 0.4 0.4'],                      # yalnızca Healthy
        'c': ['2 0.3 0.3 0.2 0.2', '1 0.7 0.7 0.1 0.1'],
    }
    for bolum in ('train', 'valid'):
        (tmp_path / bolum / 'images').mkdir(parents=True)
        (tmp_path / bolum / 'labels').mkdir(parents=True)
        for ad, satirlar in etiketler.items():
            (tmp_path / bolum / 'images' / f'{bolum}_{ad}.jpg').write_bytes(b'x')
            (tmp_path / bolum / 'labels' / f'{bolum}_{ad}.txt').write_text(
                '\n'.join(satirlar) + '\n', encoding='utf-8')
    return tmp_path


SINIFLAR = ['Leaf Spot', 'Healthy Leaf', 'Blight']


def test_healthy_sinifi_silinir_id_ler_sikisir(paket):
    bolumler = hpd.bolumleri_topla(paket)
    kalan, ist = hpd.arka_plana_al(bolumler, SINIFLAR, ['Healthy Leaf'])

    assert kalan == ['Leaf Spot', 'Blight']
    assert ist['bulunamayan'] == []
    assert ist['silinen_kutu'] == 4          # iki bölümde ikişer Healthy kutusu

    # Blight 2 -> 1 kaydı; kalan kutular yeni ID'yi kullanmalı
    satirlar = (paket / 'train' / 'labels' / 'train_a.txt').read_text(
        encoding='utf-8').split('\n')
    assert satirlar[0].startswith('0 ')      # Leaf Spot yerinde kaldı
    assert satirlar[1].startswith('1 ')      # Blight 2 -> 1


def test_tamamen_bosalan_goruntu_silinmez_background_kalir(paket):
    bolumler = hpd.bolumleri_topla(paket)
    _, ist = hpd.arka_plana_al(bolumler, SINIFLAR, ['Healthy Leaf'])

    assert ist['bosalan_goruntu'] == 2       # 'b' her iki bölümde de boşaldı
    assert (paket / 'train' / 'images' / 'train_b.jpg').exists()
    assert (paket / 'train' / 'labels' / 'train_b.txt').read_text(
        encoding='utf-8') == ''


def test_olmayan_sinif_sessizce_gecilmez(paket):
    """Yazım hatası sessiz no-op'a dönmemeli — çağıran hata verebilsin."""
    bolumler = hpd.bolumleri_topla(paket)
    kalan, ist = hpd.arka_plana_al(bolumler, SINIFLAR, ['Healty Leaf'])

    assert ist['bulunamayan'] == ['healty leaf']
    assert kalan == SINIFLAR
    assert ist['silinen_kutu'] == 0


def test_buyuk_kucuk_harf_duyarsiz(paket):
    bolumler = hpd.bolumleri_topla(paket)
    kalan, ist = hpd.arka_plana_al(bolumler, SINIFLAR, ['healthy leaf'])

    assert kalan == ['Leaf Spot', 'Blight']
    assert ist['bulunamayan'] == []


def test_bosalan_goruntu_kendi_grubunda_bolunur(paket):
    """Etiketsiz görüntüler '?' kovasına düşer, bir bölmede yığılmaz."""
    bolumler = hpd.bolumleri_topla(paket)
    hpd.arka_plana_al(bolumler, SINIFLAR, ['Healthy Leaf'])

    yeni, _ = hpd.yeniden_bol(bolumler, (0.5, 0.5, 0.0), tohum=0)
    hepsi = [g.name for c in yeni.values() for g, _ in c]
    assert len(hepsi) == 6
    assert len(set(hepsi)) == 6              # kopya yok, kayıp yok


# ─────────────────────────────────────────────────────────────────────────
# Alan tespiti — bir paket ROI boru hattına mi girer, ayri akisa mi?
#
# Ölçülen iki hata bu kuralı doğurdu: bocek_teshis ve hazelnut detection v9
# ikisi de boru hattına bağlanmak üzereydi (docs/HATA-YONETIMI.md § 2.6).
# ─────────────────────────────────────────────────────────────────────────

_spec2 = importlib.util.spec_from_file_location(
    'imgsz_oner', KOK / 'scripts' / 'imgsz_oner.py')
imgsz_oner = importlib.util.module_from_spec(_spec2)
sys.modules['imgsz_oner'] = imgsz_oner
_spec2.loader.exec_module(imgsz_oner)


def _olcum(kutu_sayilari, merkez, alan=0.2):
    """alan_tespiti'nin bekledigi olcum sozlugunu kurar."""
    n = sum(kutu_sayilari)
    return {
        'kutu_sayisi': sorted(kutu_sayilari),
        'merkez_kacikligi': sorted([merkez] * max(n, 1)),
        'kutu_alani': sorted([alan] * max(n, 1)),
    }


def test_tek_kutulu_studyo_verisi_boru_hattina_girmez():
    """findik_kalite: her goruntude tam 1 kutu, kadraj ortasinda."""
    a = imgsz_oner.alan_tespiti(_olcum([1] * 20, 0.035, alan=0.048))
    assert a['alan'] == 'stüdyo/tek nesne'
    assert a['boru_hattina_uygun'] is False


def test_makro_verisi_boru_hattina_girmez():
    """bocek_teshis: cogu tek kutulu ve ortalanmis, ara sira coklu."""
    kutular = [1] * 87 + [3] * 13          # %13 cok kutulu
    a = imgsz_oner.alan_tespiti(_olcum(kutular, 0.093, alan=0.187))
    assert a['alan'] == 'makro/yakın çekim'
    assert a['boru_hattina_uygun'] is False


def test_saha_verisi_boru_hattina_girer():
    """organ_detection: dagilmis merkez, goruntulerin %24'u cok kutulu."""
    kutular = [1] * 76 + [4] * 24
    a = imgsz_oner.alan_tespiti(_olcum(kutular, 0.199, alan=0.327))
    assert a['alan'] == 'saha'
    assert a['boru_hattina_uygun'] is True


def test_buyuk_kutu_alani_tek_basina_makro_saymaz():
    """organ_detection kutulari karenin %32'si — ama saha verisidir.

    Ilk surumde alan esigi vardi ve bu dataseti yanlisikla makro sayiyordu.
    Alan ayirt edici DEGILDIR; bu test o esigin geri gelmesini engeller.
    """
    a = imgsz_oner.alan_tespiti(_olcum([1] * 76 + [4] * 24, 0.199, alan=0.90))
    assert a['boru_hattina_uygun'] is True


def test_etiketsiz_pakette_karar_verilmez():
    a = imgsz_oner.alan_tespiti({'kutu_sayisi': [], 'merkez_kacikligi': []})
    assert a == {}
