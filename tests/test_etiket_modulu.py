"""Aday etiket inceleme modülü.

Otomatik ön-etiketleme ADAY kutular üretir; bu modül onları insan
onayından geçirir. Testler iki şeyi korur:
  1. GÜVENLİK — kare adı URL'den gelir, paket dışına yazmamalı
  2. SESSİZ VERİ KAYBI OLMAMALI — geçersiz kutu atılırsa söylenmeli
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.moduller.etiket import servis                      # noqa: E402


@pytest.fixture
def paket(tmp_path):
    """images/ + labels_aday/ + data.yaml olan küçük bir paket."""
    kok = tmp_path / 'findik' / 'deneme'
    (kok / 'images').mkdir(parents=True)
    (kok / 'labels_aday').mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        ad = f'{i:03d}_Diseased_{i}.jpg'
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            kok / 'images' / ad)
        (kok / 'labels_aday' / f'{i:03d}_Diseased_{i}.txt').write_text(
            '0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n', encoding='utf-8')
    (kok / 'data.yaml').write_text(
        'train: images\nnc: 3\nnames:\n  0: Leaf\n  1: Nut\n  2: Husk\n',
        encoding='utf-8')
    return servis.paket_bul(tmp_path, 'findik', 'deneme')


# ─────────────────────────────────────────────── keşif

def test_paket_bulunur_ve_siniflar_okunur(paket):
    assert paket is not None
    assert paket.siniflar == ['Leaf', 'Nut', 'Husk']
    assert paket.goruntu_sayisi == 4
    assert paket.kutu_sayisi == 8
    assert paket.aday_klasor == 'labels_aday'


def test_images_yoksa_paket_sayilmaz(tmp_path):
    (tmp_path / 'x' / 'y' / 'labels_aday').mkdir(parents=True)
    assert servis.paketleri_bul(tmp_path) == []


def test_ilerleme_yuzdesi(paket):
    assert paket.ilerleme == 0
    servis.onay_ver(paket, paket_kare(paket, 0), True)
    yeni = servis.paket_bul(paket.yol.parent.parent, 'findik', 'deneme')
    assert yeni.onayli_sayisi == 1
    assert yeni.ilerleme == 25


def paket_kare(paket, i):
    return servis.kareler(paket, 'ad')[i].ad


# ─────────────────────────────────────────────── GÜVENLİK

@pytest.mark.parametrize('kotu', [
    '../../../etc/passwd', '..\\..\\main.py', 'a/b.jpg', 'a\\b.jpg',
    '..', '.', '', 'x\x00.jpg',
])
def test_yol_enjeksiyonu_reddedilir(paket, kotu):
    """Kare adı URL'den gelir; paket dışına çıkmamalı."""
    assert servis.guvenli_mi(kotu) is False
    assert servis.kare_yolu(paket, kotu) is None


def test_normal_ad_kabul_edilir(paket):
    ad = paket_kare(paket, 0)
    assert servis.guvenli_mi(ad)
    assert servis.kare_yolu(paket, ad) is not None


def test_olmayan_kare_none_doner(paket):
    assert servis.kare_yolu(paket, 'yok.jpg') is None


def test_paket_adi_da_denetlenir(tmp_path):
    assert servis.paket_bul(tmp_path, 'findik', '../../gizli') is None


# ─────────────────────────────────────────────── okuma / yazma

def test_kutular_okunur(paket):
    k = servis.kare_kutulari(paket, paket_kare(paket, 0))
    assert len(k) == 2
    assert k[0].sinif == 0
    assert k[0].cx == pytest.approx(0.5)


def test_poligon_etiketi_de_cozulur(paket):
    """Segmentasyon satırı kutu sanılırsa 2. noktanın koordinatları
    genişlik/yükseklik okunur — ölçülen gerçek hata (docs 2.6b)."""
    ad = paket_kare(paket, 0)
    (paket.yol / 'labels_aday' / (Path(ad).stem + '.txt')).write_text(
        '1 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n', encoding='utf-8')
    k = servis.kare_kutulari(paket, ad)
    assert len(k) == 1
    assert k[0].cx == pytest.approx(0.4)
    assert k[0].w == pytest.approx(0.4)
    assert k[0].h == pytest.approx(0.6)


def test_kaydetme_kutulari_degistirir(paket):
    ad = paket_kare(paket, 0)
    ok, mesaj = servis.kare_kaydet(paket, ad, [servis.Kutu(2, 0.4, 0.4, 0.3, 0.3)])
    assert ok
    k = servis.kare_kutulari(paket, ad)
    assert len(k) == 1 and k[0].sinif == 2


def test_gecersiz_kutu_SESSIZCE_atilmaz(paket):
    """Kullanıcı çizdiği kutunun kaybolduğunu fark etmeli."""
    ad = paket_kare(paket, 0)
    ok, mesaj = servis.kare_kaydet(paket, ad, [
        servis.Kutu(0, 0.5, 0.5, 0.2, 0.2),      # geçerli
        servis.Kutu(9, 0.5, 0.5, 0.1, 0.1),      # sınıf yok
        servis.Kutu(0, 1.5, 0.5, 0.1, 0.1),      # kadraj dışı
        servis.Kutu(0, 0.5, 0.5, 0.0, 0.1),      # sıfır genişlik
    ])
    assert ok
    assert 'geçersiz' in mesaj
    assert '3' in mesaj
    assert len(servis.kare_kutulari(paket, ad)) == 1


def test_bos_kutu_listesi_gecerli_negatif_ornektir(paket):
    """Kutusuz kare hatalı değil — modele 'burada bir şey yok' der."""
    ad = paket_kare(paket, 0)
    ok, _ = servis.kare_kaydet(paket, ad, [])
    assert ok
    assert servis.kare_kutulari(paket, ad) == []
    assert (paket.yol / 'labels_aday' / (Path(ad).stem + '.txt')).exists()


def test_yazim_atomiktir_gecici_dosya_kalmaz(paket):
    ad = paket_kare(paket, 0)
    servis.kare_kaydet(paket, ad, [servis.Kutu(0, 0.5, 0.5, 0.2, 0.2)])
    assert not list((paket.yol / 'labels_aday').glob('*.tmp'))


# ─────────────────────────────────────────────── onay ve sıra

def test_onay_kaydedilir_ve_geri_alinir(paket):
    ad = paket_kare(paket, 0)
    assert servis.onay_ver(paket, ad, True)
    assert servis.durum_oku(paket.yol)['onayli'].get(ad) is True
    assert servis.onay_ver(paket, ad, False)
    assert ad not in servis.durum_oku(paket.yol)['onayli']


def test_bozuk_durum_dosyasi_akisi_KESMEZ(paket):
    (paket.yol / servis.DURUM_DOSYASI).write_text('{bozuk', encoding='utf-8')
    d = servis.durum_oku(paket.yol)
    assert d == {'onayli': {}}


def test_sonraki_kare_onaylanmisi_atlar(paket):
    liste = [k.ad for k in servis.kareler(paket, 'ad')]
    servis.onay_ver(paket, liste[1], True)
    assert servis.sonraki_kare(paket, liste[0], 'ad') == liste[2]


def test_hepsi_onaylandiysa_sonraki_yok(paket):
    for k in servis.kareler(paket, 'ad'):
        servis.onay_ver(paket, k.ad, True)
    assert servis.sonraki_kare(paket, paket_kare(paket, 0), 'ad') is None


def test_guven_siralamasi_en_dusugu_one_alir(paket):
    (paket.yol / 'INCELEME.csv').write_text(
        'goruntu,organ,guven,saglik,kutu\n'
        f'{paket_kare(paket, 2)},Leaf,0.11,,\n'
        f'{paket_kare(paket, 0)},Leaf,0.90,,\n', encoding='utf-8')
    kareler = servis.kareler(paket, 'guven')
    assert kareler[0].en_dusuk_guven == pytest.approx(0.11)


# ─────────────────────────────────────────────── dışa aktarma

def test_disa_aktarma_yalniz_onaylilari_yazar(paket):
    liste = [k.ad for k in servis.kareler(paket, 'ad')]
    servis.onay_ver(paket, liste[0], True)
    servis.onay_ver(paket, liste[1], True)
    s = servis.disa_aktar(paket, yalniz_onayli=True)
    assert s['yazilan'] == 2 and s['atlanan'] == 2
    assert len(list((paket.yol / 'labels').glob('*.txt'))) == 2


def test_disa_aktarma_ADAY_klasorunu_SILMEZ(paket):
    """'Ham otomatik ne demişti' bilgisi kalmalı."""
    servis.onay_ver(paket, paket_kare(paket, 0), True)
    servis.disa_aktar(paket)
    assert (paket.yol / 'labels_aday').is_dir()
    assert list((paket.yol / 'labels_aday').glob('*.txt'))


# ─────────────────────────────────────────────── kalite denetimi

def test_sabit_kutu_uyarisi(paket):
    """4310 kutunun hepsi ayni degerdeydi — olculen gercek hata."""
    for k in servis.kareler(paket, 'ad'):
        servis.kare_kaydet(paket, k.ad, [
            servis.Kutu(0, 0.499, 0.499, 0.8, 0.8)] * 5)
    p = servis.paket_bul(paket.yol.parent.parent, 'findik', 'deneme')
    uyarilar = servis.kalite_denetimi(p)
    assert any(u['tur'] == 'sabit_kutu' for u in uyarilar)


def test_tam_kadraj_uyarisi(paket):
    for k in servis.kareler(paket, 'ad'):
        servis.kare_kaydet(paket, k.ad, [servis.Kutu(0, 0.5, 0.5, 1.0, 1.0)])
    p = servis.paket_bul(paket.yol.parent.parent, 'findik', 'deneme')
    assert any(u['tur'] == 'tam_kadraj' for u in servis.kalite_denetimi(p))


def test_saglikli_pakette_uyari_yok(paket):
    import random
    r = random.Random(0)
    for k in servis.kareler(paket, 'ad'):
        servis.kare_kaydet(paket, k.ad, [
            servis.Kutu(r.randrange(3), r.uniform(.3, .7), r.uniform(.3, .7),
                        r.uniform(.1, .3), r.uniform(.1, .3))
            for _ in range(4)])
    p = servis.paket_bul(paket.yol.parent.parent, 'findik', 'deneme')
    assert servis.kalite_denetimi(p) == []


# ─────────────────────────────────────────────── katman ayrımı

def test_servis_katmani_web_bagimliligi_TASIMAZ():
    """servis.py fastapi/sqlalchemy/jinja2/app.main import etmemeli.

    Katman kuralı: saf mantık ayrı test edilebilir ve başka bir arayüze
    (CLI, betik) takılabilir kalmalı.
    """
    import ast
    kok = Path(__file__).resolve().parents[1]
    kaynak = (kok / 'app' / 'moduller' / 'etiket' / 'servis.py').read_text(
        encoding='utf-8')
    YASAK = {'fastapi', 'sqlalchemy', 'jinja2', 'starlette'}
    for dugum in ast.walk(ast.parse(kaynak)):
        if isinstance(dugum, ast.Import):
            adlar = [a.name.split('.')[0] for a in dugum.names]
        elif isinstance(dugum, ast.ImportFrom):
            adlar = [(dugum.module or '').split('.')[0]]
        else:
            continue
        for a in adlar:
            assert a not in YASAK, f'servis.py {a} import ediyor'
        assert 'app.main' not in (getattr(dugum, 'module', '') or '')
