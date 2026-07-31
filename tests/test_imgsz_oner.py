"""imgsz önerisi — dataset ölçülerek eğitim çözünürlüğü seçimi.

NEDEN TEST?
    imgsz her dataset için 1024 yazılıydı. Ölçtük: böcek dataset'i 416x416,
    yaprak/meyve 280 px kaynak çözünürlükte. Hepsi 1024'e BÜYÜTÜLEREK
    eğitiliyordu — bilgi eklemeden 10 kat hesap ve 10 kat RAM.

    Ters yön daha sinsi: büyük saha fotoğrafında küçük lezyon varsa ve
    imgsz düşürülürse lezyon birkaç piksele iner. Eğitim sorunsuz görünür,
    mAP düşük çıkar, sebebi anlaşılmaz.

    Öneri mantığı bu iki hatayı da önlemeli; ikisi de sessizdir.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    'imgsz_oner', KOK / 'scripts' / 'imgsz_oner.py')
io_ = importlib.util.module_from_spec(_spec)
sys.modules['imgsz_oner'] = io_
_spec.loader.exec_module(io_)


def _olcum(uzun_kenar, kutu_payi, goruntu=1000):
    """Elle kurulmuş ölçüm sözlüğü."""
    paylar = sorted(kutu_payi)
    return {
        'goruntu_sayisi': goruntu,
        'orneklenen': len(uzun_kenar),
        'okunamayan': 0,
        'uzun_kenar': sorted(uzun_kenar),
        'kutu_payi': paylar,
        'kutu_px': sorted(p * uzun_kenar[0] for p in paylar),
    }


class TestBuyutmeyiOnler:
    def test_kucuk_kaynak_icin_buyuk_imgsz_onerilmez(self):
        """416x416 makro dataset 1024'e büyütülmemeli."""
        o = _olcum([416] * 50, [0.30] * 50)
        s = io_.oner(o)
        assert s['imgsz'] <= 416, s

    def test_oneri_kaynak_cozunurlugu_asmaz(self):
        o = _olcum([640] * 50, [0.05] * 50)
        s = io_.oner(o)
        assert s['imgsz'] <= 640

    def test_buyutme_tabloda_isaretlenir(self):
        o = _olcum([416] * 20, [0.3] * 20)
        s = io_.oner(o)
        for satir in s['tablo']:
            assert satir['buyutme'] == (satir['imgsz'] > 416)


class TestKucukNesneyiKorur:
    def test_kucuk_nesne_varsa_imgsz_dusurulmez(self):
        """Nesne görüntünün %2'siyse 320'de 6 piksel kalır — öğrenilemez."""
        o = _olcum([2048] * 50, [0.02] * 50)
        s = io_.oner(o)
        assert s['kucuk_nesne_px'] >= io_.EN_KUCUK_NESNE_PX, s

    def test_buyuk_nesne_varsa_kucuk_imgsz_secilir(self):
        """Nesne kareyi dolduruyorsa yüksek çözünürlük israftır."""
        o = _olcum([2048] * 50, [0.50] * 50)
        s = io_.oner(o)
        assert s['imgsz'] <= 512, s

    def test_en_kucuk_yeterli_boy_secilir(self):
        """Hız için: yeterli olan EN KÜÇÜK aday seçilmeli."""
        o = _olcum([4000] * 50, [0.05] * 50)
        s = io_.oner(o)
        adaylar = [x for x in s['tablo'] if x['yeterli'] and not x['buyutme']]
        assert s['imgsz'] == adaylar[0]['imgsz']


class TestVeriSorunuAyirt:
    def test_kaynakta_zaten_kucukse_veri_sorunu_denir(self):
        """Kaynakta 3 piksellik kutu — imgsz ile çözülemez."""
        o = _olcum([280] * 50, [0.01] * 50)
        s = io_.oner(o)
        assert s['veri_sorunu'] is True
        assert s['kaynak_nesne_px'] < io_.EN_KUCUK_NESNE_PX

    def test_kaynakta_yeterliyse_veri_sorunu_yok(self):
        o = _olcum([2048] * 50, [0.05] * 50)
        assert io_.oner(o)['veri_sorunu'] is False


class TestGuvenliAlternatif:
    def test_guvenli_kaynak_cozunurluktur(self):
        o = _olcum([640] * 50, [0.20] * 50)
        s = io_.oner(o)
        assert s['guvenli'] == 640
        assert s['imgsz'] <= s['guvenli']

    def test_guvenli_1024_ile_sinirli(self):
        """4000 px kaynakta 4000'de eğitmek pratik değil."""
        o = _olcum([4000] * 50, [0.20] * 50)
        assert io_.oner(o)['guvenli'] == 1024


class TestRam:
    def test_ram_imgsz_karesiyle_artar(self):
        assert io_.ram_gb(1000, 640) == pytest.approx(io_.ram_gb(1000, 320) * 4)

    def test_gercek_olcek(self):
        """16.358 görüntü @1024 ≈ 51 GB — ölçülen değerle uyuşmalı."""
        assert 50 < io_.ram_gb(16358, 1024) < 53


class TestDayaniklilik:
    def test_bos_olcum_cokmez(self):
        assert io_.oner({}) == {}

    def test_kutusuz_dataset_cokmez(self):
        o = _olcum([640] * 10, [])
        s = io_.oner(o)
        assert s['nesne_yok'] is True
        assert s['imgsz'] > 0

    def test_oneri_32nin_kati(self):
        for uzun in (280, 416, 640, 1000, 4000):
            for pay in (0.01, 0.05, 0.2, 0.6):
                s = io_.oner(_olcum([uzun] * 20, [pay] * 20))
                assert s['imgsz'] % 32 == 0, (uzun, pay, s['imgsz'])


class TestGercekDatasetler:
    """Depodaki dataset'ler gerçekten ölçülebiliyor mu?"""

    @pytest.mark.parametrize('ad', ['organ_detection', 'leaf_disease',
                                    'fruit_disease', 'fruit_ripeness'])
    def test_olculebiliyor(self, ad):
        d = KOK / 'datasets' / 'cilek' / ad
        if not d.is_dir():
            pytest.skip(f'{ad} paketi yok')
        o = io_.olc(d, ornek=30)
        assert o and o['uzun_kenar'], f'{ad} ölçülemedi'
        s = io_.oner(o)
        assert s['imgsz'] in io_.ADAYLAR

    def test_hicbiri_1024e_ihtiyac_duymuyor(self):
        """Ölçüm: hepsinin kaynağı 640 px ve altı — 1024 hepsinde büyütme."""
        for ad in ('organ_detection', 'leaf_disease', 'fruit_disease'):
            d = KOK / 'datasets' / 'cilek' / ad
            if not d.is_dir():
                pytest.skip(f'{ad} paketi yok')
            s = io_.oner(io_.olc(d, ornek=30))
            assert s['imgsz'] < 1024, f'{ad}: {s["imgsz"]}'
