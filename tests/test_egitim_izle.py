"""Eğitim gözcüsü — Drive eşitleme kopyalarını birleştirme.

NEDEN TEST?
    Google Drive masaüstü uygulaması eşitleme çakışmasında yerel aynada
    "<ad> (1)" klasörü üretir. Colab tarafında TEK klasör vardır. Gerçek
    örnek:

        bocek_teshis (1)  → epoch   1-108   (results.csv 108 satır)
        bocek_teshis      → epoch 109-200   (results.csv  92 satır)

    Ayrı okununca ikisi de "yarım" göründü ve gözcü "eğitim ölmüş, yeniden
    çalıştırın" dedi. Oysa koşu 200/200 TAMAMLANMIŞTI. Yanlış teşhis
    kullanıcıyı gereksiz yere yeniden eğitim başlatmaya iterdi — saatler.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    'egitim_izle', KOK / 'scripts' / 'egitim_izle.py')
izle = importlib.util.module_from_spec(_spec)
sys.modules['egitim_izle'] = izle
_spec.loader.exec_module(izle)


def _kosu_yaz(kok: Path, ad: str, epochlar, hedef=200, zaman_basla=0.0):
    """Sahte koşu klasörü: results.csv + args.yaml + weights."""
    d = kok / ad
    (d / 'weights').mkdir(parents=True, exist_ok=True)
    (d / 'weights' / 'last.pt').write_bytes(b'x')
    (d / 'args.yaml').write_text(f'epochs: {hedef}\n', encoding='utf-8')
    with (d / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        y = csv.writer(f)
        y.writerow(['epoch', 'time', 'metrics/mAP50-95(B)'])
        for i, e in enumerate(epochlar):
            y.writerow([e, zaman_basla + i * 40, 0.30 + e * 0.001])
    return d


class TestTabanAd:
    @pytest.mark.parametrize('ad,beklenen', [
        ('bocek_teshis (1)', 'bocek_teshis'),
        ('bocek_teshis (12)', 'bocek_teshis'),
        ('bocek_teshis', 'bocek_teshis'),
        ('organ_detection-2', 'organ_detection-2'),      # Ultralytics eki, kopya DEĞİL
        ('strawberry_exp-4', 'strawberry_exp-4'),
    ])
    def test_kopya_soneki_ayirt_edilir(self, ad, beklenen):
        assert izle.taban_ad(ad) == beklenen

    def test_ultralytics_eki_kopya_sayilmaz(self):
        """'-2' Ultralytics'in yeni koşu ekidir; birleştirilirse AYRI
        eğitimler tek koşu sanılır ve epoch sayısı saçmalar."""
        assert izle.taban_ad('organ_detection-2') != izle.taban_ad('organ_detection')


class TestKopyaBirlestirme:
    def test_bolunmus_kosu_tamamlanmis_sayilir(self, tmp_path):
        """ASIL HATA: 108 + 92 satır, iki klasör → 'yarım' görünüyordu."""
        _kosu_yaz(tmp_path, 'bocek_teshis (1)', range(1, 109))
        _kosu_yaz(tmp_path, 'bocek_teshis', range(109, 201), zaman_basla=0)

        g = izle.gruplar(tmp_path)
        assert set(g) == {'bocek_teshis'}
        o = izle.olcum(g['bocek_teshis'][0], g['bocek_teshis'])
        assert o['epoch'] == 200
        assert o['durum'] == 'bitti'
        assert o['kopya'] == 2

    def test_ilerleme_satir_sayisindan_DEGIL_epoch_sutunundan(self, tmp_path):
        """Satır sayısı bölünmede yanıltır; epoch numarası gerçeği söyler."""
        _kosu_yaz(tmp_path, 'x (1)', range(1, 51))
        _kosu_yaz(tmp_path, 'x', range(151, 201))
        g = izle.gruplar(tmp_path)
        o = izle.olcum(g['x'][0], g['x'])
        assert o['epoch'] == 200, 'satır sayısı 100 ama epoch 200'

    def test_en_iyi_map_kopyalar_arasindan(self, tmp_path):
        _kosu_yaz(tmp_path, 'y (1)', range(1, 101))
        _kosu_yaz(tmp_path, 'y', range(101, 201))
        g = izle.gruplar(tmp_path)
        o = izle.olcum(g['y'][0], g['y'])
        assert o['en_iyi'] == pytest.approx(0.30 + 200 * 0.001)

    def test_kopyasiz_kosu_etkilenmez(self, tmp_path):
        _kosu_yaz(tmp_path, 'tek', range(1, 201))
        g = izle.gruplar(tmp_path)
        o = izle.olcum(g['tek'][0], g['tek'])
        assert o['epoch'] == 200 and o['kopya'] == 1

    def test_farkli_kosular_birlestirilmez(self, tmp_path):
        _kosu_yaz(tmp_path, 'a', range(1, 51))
        _kosu_yaz(tmp_path, 'b', range(1, 51))
        g = izle.gruplar(tmp_path)
        assert set(g) == {'a', 'b'}


class TestDurum:
    def test_yarim_kosu_DURDU(self, tmp_path):
        """Yarım koşu, uzun süredir yazmıyorsa ölmüştür."""
        import os
        import time
        d = _kosu_yaz(tmp_path, 'z', range(1, 51), hedef=200)
        eski = time.time() - 3600            # 1 saat önce yazılmış
        os.utime(d / 'results.csv', (eski, eski))

        g = izle.gruplar(tmp_path)
        o = izle.olcum(g['z'][0], g['z'])
        assert o['durum'] == 'DURDU'
        assert o['epoch'] == 50

    def test_yeni_yazan_yarim_kosu_CALISIYOR(self, tmp_path):
        """Az önce yazmışsa çalışıyordur — gözcü süreci göremez, zamana bakar."""
        _kosu_yaz(tmp_path, 'z2', range(1, 51), hedef=200)
        g = izle.gruplar(tmp_path)
        assert izle.olcum(g['z2'][0], g['z2'])['durum'] == 'ÇALIŞIYOR'

    def test_hedef_args_yamldan_okunur(self, tmp_path):
        _kosu_yaz(tmp_path, 'w', range(1, 61), hedef=60)
        g = izle.gruplar(tmp_path)
        assert izle.olcum(g['w'][0], g['w'])['durum'] == 'bitti'

    def test_bos_klasor_cokmez(self, tmp_path):
        (tmp_path / 'bos').mkdir()
        g = izle.gruplar(tmp_path)
        o = izle.olcum(g['bos'][0], g['bos'])
        assert o['epoch'] == 0


class TestIsabet:
    def test_gercek_drive_klasoru(self):
        """Depodaki gerçek Drive klasörü okunabiliyor mu?"""
        kok = izle.drive_bul(None) / 'results' if any(
            Path(a).is_dir() for a in izle.VARSAYILAN_DRIVE) else None
        if kok is None or not kok.is_dir():
            pytest.skip('Drive bağlı değil')
        g = izle.gruplar(kok)
        assert g, 'hiç koşu bulunamadı'
        for ad, kopyalar in g.items():
            o = izle.olcum(kopyalar[0], kopyalar)
            assert o['epoch'] >= 0
