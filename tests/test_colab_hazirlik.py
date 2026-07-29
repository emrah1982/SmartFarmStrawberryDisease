"""prepare_colab_dataset.py — uzman model dalı.

NEDEN TEST?
    Bu kod yalnızca Colab'de çalışır; hata ancak orada, eğitim başlamak
    üzereyken görülür. Arşiv kökünü yanlış seçmek sessiz bir hatadır:
    dataset açılır, `datasets/<urun>/<model>/data.yaml` bulunamaz ve
    Ultralytics "images not found" der. Burada yerelde sabitliyoruz.
"""

import importlib.util
import sys
import zipfile
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    'prepare_colab_dataset', KOK / 'scripts' / 'prepare_colab_dataset.py')
hazirlik = importlib.util.module_from_spec(_spec)
sys.modules['prepare_colab_dataset'] = hazirlik
_spec.loader.exec_module(hazirlik)


def _paket_yaz(hedef: Path, kok: str = '') -> Path:
    """Küçük ama gerçek yapıda bir uzman dataset zip'i üretir."""
    on = f'{kok}/' if kok else ''
    hedef.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(hedef, 'w') as z:
        z.writestr(f'{on}data.yaml',
                   'train: train/images\nval: valid/images\nnc: 3\n'
                   "names: {0: Leaf, 1: Fruit, 2: Flower}\n")
        for b in ('train', 'valid', 'test'):
            z.writestr(f'{on}{b}/images/a.jpg', b'jpeg')
            z.writestr(f'{on}{b}/labels/a.txt', '0 0.5 0.5 0.2 0.2\n')
    return hedef


class TestArsivKoku:
    def test_sarmalayicili(self):
        adlar = ['organ_detection/data.yaml', 'organ_detection/train/images/a.jpg']
        assert hazirlik.model_archive_root(adlar) == 'organ_detection'

    def test_sarmalayicisiz(self):
        assert hazirlik.model_archive_root(['data.yaml', 'train/images/a.jpg']) == ''

    def test_en_sig_data_yaml_secilir(self):
        """Derindeki kopya seçilirse train/ bulunamaz — sessiz hata olurdu."""
        adlar = ['pkt/data.yaml', 'pkt/train/images/a.jpg', 'pkt/eski/yedek/data.yaml']
        assert hazirlik.model_archive_root(adlar) == 'pkt'

    def test_data_yaml_yoksa_none(self):
        assert hazirlik.model_archive_root(['train/images/a.jpg']) is None


class TestUzmanHazirlik:
    def test_sarmalayicili_arsiv_acilir(self, tmp_path):
        depo = tmp_path / 'depo'
        depo.mkdir()
        drive = tmp_path / 'drive'
        _paket_yaz(drive / 'dataset' / 'organ_detection.zip', 'organ_detection')

        assert hazirlik.prepare(drive, depo, 'organ_detection', 'cilek') == 0
        hedef = depo / 'datasets' / 'cilek' / 'organ_detection'
        assert (hedef / 'data.yaml').exists()
        assert (hedef / 'train' / 'images' / 'a.jpg').exists()
        # Fazladan seviye oluşmamalı
        assert not (hedef / 'organ_detection').exists()

    def test_sarmalayicisiz_arsiv_de_calisir(self, tmp_path):
        depo = tmp_path / 'depo'
        depo.mkdir()
        drive = tmp_path / 'drive'
        _paket_yaz(drive / 'dataset' / 'leaf_disease.zip', '')

        assert hazirlik.prepare(drive, depo, 'leaf_disease', 'cilek') == 0
        assert (depo / 'datasets' / 'cilek' / 'leaf_disease' / 'data.yaml').exists()

    def test_hazirsa_yeniden_acmaz(self, tmp_path):
        depo = tmp_path / 'depo'
        hedef = depo / 'datasets' / 'cilek' / 'organ_detection'
        (hedef / 'train' / 'images').mkdir(parents=True)
        (hedef / 'data.yaml').write_text('nc: 3\n', encoding='utf-8')
        imza = (hedef / 'data.yaml').read_text(encoding='utf-8')

        # Drive'da arşiv olmasa bile başarılı olmalı
        assert hazirlik.prepare(depo / 'yok', depo, 'organ_detection', 'cilek') == 0
        assert (hedef / 'data.yaml').read_text(encoding='utf-8') == imza

    def test_arsiv_yoksa_hata_kodu(self, tmp_path):
        depo = tmp_path / 'depo'
        depo.mkdir()
        assert hazirlik.prepare(tmp_path / 'bos', depo, 'pest_detection', 'cilek') == 1

    def test_urun_klasoru_ayrilir(self, tmp_path):
        """Farklı ürünler birbirinin dataset'ini ezmemeli."""
        depo = tmp_path / 'depo'
        depo.mkdir()
        drive = tmp_path / 'drive'
        _paket_yaz(drive / 'dataset' / 'organ_detection.zip', 'organ_detection')

        assert hazirlik.prepare(drive, depo, 'organ_detection', 'domates') == 0
        assert (depo / 'datasets' / 'domates' / 'organ_detection' / 'data.yaml').exists()
        assert not (depo / 'datasets' / 'cilek').exists()


class TestArsivKonumu:
    """Paketler Drive'a depodaki `datasets/<urun>/` düzeniyle yüklenir.

    Notebook eskiden yalnızca `dataset/` klasörüne bakıyordu; kullanıcı
    klasörü olduğu gibi kopyaladığında "arşiv yok" diyordu. Aranan konumlar
    burada sabitlenmiştir.
    """

    def test_datasets_urun_altinda_bulunur(self, tmp_path):
        depo = tmp_path / 'depo'
        depo.mkdir()
        drive = tmp_path / 'drive'
        _paket_yaz(drive / 'datasets' / 'cilek' / 'organ_detection.zip', 'organ_detection')

        assert hazirlik.prepare(drive, depo, 'organ_detection', 'cilek') == 0
        assert (depo / 'datasets' / 'cilek' / 'organ_detection' / 'data.yaml').exists()

    def test_eski_dataset_klasoru_hala_calisir(self, tmp_path):
        depo = tmp_path / 'depo'
        depo.mkdir()
        drive = tmp_path / 'drive'
        _paket_yaz(drive / 'dataset' / 'leaf_disease.zip', 'leaf_disease')

        assert hazirlik.prepare(drive, depo, 'leaf_disease', 'cilek') == 0
        assert (depo / 'datasets' / 'cilek' / 'leaf_disease' / 'data.yaml').exists()

    def test_urun_klasoru_oncelikli(self, tmp_path):
        """İki yerde de varsa ürünün klasörü kazanmalı (yanlış ürünü eğitmeyelim)."""
        depo = tmp_path / 'depo'
        depo.mkdir()
        drive = tmp_path / 'drive'
        _paket_yaz(drive / 'datasets' / 'cilek' / 'organ_detection.zip', 'organ_detection')
        _paket_yaz(drive / 'dataset' / 'organ_detection.zip', 'organ_detection')

        dizinler = hazirlik.model_arsiv_dizinleri(drive, 'cilek')
        assert dizinler[0] == drive / 'datasets' / 'cilek'
        bulunan = hazirlik.find_archive(drive, tmp_path / 'yok',
                                        ('organ_detection.zip',), dizinler)
        assert bulunan == drive / 'datasets' / 'cilek' / 'organ_detection.zip'


class TestBirlesikBozulmadi:
    def test_varsayilan_eski_dala_gider(self, tmp_path):
        """model='birlesik' eski davranışı korumalı (geriye dönük uyum)."""
        depo = tmp_path / 'depo'
        depo.mkdir()
        # Arşiv yok → eski dal 1 döner ve datasets/ oluşturmaz
        assert hazirlik.prepare(tmp_path / 'bos', depo, 'birlesik', 'cilek') == 1
        assert not (depo / 'datasets').exists()
