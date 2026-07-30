"""Betiklerin ürün kapsamlı yapılandırmayı bulabildiğini doğrular.

configs/strawberry_data.yaml → configs/urunler/cilek/veri.yaml taşınması
BEŞ betiği sessizce kırmıştı: --help bile FileNotFoundError veriyordu.
Bu test onu bir daha yaşamamak için.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parent.parent

# --help ile açılabilen betikler (dış bağımlılığı olmayanlar)
BETIKLER = [
    'train_yolo.py', 'dataset_ayir.py', 'etiket_temizle.py', 'merge_datasets.py',
    'model_karsilastir.py', 'augment_by_class.py', 'epoch_oner.py',
    'prepare_colab_dataset.py', 'evaluate_model.py', 'split_dataset.py',
    'add_background_images.py', 'collect_field_data.py', 'sahi_predict.py',
]


@pytest.mark.parametrize('betik', BETIKLER)
def test_betik_help_calisir(betik):
    """Varsayılan yapılandırma yolu argparse kurulurken çözülür; yol yanlışsa
    --help bile çöker."""
    r = subprocess.run([sys.executable, str(KOK / 'scripts' / betik), '--help'],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f'{betik}: {r.stderr[-400:]}'


def test_veri_yapilandirmasi_urun_klasorunu_bulur():
    sys.path.insert(0, str(KOK / 'scripts'))
    import dataset_ayir
    yol = dataset_ayir.veri_yapilandirmasi('cilek')
    assert yol.exists(), yol
    assert yol.parent.name == 'cilek'


def test_veri_yapilandirmasi_bilinmeyen_urunde_eski_yola_duser():
    sys.path.insert(0, str(KOK / 'scripts'))
    import dataset_ayir
    yol = dataset_ayir.veri_yapilandirmasi('olmayan_bitki')
    assert yol.name == 'strawberry_data.yaml'


def _calistir(*argv):
    """Alt süreci UTF-8 okur.

    Windows konsolu cp1254'tür ve Türkçe sınıf adlarındaki bazı karakterleri
    çözemez; text=True yerel kod sayfasını kullandığı için çıktı okunamıyordu
    (UnicodeDecodeError). Kodlamayı iki uçta da açıkça UTF-8 yapıyoruz.
    """
    ortam = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=120, env=ortam)


def test_sinif_ekle_listeler():
    r = _calistir(str(KOK / 'scripts' / 'sinif_ekle.py'), '--listele')
    assert r.returncode == 0, r.stderr[-400:]
    assert 'Gray Mold' in r.stdout


def test_sinif_ekle_urun_kutugunu_bulur():
    """GERÇEK HATA: yol taşınınca --listele sessizce BOŞ liste veriyordu.

    Çökmediği için fark edilmesi zordu: kullanıcı sınıf eklese okunmayan bir
    dosyaya yazılırdı.
    """
    sys.path.insert(0, str(KOK / 'scripts'))
    import importlib
    import sinif_ekle
    importlib.reload(sinif_ekle)
    assert sinif_ekle.KUTUK.exists(), sinif_ekle.KUTUK
    assert sinif_ekle.EGITIM.exists(), sinif_ekle.EGITIM
    assert sinif_ekle.KUTUK.parent.name == 'cilek'
