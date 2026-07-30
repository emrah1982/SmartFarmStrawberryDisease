"""Eğitilen modelin boru hattına kurulması.

Elle kopyalamada üç sessiz hata olur: yanlış ada kopyalama (model hiç
kullanılmaz), yanlış modeli kopyalama, sınıfları uymayan model (boru hattı
çalışır ama sonuçlar saçmadır). Doğrulama kuralları burada sabitlenir.
"""

import subprocess
import sys
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / 'scripts'))


def test_kutuk_sinif_siralari_datasetlerle_ayni():
    """EN KRİTİK TUTARLILIK: eğitim ID'lerini dataset belirler.

    modeller.yaml'daki sıra dataset'ten farklıysa eğitilen model kütükle
    uyuşmaz ve kurulum reddedilir. Daha kötüsü --zorla ile kurulursa
    sınıflar kayar: 'Gray Mold' tespiti 'Powdery Mildew Leaf' görünür.
    """
    from app import modeller

    esleme = {'organ': 'organ_detection', 'yaprak_hastalik': 'leaf_disease',
              'meyve_hastalik': 'fruit_disease', 'olgunluk': 'fruit_ripeness'}
    for model_ad, ds_ad in esleme.items():
        y = KOK / 'datasets' / 'cilek' / ds_ad / 'data.yaml'
        if not y.exists():
            continue                      # dataset üretilmemişse atla
        v = yaml.safe_load(y.read_text(encoding='utf-8'))
        ds = [v['names'][i] for i in sorted(v['names'])]
        kutuk = modeller.tanim(model_ad).siniflar
        assert [s.lower() for s in kutuk] == [s.lower() for s in ds], (
            f'{model_ad}: kütük {kutuk} != dataset {ds}')


def test_listele_calisir():
    r = subprocess.run([sys.executable, str(KOK / 'scripts' / 'model_kur.py'), '--listele'],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    assert 'organ' in r.stdout and 'miras' in r.stdout


def test_bilinmeyen_model_adi_reddedilir():
    import model_kur
    assert model_kur.kur('olmayan_model', KOK / 'models' / 'cilek' / 'best.pt') == 1


def test_olmayan_dosya_reddedilir():
    import model_kur
    assert model_kur.kur('organ', KOK / 'olmayan_dosya.pt') == 1


def test_her_kutuk_modeli_dataset_ile_eslesir():
    """Kütükteki her uzman modelin karşılığı bir dataset olmalı (zararlı hariç:
    verisi henüz toplanmadı)."""
    from app import modeller

    beklenen = {'organ': 'organ_detection', 'yaprak_hastalik': 'leaf_disease',
                'meyve_hastalik': 'fruit_disease', 'olgunluk': 'fruit_ripeness',
                'zararli': 'pest_detection', 'bocek_teshis': 'bocek_teshis'}
    for ad, t in modeller.tanimlar().items():
        if t.rol == 'miras':
            continue
        assert ad in beklenen, f'{ad} için dataset eşlemesi tanımsız'
