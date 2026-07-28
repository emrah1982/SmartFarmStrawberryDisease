"""Notebook'taki koşu keşif mantığı (devam etme) testleri.

Bu fonksiyonlar notebook hücresinde yaşıyor; buradan çıkarılıp sınanır.
Yanlış çalışırlarsa iki pahalı hata olur:
  - yanlış koşudan devam edilir (sıfırdan eğitim ince ayarın üstüne yazar),
  - biten bir koşu "yarım" sanılıp gereksiz yere sürdürülür.
"""

import json
from pathlib import Path

import pytest
import yaml

KOK = Path(__file__).resolve().parent.parent
NOTEBOOK = KOK / 'StrawberryVision_Colab_Production.ipynb'


def _yardimcilari_yukle(proje_dizini):
    """Notebook'un eğitim hücresinden koşu yardımcılarını çıkarır."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    egitim = next(''.join(c['source']) for c in nb['cells']
                  if c['cell_type'] == 'code' and 'def _kosular' in ''.join(c['source']))
    bas = egitim.index('INCE_EK =')
    son = egitim.index('# Mevcut koşuları göster')
    ortam = {'TRAIN_CONFIG': {'project': str(proje_dizini),
                              'name': 'strawberry_exp', 'epochs': 200},
             'Path': Path}
    exec(egitim[bas:son], ortam)
    return ortam


def _kosu_olustur(kok, ad, hedef_epoch, tamamlanan, agirlik=True):
    d = kok / ad
    (d / 'weights').mkdir(parents=True)
    if agirlik:
        (d / 'weights' / 'last.pt').write_bytes(b'x')
    (d / 'args.yaml').write_text(yaml.dump({'epochs': hedef_epoch}), encoding='utf-8')
    satirlar = ['epoch'] + [str(i) for i in range(1, tamamlanan + 1)]
    (d / 'results.csv').write_text(chr(10).join(satirlar), encoding='utf-8')
    return d


@pytest.fixture
def ortam(tmp_path):
    _kosu_olustur(tmp_path, 'strawberry_exp', 200, 200)            # sıfırdan, bitti
    _kosu_olustur(tmp_path, 'strawberry_exp-4', 200, 120)          # sıfırdan, yarım
    _kosu_olustur(tmp_path, 'strawberry_exp_ince_ayar', 60, 60)    # ince ayar, bitti
    _kosu_olustur(tmp_path, 'strawberry_exp_ince_ayar-2', 60, 18)  # ince ayar, yarım
    return _yardimcilari_yukle(tmp_path), tmp_path


def test_ince_ayar_kosusu_ayirt_edilir(ortam):
    g, kok = ortam
    assert g['_ince_mi'](kok / 'strawberry_exp_ince_ayar') is True
    assert g['_ince_mi'](kok / 'strawberry_exp-4') is False


def test_hedef_epoch_kosunun_kendisinden_okunur(ortam):
    """60 epochluk ince ayar, 200 hedefiyle kıyaslanmamalı."""
    g, kok = ortam
    assert g['_hedef_epoch'](kok / 'strawberry_exp_ince_ayar') == 60
    assert g['_hedef_epoch'](kok / 'strawberry_exp') == 200


def test_biten_ince_ayar_yarim_sayilmaz(ortam):
    """En sinsi hata: 60/60 biten koşu, 200 hedefiyle 'yarım' görünürdü."""
    g, kok = ortam
    assert g['_durum'](kok / 'strawberry_exp_ince_ayar')[0] == 'bitti'


def test_sifirdan_devam_ince_ayari_secmez(ortam):
    g, _ = ortam
    assert g['_devam_edilebilir'](ince=False)[0].name == 'strawberry_exp-4'


def test_ince_ayar_devam_kendi_kosusunu_secer(ortam):
    g, _ = ortam
    assert g['_devam_edilebilir'](ince=True)[0].name == 'strawberry_exp_ince_ayar-2'


def test_en_ilerlemis_kosu_secilir(tmp_path):
    """'En yeni' değil 'en ilerlemiş': yanlışlıkla açılan 1 epochluk koşu
    120 epoch ilerlemiş koşunun önüne geçmemeli."""
    _kosu_olustur(tmp_path, 'strawberry_exp-1', 200, 120)
    _kosu_olustur(tmp_path, 'strawberry_exp-9', 200, 1)     # daha yeni ama boş
    g = _yardimcilari_yukle(tmp_path)
    assert g['_devam_edilebilir'](ince=False)[0].name == 'strawberry_exp-1'


def test_checkpointsiz_kosu_devam_adayi_degil(tmp_path):
    _kosu_olustur(tmp_path, 'strawberry_exp-3', 200, 50, agirlik=False)
    g = _yardimcilari_yukle(tmp_path)
    assert g['_durum'](tmp_path / 'strawberry_exp-3')[0] == 'yok'
    assert g['_devam_edilebilir'](ince=False)[0] is None


def test_hic_kosu_yoksa_none(tmp_path):
    g = _yardimcilari_yukle(tmp_path)
    assert g['_devam_edilebilir']()[0] is None
