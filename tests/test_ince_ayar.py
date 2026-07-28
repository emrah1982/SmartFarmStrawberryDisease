"""İnce ayar (fine-tuning) güvenlik kontrolleri.

Sınıf uyumsuzluğu bu akıştaki en pahalı hatadır: Ultralytics hata vermez,
tespit başını sessizce yeniden kurar veya ID kaydığı için yanlış sınıfları
öğrenir. Sonuç saatlerce GPU harcandıktan sonra fark edilir. Kontrolün
davranışı bu yüzden testle sabitlenir.
"""

import sys
from pathlib import Path

import pytest
import yaml

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / 'scripts'))
import train_yolo as ty


SINIFLAR = ['Angular Leafspot', 'Anthracnose Fruit Rot', 'Blossom Blight', 'Gray Mold',
            'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf',
            'strawberry_ripe', 'strawberry_semi_ripe', 'strawberry_unripe']


@pytest.fixture
def data_yaml(tmp_path):
    p = tmp_path / 'data.yaml'
    p.write_text(yaml.dump({'nc': len(SINIFLAR), 'names': dict(enumerate(SINIFLAR)),
                            'train': 'x', 'val': 'y'}), encoding='utf-8')
    return str(p)


def _agirlik(monkeypatch, siniflar):
    """torch gerekmeden kontrol noktası sınıflarını taklit eder."""
    monkeypatch.setattr(ty, 'agirlik_siniflari', lambda yol: siniflar)


# ────────────────────────────────────────────────── uyumlu / uyumsuz
def test_ayni_siniflar_gecer(monkeypatch, data_yaml):
    _agirlik(monkeypatch, list(SINIFLAR))
    assert ty.sinif_uyumu_kontrol('models/best.pt', data_yaml) is True


def test_sinif_sayisi_farkliysa_egitim_durur(monkeypatch, data_yaml):
    """Yeni sınıf eklendiyse ince ayar yapılamaz — tespit başı yeniden kurulmalı."""
    _agirlik(monkeypatch, SINIFLAR + ['Spider Mites'])
    assert ty.sinif_uyumu_kontrol('models/best.pt', data_yaml) is False


def test_sinif_sirasi_kaydiysa_egitim_durur(monkeypatch, data_yaml):
    """En sinsi hata: sayı aynı ama sıra farklı → etiketler yanlış sınıfa gider."""
    kaymis = list(SINIFLAR)
    kaymis[3], kaymis[4] = kaymis[4], kaymis[3]
    _agirlik(monkeypatch, kaymis)
    assert ty.sinif_uyumu_kontrol('models/best.pt', data_yaml) is False


def test_hazir_model_kontrolden_muaf(monkeypatch, data_yaml):
    """yolo26s.pt gibi indirilecek modelde karşılaştırılacak sınıf yoktur."""
    _agirlik(monkeypatch, None)
    assert ty.sinif_uyumu_kontrol('yolo26s.pt', data_yaml) is True


def test_var_olmayan_dosya_hazir_model_sayilir():
    """Ağırlık okunamıyorsa kontrol eğitim yolunu tıkamamalı."""
    assert ty.agirlik_siniflari('yolo26s.pt') is None
    assert ty.agirlik_siniflari('olmayan_model.pt') is None


def test_uyumsuzlukta_sebep_yazilir(monkeypatch, data_yaml, caplog):
    """Kullanıcı neden durduğunu ve ne yapacağını görmeli."""
    import logging
    _agirlik(monkeypatch, SINIFLAR + ['Spider Mites'])
    with caplog.at_level(logging.ERROR):
        ty.sinif_uyumu_kontrol('models/best.pt', data_yaml)
    metin = caplog.text
    assert 'EGITIM BASLATILMADI' in metin
    assert 'Spider Mites' in metin, 'hangi sınıfın fazla olduğu yazılmalı'
    assert 'NE YAPMALI' in metin, 'çözüm yolu gösterilmeli'
    assert 'sifirdan' in metin


def test_bozuk_data_yaml_egitimi_durdurur(monkeypatch, tmp_path):
    _agirlik(monkeypatch, list(SINIFLAR))
    assert ty.sinif_uyumu_kontrol('models/best.pt', str(tmp_path / 'yok.yaml')) is False


# ──────────────────────────────────────────── ince ayar yapılandırması
@pytest.fixture(scope='module')
def ince():
    return yaml.safe_load((KOK / 'configs' / 'finetune_config.yaml').read_text(encoding='utf-8'))


def test_optimizer_auto_degil(ince):
    """optimizer: auto lr0'ı YOK SAYAR ve sıfırdan eğitim için yüksek bir değer
    seçer; ince ayarda bu öğrenilmiş ağırlıkları bozar."""
    assert ince['optimizer'] != 'auto'


def test_ogrenme_orani_sifirdan_egitimden_dusuk(ince):
    sifirdan = yaml.safe_load((KOK / 'configs' / 'train_config.yaml').read_text(encoding='utf-8'))
    assert ince['lr0'] < sifirdan['lr0'], 'ince ayarda lr0 daha küçük olmalı'


def test_epoch_sayisi_makul(ince):
    """Çok kısa: etiket geometrisi düzeltmeleri öğrenilmez. Çok uzun: warm
    start'ın anlamı kalmaz."""
    assert 40 <= ince['epochs'] <= 100


def test_imgsz_mevcut_modelle_ayni(ince):
    """Farklı imgsz warm start'ın kazancını büyük ölçüde siler."""
    sifirdan = yaml.safe_load((KOK / 'configs' / 'train_config.yaml').read_text(encoding='utf-8'))
    assert ince['imgsz'] == sifirdan['imgsz']


def test_ayri_kosu_adi(ince):
    """İnce ayar sıfırdan eğitimin klasörünü ezmemeli."""
    sifirdan = yaml.safe_load((KOK / 'configs' / 'train_config.yaml').read_text(encoding='utf-8'))
    assert ince['name'] != sifirdan.get('name')
