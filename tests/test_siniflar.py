"""Sınıf kütüğü: eşik, açma-kapama, ID bütünlüğü ve yerelleştirilmiş çizim."""

import numpy as np
import pytest

from app import cizim, siniflar
from app.detector import Kutu


# ─────────────────────────────────────────────────── sınıf bazlı eşik
def test_kapali_sinif_hic_gosterilmez():
    """strawberry_unripe yaprakları çilek sanıyor — kapalıyken hiç çıkmamalı."""
    assert siniflar.aktif_mi('strawberry_unripe') is False
    assert siniflar.kabul_edilir_mi('strawberry_unripe', 0.99) is False


def test_hastalik_siniflari_dusuk_guvende_bile_gecer():
    """Erken evre hastalık düşük güvenle bulunur; genel eşik yükseltilmemeli."""
    assert siniflar.kabul_edilir_mi('Gray Mold', 0.30) is True
    assert siniflar.kabul_edilir_mi('Leaf Spot', 0.26) is True


def test_sinif_esigi_uygulanir():
    esik = siniflar.esik('strawberry_ripe')
    assert siniflar.kabul_edilir_mi('strawberry_ripe', esik - 0.01) is False
    assert siniflar.kabul_edilir_mi('strawberry_ripe', esik + 0.01) is True


def test_model_en_dusuk_esikle_calistirilir():
    """Yüksek eşikli bir sınıf yüzünden diğerleri kaybolmamalı."""
    assert siniflar.en_dusuk_esik() <= min(
        siniflar.esik(a) for a in ('Gray Mold', 'Leaf Spot'))


def test_bilinmeyen_sinif_varsayilana_duser():
    from app import config
    assert siniflar.esik('Olmayan Sinif') == config.CONF_THRESHOLD
    assert siniflar.aktif_mi('Olmayan Sinif') is True


def test_ortam_degiskeniyle_kapatma(monkeypatch):
    monkeypatch.setattr(siniflar, '_KAPALI_ENV', {'Gray Mold'})
    assert siniflar.aktif_mi('Gray Mold') is False


# ────────────────────────────────────────────────────── ID bütünlüğü
def test_id_haritasi_egitim_yamliyla_ayni():
    """ID kayması geçmiş etiketleri yanlış sınıfa çevirir — sabit kalmalı."""
    harita = siniflar.id_haritasi()
    assert harita[3] == 'Gray Mold'
    assert harita[9] == 'strawberry_unripe'
    assert len(harita) == len(set(harita.values())), 'aynı ad iki ID altında olamaz'


def test_yeni_id_bos_olani_verir():
    assert siniflar.yeni_id() == max(siniflar.id_haritasi()) + 1


def test_planlanan_siniflar_etiketlemede_cikmaz():
    """ID'si olmayan sınıf etiketleme listesine girmemeli (eğitimle uyumsuz olur)."""
    assert 'Spider Mites' not in siniflar.id_haritasi().values()
    assert siniflar.egitimde_mi('Spider Mites') is False
    assert siniflar.grup('Spider Mites') == 'zararli'


# ──────────────────────────────────────────── yerelleştirilmiş çizim
def _kare():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def test_cizim_turkce_karakterde_cokmez(tmp_path):
    """'Olgunlaşmamış Çilek' gibi etiketler görsele yazılabilmeli."""
    kutular = [Kutu(9, 'strawberry_unripe', 0.71, 0.5, 0.5, 0.3, 0.3)]
    cikti = tmp_path / 'c.jpg'
    assert cizim.sonuc_yaz(_kare(), kutular, str(cikti),
                           ad_cevir=lambda a: 'Olgunlaşmamış Çilek') is True
    assert cikti.exists() and cikti.stat().st_size > 0


def test_cizim_bos_kutu_listesiyle_calisir(tmp_path):
    cikti = tmp_path / 'bos.jpg'
    assert cizim.sonuc_yaz(_kare(), [], str(cikti)) is True


def test_ascii_yedegi_turkce_harfleri_bozmaz():
    """Yazı tipi bulunamazsa bile okunur çıktı üretilmeli."""
    assert cizim._asciye_indir('Olgunlaşmamış Çilek') == 'Olgunlasmamis Cilek'


def test_cizim_gorseli_degistirir(tmp_path):
    """Kutu çizilince görüntü gerçekten değişmeli (sessiz başarısızlık olmasın)."""
    kare = _kare()
    kutular = [Kutu(3, 'Gray Mold', 0.9, 0.5, 0.5, 0.4, 0.4)]
    yeni = cizim.kutulari_ciz(kare, kutular, lambda a: 'Kurşuni Küf')
    assert not np.array_equal(kare, yeni)
