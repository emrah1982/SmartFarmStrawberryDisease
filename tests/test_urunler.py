"""Ürün (bitki türü) kapsamı — çok bitkili kurulumun temeli.

İkinci bitki eklendiğinde en sinsi hata SINIF ID ÇAKIŞMASIDIR: domatesin
"Leaf Spot"u çileğinkiyle aynı ID'yi alırsa model iki farklı hastalığı tek
sınıf sanır. Kapsam kuralları bu yüzden testle sabitlenir.
"""

import yaml

from app import modeller, siniflar, urunler


# ─────────────────────────────────────────────────────── slug
def test_turkce_ad_slug_olur():
    """Sera kaydında 'Çilek' yazar; klasör/ortam anahtarı ASCII olmalı."""
    assert urunler.slug('Çilek') == 'cilek'
    assert urunler.slug('Domates') == 'domates'
    assert urunler.slug('Kırmızı Biber') == 'kirmizi_biber'


def test_bos_ad_varsayilana_duser():
    assert urunler.slug('') == urunler.VARSAYILAN
    assert urunler.slug(None) == urunler.VARSAYILAN


# ────────────────────────────────────────────── yapılandırma yolu
def test_urun_yapilandirmasi_kendi_klasorunden_okunur():
    yol = urunler.yapilandirma('cilek', 'siniflar.yaml')
    assert yol.exists()
    assert yol.parent.name == 'cilek'
    assert yol.parent.parent.name == 'urunler'


def test_tanimsiz_urun_eski_yola_duser(tmp_path):
    """Geriye dönük uyumluluk: ürün klasörü yoksa kapsamsız yol kullanılır."""
    yol = urunler.yapilandirma('olmayan_bitki', 'train_config.yaml')
    assert yol.name == 'train_config.yaml'
    assert yol.parent.name == 'configs'


def test_model_dizini_urune_gore():
    d = urunler.model_dizini('cilek')
    assert d.name == 'cilek'


# ──────────────────────────────────── sınıf kütüğü ürün başına
def test_urun_kutugu_bagimsiz(tmp_path, monkeypatch):
    """İki ürünün sınıf listesi birbirinden BAĞIMSIZ olmalı."""
    sahte = tmp_path / 'urunler' / 'domates'
    sahte.mkdir(parents=True)
    (sahte / 'siniflar.yaml').write_text(yaml.dump({
        'Tomato Leaf Spot': {'tr': 'Domates Yaprak Lekesi', 'esik': 0.5},
    }, allow_unicode=True), encoding='utf-8')

    monkeypatch.setattr(urunler, 'URUN_KOK', tmp_path / 'urunler')
    siniflar.bosalt_onbellek()

    assert siniflar.esik('Tomato Leaf Spot', 'domates') == 0.5
    # Çilek sınıfı domates kütüğünde YOK → varsayılan eşiğe düşer
    from app import config
    assert siniflar.esik('Gray Mold', 'domates') == config.CONF_THRESHOLD
    siniflar.bosalt_onbellek()


def test_cilek_kutugu_beklenen_siniflari_icerir():
    k = siniflar.kutuk('cilek')
    assert 'Gray Mold' in k and 'strawberry_unripe' in k


# ─────────────────────────────────────── model kütüğü ürün başına
def test_urune_ozgu_model_yollari_urun_klasorunde():
    """Ürüne özgü modeller models/<urun>/ altında olmalı.

    İSTİSNA: `ortak: true` olanlar. Böcek türü ürüne bağlı değildir
    (danaburnu her bitkide aynı türdür), o yüzden ağırlığı models/
    kökünde durur ve her ürün aynı dosyayı kullanır.
    Bkz. app/urunler.py "ORTAK KAPSAM".
    """
    ortaklar = {t.ad for t in modeller.tanimlar('cilek').values() if t.ortak}
    assert ortaklar, 'ortak model kalmadıysa bu testin istisnası da gereksiz'
    for d in modeller.durum():
        if d['ad'] in ortaklar:
            continue
        assert f'models{chr(92)}cilek' in d['yol'] or 'models/cilek' in d['yol'], \
            d['yol']


def test_ortak_model_urun_klasorunde_ARANMAZ():
    """Ortak model ürün klasörüne kopyalanmamalı — tek dosya, n ürün."""
    for t in modeller.tanimlar('cilek').values():
        if not t.ortak:
            continue
        yol = str(t.yol)
        assert f'models{chr(92)}cilek' not in yol and 'models/cilek' not in yol, yol


def test_miras_modeli_urun_klasorunde_aranir():
    """Miras model ÜRÜNÜN klasöründe aranmalı — dosyanın kendisi zorunlu değil.

    Eskiden dosyanın varlığı test ediliyordu. Uzman modeller eğitildikçe
    miras modelin işlevi bitiyor ve models/cilek/best.pt silinebiliyor;
    bu bir arıza değil, mimarinin hedefi. Test edilmesi gereken şey yolun
    ürün kapsamında çözülmesi.
    """
    t = modeller.tanim('miras')
    assert t is not None, 'miras kütükte tanımlı olmalı'
    yol = str(t.yol)
    assert f'models{chr(92)}cilek' in yol or 'models/cilek' in yol, yol
    assert yol.endswith('best.pt')


def test_bir_model_yoksa_boru_hatti_cokmez():
    """Kütükte yazılı ama dosyası olmayan model None dönmeli, hata atmamalı."""
    assert modeller.yukle('bocek_teshis', 'cilek') is None or True  # yalnızca çökmemeli
    for t in modeller.tanimlar('cilek').values():
        if not t.var:
            assert modeller.yukle(t.ad, 'cilek') is None, (
                f'{t.ad}: dosya yokken None dönmeli')


def test_urun_listesi_cilegi_gorur():
    adlar = [u['ad'] for u in urunler.liste()]
    assert 'cilek' in adlar


# ──────────────────────────────────────────── kayıt kapsamı
def test_analiz_kaydinda_urun_sutunu_var():
    from app.database import Analiz
    assert hasattr(Analiz, 'urun'), 'kayıt hangi bitkiye ait olduğunu bilmeli'


def test_seradan_urun_cikarilir():
    class SahteSera:
        urun = 'Domates'
    assert urunler.seradan(SahteSera()) == 'domates'
    assert urunler.seradan(None) == urunler.VARSAYILAN
