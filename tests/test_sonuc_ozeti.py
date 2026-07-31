"""Sonucun ORGANA göre gruplanması.

NEDEN TEST?
    Gerçek sahnede yaprak, çiçek ve meyve bir aradadır. Eski sonuç ekranı
    tespitleri yalnızca sınıf adına göre gruplayınca şu çıkıyordu:

        Gray Mold    5 adet    %81

    "Gray Mold" HEM yaprak HEM meyve modelinde tanımlı. 3'ü yaprakta,
    2'si meyvedeyse kullanıcı bunu göremiyordu — oysa meyvedeki kurşuni küf
    acil hasat/imha, yapraktaki havalandırma demektir. Yanlış tarımsal
    karar doğuran sessiz bir kayıptı.
"""

from dataclasses import dataclass

import pytest

from app import modeller, siniflar, sonuc_ozeti


@dataclass
class SahteTespit:
    sinif_adi: str
    guven: float = 0.8
    organ: str = ''


TEDAVI = {
    'Gray Mold': {'aciliyet': 'yuksek', 'onlem': ['Enfekte kısımları uzaklaştırın']},
    'Leaf Spot': {'aciliyet': 'orta', 'onlem': ['Alt yaprakları seyreltin']},
    'strawberry_ripe': {'aciliyet': 'bilgi'},
}


@pytest.fixture(autouse=True)
def temiz():
    modeller.bosalt_kutuk()
    siniflar.bosalt_onbellek()
    yield
    modeller.bosalt_kutuk()
    siniflar.bosalt_onbellek()


class TestOrganAyrimi:
    def test_ayni_sinif_iki_organda_ayri_gosterilir(self):
        """ASIL HATA: yapraktaki ve meyvedeki Gray Mold tek satırda birleşiyordu."""
        tespitler = [
            SahteTespit('Gray Mold', 0.81, 'Fruit'),
            SahteTespit('Gray Mold', 0.77, 'Fruit'),
            SahteTespit('Gray Mold', 0.64, 'Leaf'),
            SahteTespit('Gray Mold', 0.60, 'Leaf'),
            SahteTespit('Gray Mold', 0.58, 'Leaf'),
        ]
        iz = {'organlar': {'Leaf': 5, 'Fruit': 3}}
        o = sonuc_ozeti.kur(tespitler, iz, TEDAVI, 'cilek')

        organlar = {g.organ: g for g in o.gruplar}
        assert set(organlar) == {'Leaf', 'Fruit'}
        assert organlar['Fruit'].siniflar[0].adet == 2
        assert organlar['Leaf'].siniflar[0].adet == 3
        # Güven de organa göre ayrılmalı, yoksa yapraktaki bulgu meyvenin
        # %81'ini devralıp olduğundan ciddi görünür
        assert organlar['Fruit'].siniflar[0].max_guven == pytest.approx(0.81)
        assert organlar['Leaf'].siniflar[0].max_guven == pytest.approx(0.64)

    def test_acil_grup_uste_cikar(self):
        tespitler = [SahteTespit('Leaf Spot', 0.7, 'Leaf'),
                     SahteTespit('Leaf Spot', 0.7, 'Leaf'),
                     SahteTespit('Leaf Spot', 0.7, 'Leaf'),
                     SahteTespit('Gray Mold', 0.6, 'Fruit')]
        o = sonuc_ozeti.kur(tespitler, {'organlar': {'Leaf': 3, 'Fruit': 1}},
                            TEDAVI, 'cilek')
        assert o.gruplar[0].organ == 'Fruit', 'acil bulgu üstte olmalı'
        assert o.gruplar[0].aciliyet == 'yuksek'

    def test_organsiz_tespitler_kendi_grubuna_duser(self):
        """Miras model ve elle etiketleme organ üretmez — kaybolmamalılar."""
        o = sonuc_ozeti.kur([SahteTespit('Gray Mold', 0.9, '')], {}, TEDAVI, 'cilek')
        assert len(o.gruplar) == 1
        assert o.gruplar[0].organ == ''
        assert o.gruplar[0].tespit_sayisi == 1


class TestGorulenAmaBulunmayan:
    def test_bakildi_bulgu_yok_grubu_acilir(self):
        """5 yaprak görülüp hastalık bulunmadıysa bu AYRI bir bilgidir.

        Tespit listesi boş olduğu için grup yalnızca izden doğabilir.
        """
        o = sonuc_ozeti.kur([SahteTespit('Gray Mold', 0.8, 'Fruit')],
                            {'organlar': {'Leaf': 5, 'Fruit': 1}}, TEDAVI, 'cilek')
        yaprak = next(g for g in o.gruplar if g.organ == 'Leaf')
        assert yaprak.gorulen == 5
        assert yaprak.tespit_sayisi == 0
        assert 'bulgu yok' in yaprak.not_.lower()

    def test_uzman_modeli_olmayan_organ_belirtilir(self):
        """Çiçek için model yok — 'temiz' demek yanlış olur."""
        o = sonuc_ozeti.kur([], {'organlar': {'Flower': 2}}, TEDAVI, 'cilek')
        cicek = next(g for g in o.gruplar if g.organ == 'Flower')
        assert 'değerlendirilmedi' in cicek.not_.lower()

    def test_sahne_metni_organ_sayilarini_verir(self):
        o = sonuc_ozeti.kur([], {'organlar': {'Leaf': 5, 'Fruit': 3}}, TEDAVI, 'cilek')
        assert o.sahne_var
        assert '5 yaprak' in o.sahne_metni
        assert '3 meyve' in o.sahne_metni


class TestKontrolEdilmeyenler:
    def test_meyve_yoksa_olgunluk_kontrol_edilmedi_denir(self):
        """Kullanıcı 'bulgu yok'u 'meyvem sağlıklı' sanmamalı."""
        o = sonuc_ozeti.kur([SahteTespit('Leaf Spot', 0.7, 'Leaf')],
                            {'organlar': {'Leaf': 4}}, TEDAVI, 'cilek')
        metin = ' '.join(o.kontrol_edilmeyen).lower()
        assert 'meyve' in metin, o.kontrol_edilmeyen

    def test_organ_gorulduyse_o_kontrol_listede_olmaz(self):
        o = sonuc_ozeti.kur([], {'organlar': {'Leaf': 2, 'Fruit': 2}}, TEDAVI, 'cilek')
        for satir in o.kontrol_edilmeyen:
            assert 'yaprak' not in satir.lower() or 'meyve' in satir.lower()


class TestBozukVeriCokmez:
    @pytest.mark.parametrize('iz', ['', None, 'bozuk json {{', '[]', '123'])
    def test_bozuk_iz_akisi_kesmez(self, iz):
        o = sonuc_ozeti.kur([SahteTespit('Gray Mold', 0.8, 'Fruit')], iz, TEDAVI, 'cilek')
        assert o.gruplar and o.gruplar[0].tespit_sayisi == 1
        assert o.organ_sayilari == {}

    def test_tespit_yoksa_ve_iz_yoksa_bos_ozet(self):
        o = sonuc_ozeti.kur([], None, TEDAVI, 'cilek')
        assert o.gruplar == []
        assert not o.sahne_var

    def test_guven_none_ise_cokmez(self):
        o = sonuc_ozeti.kur([SahteTespit('Gray Mold', None, 'Fruit')], {}, TEDAVI, 'cilek')
        assert o.gruplar[0].siniflar[0].max_guven == 0.0


class TestGorunum:
    def test_bilinmeyen_organ_varsayilan_gorunum_alir(self):
        g = sonuc_ozeti.gorunum('bilinmeyen_sey')
        assert g['simge'] and g['baslik']

    def test_organ_adi_buyuk_kucuk_harf_duyarsiz(self):
        assert sonuc_ozeti.gorunum('leaf') == sonuc_ozeti.gorunum('Leaf')
