"""Tedavi önerisinin ORGANA göre çözülmesi.

NEDEN TEST?
    Botrytis hem yaprakta hem meyvede aynı sınıf olarak tespit edilir
    ("Gray Mold" — model etiketleri öyle). Ama yapılacak iş farklıdır:

        meyvede : ACİL — enfekte meyveyi hemen topla, hasadı geciktirme
        yaprakta: ORTA — yaşlı yaprakları temizle, nem ve havalandırma

    Öneri sınıf bazlı kaldığı sürece yapraktaki bulguya "Meyvede gri, tozlu
    küf tabakası" metni gösteriliyordu. Kullanıcı olmayan bir belirtiyi arar,
    yapılması gerekeni yapmazdı.

    İki hastalığı ayrı SINIF yapmak çözüm değil: model ikisini de aynı
    öğrenir. Ayrım sunum katmanında yapılmalı — test ettiğimiz şey bu.
"""

import pytest

from app import tedavi

KUTUK = {
    'Gray Mold': {
        'ad': 'Kurşuni Küf',
        'etken': 'Botrytis cinerea',
        'aciliyet': 'yuksek',
        'belirti': 'Gri, tozlu küf tabakası.',
        'onlem': ['Ortak önlem A', 'Ortak önlem B'],
        'uzman': 'Ortak uzman notu',
        'organ': {
            'fruit': {
                'belirti': 'Meyvede gri küf; hızlı çürüme.',
                'onlem': ['Meyveyi HEMEN toplayın'],
            },
            'leaf': {
                'aciliyet': 'orta',
                'belirti': 'Yaşlı yaprakta kahverengi kuruma.',
                'onlem': ['Yaşlı yaprakları temizleyin'],
                'uzman': 'Yaprak odağı çiçeklenmeye denk gelirse risk artar',
            },
        },
    },
    'Leaf Spot': {          # organ bloğu YOK — eski biçim
        'ad': 'Yaprak Lekesi',
        'aciliyet': 'dusuk',
        'belirti': 'Mor kenarlı lekeler.',
        'onlem': ['Alt yaprakları seyreltin'],
    },
}


class TestOrganaGoreCozme:
    def test_meyve_kendi_metnini_alir(self):
        t = tedavi.coz(KUTUK, 'Gray Mold', 'Fruit')
        assert t['belirti'] == 'Meyvede gri küf; hızlı çürüme.'
        assert t['onlem'] == ['Meyveyi HEMEN toplayın']

    def test_yaprak_kendi_metnini_ve_ACILIYETINI_alir(self):
        """Aciliyet de organa göre değişir; yapraktaki botrytis acil değil."""
        t = tedavi.coz(KUTUK, 'Gray Mold', 'Leaf')
        assert t['aciliyet'] == 'orta'
        assert 'Yaşlı yaprakta' in t['belirti']
        assert t['uzman'].startswith('Yaprak odağı')

    def test_verilmeyen_alan_ortaktan_gelir(self):
        """fruit bloğunda aciliyet/etken yok — ortaktan devralınmalı."""
        t = tedavi.coz(KUTUK, 'Gray Mold', 'Fruit')
        assert t['aciliyet'] == 'yuksek'
        assert t['etken'] == 'Botrytis cinerea'
        assert t['uzman'] == 'Ortak uzman notu'

    def test_onlem_listesi_BIRLESTIRILMEZ_degistirilir(self):
        """Kısmi birleştirme alakasız tavsiye üretirdi.

        Yaprak bulgusuna "Meyveyi hemen toplayın" maddesi eklenirse
        kullanıcı olmayan bir işi yapmaya çalışır.
        """
        t = tedavi.coz(KUTUK, 'Gray Mold', 'Leaf')
        assert t['onlem'] == ['Yaşlı yaprakları temizleyin']
        assert 'Ortak önlem A' not in t['onlem']

    def test_gorunen_ad_organa_gore_DEGISMEZ(self):
        """Aynı hastalık iki isimle anılmamalı."""
        assert (tedavi.coz(KUTUK, 'Gray Mold', 'Leaf')['ad']
                == tedavi.coz(KUTUK, 'Gray Mold', 'Fruit')['ad'])

    def test_buyuk_kucuk_harf_duyarsiz(self):
        assert tedavi.coz(KUTUK, 'Gray Mold', 'FRUIT') == tedavi.coz(KUTUK, 'Gray Mold', 'fruit')


class TestGeriyeDonukUyum:
    def test_organ_blogu_olmayan_sinif_eskisi_gibi(self):
        for organ in ('', 'Leaf', 'Fruit', 'Flower'):
            t = tedavi.coz(KUTUK, 'Leaf Spot', organ)
            assert t['belirti'] == 'Mor kenarlı lekeler.'
            assert t['aciliyet'] == 'dusuk'

    def test_organ_bos_ise_ortak_metin(self):
        """Miras model ve elle etiketleme organ üretmez."""
        t = tedavi.coz(KUTUK, 'Gray Mold', '')
        assert t['belirti'] == 'Gri, tozlu küf tabakası.'
        assert 'organ' not in t, 'organ bloğu çıktıya sızmamalı'

    def test_tanimsiz_organ_ortak_metne_duser(self):
        t = tedavi.coz(KUTUK, 'Gray Mold', 'Stem')
        assert t['belirti'] == 'Gri, tozlu küf tabakası.'

    def test_bilinmeyen_sinif_bos_doner(self):
        assert tedavi.coz(KUTUK, 'Olmayan Hastalik', 'Leaf') == {}

    def test_bos_kutuk_cokmez(self):
        assert tedavi.coz({}, 'Gray Mold', 'Leaf') == {}
        assert tedavi.coz(None, 'Gray Mold', 'Leaf') == {}


class TestGercekYapilandirma:
    """Depodaki tedavi_onerileri.yaml gerçekten çalışıyor mu?"""

    @pytest.fixture
    def kutuk(self):
        return tedavi.yukle('cilek')

    def test_gray_mold_organa_ozel(self, kutuk):
        assert tedavi.organa_ozel_mi(kutuk, 'Gray Mold')
        meyve = tedavi.coz(kutuk, 'Gray Mold', 'Fruit')
        yaprak = tedavi.coz(kutuk, 'Gray Mold', 'Leaf')
        assert meyve['belirti'] != yaprak['belirti']
        assert meyve['onlem'] != yaprak['onlem']
        assert meyve['aciliyet'] == 'yuksek' and yaprak['aciliyet'] == 'orta'

    def test_yapraktaki_metinde_meyve_hasadi_gecmez(self, kutuk):
        """Asıl şikâyet buydu: yaprak bulgusuna meyve talimatı gösteriliyordu."""
        yaprak = tedavi.coz(kutuk, 'Gray Mold', 'Leaf')
        metin = ' '.join(yaprak.get('onlem', [])).lower()
        assert 'hasad' not in metin, yaprak['onlem']
        assert 'soğut' not in metin, yaprak['onlem']

    def test_her_sinifin_gorunen_adi_var(self, kutuk):
        for ad, v in kutuk.items():
            assert (v or {}).get('ad'), f'{ad}: görünen ad yok'

    def test_organ_bloklari_gecerli_organ_adi_kullanir(self, kutuk):
        from app import sonuc_ozeti
        for ad, v in kutuk.items():
            for organ in ((v or {}).get('organ') or {}):
                assert organ in sonuc_ozeti.ORGAN_GORUNUM, (
                    f'{ad}: tanınmayan organ {organ!r} — bu blok hiç kullanılmaz')
