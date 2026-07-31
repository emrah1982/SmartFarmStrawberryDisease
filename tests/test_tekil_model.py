"""Boru hattına GİRMEYEN modeller (rol: tekil) gerçekten girmiyor mu?

NEDEN TEST?
    `bocek_teshis` modeli 416x416 makro böcek fotoğraflarıyla eğitilir;
    kutu alanı medyanı karenin %15'idir. ROI boru hattı ise saha
    görüntüsünden kırpılmış yaprak/meyve parçası verir — zararlı orada
    kırpıntının %1'inden azını kaplar.

    Bu model yanlışlıkla ROI akışına bağlanırsa (kütükte `tetik` dolu
    yazılırsa) her yaprak kırpıntısında yanlış ölçekte çalışır ve
    gerçekte olmayan böcekler bulur. Hata sessizdir: sistem çalışıyor
    görünür, sonuçlar saçmadır.

    Güvence tek satırlık bir yapılandırmaya dayandığı için testle
    sabitlenmiştir.
"""

from pathlib import Path

import pytest
import yaml

from app import modeller, siniflar

KOK = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def temiz():
    modeller.bosalt_kutuk()
    siniflar.bosalt_onbellek()
    yield
    modeller.bosalt_kutuk()
    siniflar.bosalt_onbellek()


ORGANLAR = ('Leaf', 'Fruit', 'Flower', 'leaf', 'fruit', 'flower')


class TestTekilModelIzoleKalir:
    def test_hicbir_organ_bocek_teshisi_tetiklemez(self):
        for organ in ORGANLAR:
            adlar = [t.ad for t in modeller.tetiklenen(organ, 'cilek')]
            assert 'bocek_teshis' not in adlar, (
                f'{organ} organı bocek_teshis modelini tetikledi — makro böcek '
                'modeli ROI kırpıntısında çalışırsa saçma tespit üretir')

    def test_tetik_listesi_bos(self):
        t = modeller.tanim('bocek_teshis', 'cilek')
        assert t is not None, 'bocek_teshis kütükte tanımlı olmalı'
        assert t.tetik == [], f'tetik BOŞ olmalı, bulunan: {t.tetik}'
        assert t.rol == 'tekil'

    def test_organ_rolu_degil(self):
        """rol='organ' olsaydı boru hattının giriş modeli olurdu."""
        organlar = [t.ad for t in modeller.tanimlar('cilek').values() if t.rol == 'organ']
        assert organlar == ['organ']

    def test_hiyerarsi_hazirligini_etkilemez(self):
        """Dosyası yokken bile sistem 'organ modeli var mı' kararını bozmamalı."""
        hazir = modeller.hiyerarsik_hazir('cilek')
        organ = modeller.tanim('organ', 'cilek')
        assert hazir == (organ.var and organ.aktif)


class TestKutukTutarli:
    def test_siniflar_dataset_ile_ayni(self):
        """Kütükteki sıra dataset'ten sapmışsa model_kur.py kurulumu reddeder."""
        veri = KOK / 'datasets' / 'cilek' / 'bocek_teshis' / 'data.yaml'
        if not veri.exists():
            pytest.skip('bocek_teshis dataset paketi yok')
        cfg = yaml.safe_load(veri.read_text(encoding='utf-8'))
        n = cfg['names']
        dataset = [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)
        kutuk = modeller.tanim('bocek_teshis', 'cilek').siniflar
        assert kutuk == dataset, (
            f'sıra/ad uyuşmuyor.\n  kütük  : {kutuk}\n  dataset: {dataset}')

    def test_turkce_adlar_tanimli(self):
        for ad in modeller.tanim('bocek_teshis', 'cilek').siniflar:
            tr = siniflar.bilgi(ad, 'cilek').get('tr')
            assert tr, f'{ad} için Türkçe ad yok (configs/urunler/cilek/siniflar.yaml)'

    def test_bocek_siniflarina_id_verilmemis(self):
        """ID birleşik modelin etiket dosyalarındaki sayıdır; bu model kendi
        dataset'inde 0-5 kullanır. ID verilirse etiketleme ekranındaki
        numaralarla çakışır ve geçmiş etiketler yanlış sınıfa kayar."""
        for ad in modeller.tanim('bocek_teshis', 'cilek').siniflar:
            kimlik = siniflar.bilgi(ad, 'cilek').get('id')
            assert kimlik is None, f'{ad} sınıfına id verilmiş ({kimlik}) — çakışma riski'

    def test_saha_zararli_modeli_kapali(self):
        """Verisi olmayan model açık kalırsa boru hattı her karede boşuna arar."""
        z = modeller.tanim('zararli', 'cilek')
        assert z.aktif is False, 'zararli modelinin verisi yok, aktif olmamalı'


class TestCikarimCozunurlugu:
    """Her model kendi ÇIKARIM imgsz'ini taşımalı.

    GERÇEK HATA: tek genel imgsz (1024) bütün modellere dayatılıyordu.
    Uzman modeller TAM görüntüyü değil ROI KIRPINTISINI görür (60-250 px);
    1024'e büyütülünce neredeyse çöküyorlardı. Bir sera fotoğrafında
    ölçüldü:

        1024 dayatılmış : 1 tespit
        model başına    : 4 tespit

    Kırpıntıdaki olgunluk modeli 1024'te 0.066, 128'de 0.906 güven veriyordu.
    """

    @pytest.mark.parametrize('ad', ['organ', 'yaprak_hastalik',
                                    'meyve_hastalik', 'olgunluk'])
    def test_kurulu_modellerin_imgszi_tanimli(self, ad):
        t = modeller.tanim(ad, 'cilek')
        assert t.imgsz, f'{ad}: imgsz tanımsız — genel 1024 dayatılır'

    def test_uzman_modeller_organdan_kucuk_cozunurlukte(self):
        """ROI kırpıntısı tam görüntüden küçüktür; imgsz de öyle olmalı."""
        organ = modeller.tanim('organ', 'cilek').imgsz
        for ad in ('yaprak_hastalik', 'meyve_hastalik', 'olgunluk'):
            t = modeller.tanim(ad, 'cilek')
            assert t.imgsz < organ, f'{ad}: {t.imgsz} >= organ {organ}'

    def test_imgsz_32nin_kati(self):
        for t in modeller.tanimlar('cilek').values():
            if t.imgsz:
                assert t.imgsz % 32 == 0, f'{t.ad}: {t.imgsz}'

    def test_organ_esigi_comert(self):
        """Kaçırılan organ tüm zinciri keser; yanlış ROI ise zararsızdır."""
        assert modeller.tanim('organ', 'cilek').esik <= 0.25


class TestSinifAdlariTekil:
    """Aynı canlı iki farklı sınıf adıyla anılmamalı.

    GERÇEK HATA: böcek teşhis dataseti kırmızı örümceği "Red Spider Mite",
    saha zararlı modeli "Spider Mites" diyordu. İkisinin Türkçe adı da
    "Kırmızı Örümcek"ti. Sonucu:
      - arayüzde iki ayrı "Kırmızı Örümcek" satırı
      - iki ayrı tedavi metni yazma zorunluluğu
      - geçmiş sorgusunda aynı zararlının kayıtlarının ikiye bölünmesi

    Modeller ayrı kalır (girdi alanları farklı), sınıf ADI paylaşılır.
    """

    def test_ayni_turkce_ad_iki_sinifa_verilmemis(self):
        tr_haritasi = {}
        for t in modeller.tanimlar('cilek').values():
            for ad in t.siniflar:
                tr = siniflar.bilgi(ad, 'cilek').get('tr')
                if tr:
                    tr_haritasi.setdefault(tr, set()).add(ad)
        cakisan = {tr: sorted(adlar) for tr, adlar in tr_haritasi.items()
                   if len(adlar) > 1}
        assert not cakisan, f'aynı Türkçe ad birden çok sınıfta: {cakisan}'

    def test_kirmizi_orumcek_tek_sinif_adiyla(self):
        zararli = modeller.tanim('zararli', 'cilek').siniflar
        bocek = modeller.tanim('bocek_teshis', 'cilek').siniflar
        ortak = set(zararli) & set(bocek)
        assert ortak == {'Spider Mites'}, (
            f'iki modelin ortak sınıfı yalnızca kırmızı örümcek olmalı: {ortak}')
        assert 'Red Spider Mite' not in bocek

    def test_ortak_sinif_tek_tedavi_kaydi_kullanir(self):
        from app import tedavi
        kutuk = tedavi.yukle('cilek')
        assert 'Spider Mites' in kutuk
        assert 'Red Spider Mite' not in kutuk, 'ikinci kayıt tekrar üretir'


class TestPaketTemizlendi:
    """harici_paket_duzelt.py sızıntıyı gerçekten kesti mi?

    Ham Roboflow paketinde valid'in %99,6'sı train görüntülerinin
    artırılmış kopyasıydı (aynı fotoğraf, biri döndürülmüş). Bölme
    kaynak grubuna göre yeniden yapıldı.
    """

    @staticmethod
    def _kaynaklar(bolum: Path):
        import importlib.util
        import sys
        yol = KOK / 'scripts' / 'harici_paket_duzelt.py'
        spec = importlib.util.spec_from_file_location('hpd', yol)
        m = importlib.util.module_from_spec(spec)
        sys.modules['hpd'] = m
        spec.loader.exec_module(m)
        return {m.kaynak_kimligi(p.name) for p in (bolum / 'images').iterdir()}

    def test_bolmeler_arasi_ortak_kaynak_yok(self):
        kok = KOK / 'datasets' / 'cilek' / 'bocek_teshis'
        if not kok.is_dir():
            pytest.skip('bocek_teshis dataset paketi yok')
        kume = {b: self._kaynaklar(kok / b)
                for b in ('train', 'valid', 'test') if (kok / b / 'images').is_dir()}
        adlar = list(kume)
        for i, a in enumerate(adlar):
            for b in adlar[i + 1:]:
                ortak = kume[a] & kume[b]
                assert not ortak, (
                    f'{a} ↔ {b} arasında {len(ortak)} ortak kaynak fotoğraf — '
                    'doğrulama skoru anlamsız olur')

    def test_data_yaml_tasinabilir(self):
        y = KOK / 'datasets' / 'cilek' / 'bocek_teshis' / 'data.yaml'
        if not y.exists():
            pytest.skip('bocek_teshis dataset paketi yok')
        cfg = yaml.safe_load(y.read_text(encoding='utf-8'))
        assert 'path' not in cfg, "'path' yazılmamalı — klasör taşınınca bozulur"
        for k in ('train', 'val', 'test'):
            if k in cfg:
                assert not str(cfg[k]).startswith('..'), (
                    f'{k}: {cfg[k]!r} bir üst dizine çıkıyor')
                assert (y.parent / str(cfg[k])).is_dir(), f'{k} dizini yok: {cfg[k]}'
