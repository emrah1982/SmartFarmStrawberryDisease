"""Kareler arası takip — video/drone'da BENZERSİZ sayım.

NEDEN?
    Video işlenirken her karenin kutuları biriktiriliyordu, eşleştirme
    yoktu. Ölçüldü: 4 meyveli SABİT sahne, 4 kare → 11 kutu. Kullanıcı
    bunu "11 hastalıklı meyve" diye okursa yanlış tarımsal karar verir.

FPS NEDEN ÖNEMLİ (kullanıcının tespiti)
    Örnekleme sabit KARE adımıyla yapılıyordu (her 15. kare): 30 fps'te
    0,5 sn, 60 fps'te 0,25 sn. Aynı ayar farklı videolarda farklı davranır
    ve takibin arama penceresi kayar. Süre sabitlenmeli.
"""

from dataclasses import dataclass

import pytest

from app import takip


@dataclass
class K:
    """Test kutusu — Kutu'nun takip için gereken alanları."""
    sinif_adi: str
    x: float
    y: float
    w: float = 0.1
    h: float = 0.1
    guven: float = 0.8


class TestOrneklemeAdimi:
    def test_fpse_gore_hesaplanir(self):
        """0,5 sn: 30 fps'te 15 kare, 60 fps'te 30 kare."""
        assert takip.ornekleme_adimi(30, 0.5) == 15
        assert takip.ornekleme_adimi(60, 0.5) == 30

    def test_ayni_sure_farkli_fps_ayni_davranir(self):
        """ASIL NOKTA: iki video da saniyede 2 kare örneklemeli."""
        for fps in (24, 25, 30, 50, 60, 120):
            adim = takip.ornekleme_adimi(fps, 0.5)
            assert abs(adim / fps - 0.5) < 0.05, f'{fps} fps sapıyor'

    def test_fps_okunamazsa_makul_deger(self):
        assert takip.ornekleme_adimi(0, 0.5) == 15
        assert takip.ornekleme_adimi(None, 0.5) == 15

    def test_en_az_bir_kare(self):
        assert takip.ornekleme_adimi(30, 0.001) >= 1


class TestSabitSahne:
    """Kamera hareketsiz, nesneler yerinde — hepsi TEK nesne sayılmalı."""

    def test_ayni_nesne_tekrar_sayilmaz(self):
        t = takip.Takipci(fps=30)
        kutular = [K('strawberry_ripe', 0.2, 0.5), K('strawberry_ripe', 0.5, 0.5),
                   K('strawberry_ripe', 0.8, 0.5), K('strawberry_unripe', 0.35, 0.7)]
        for kare in (0, 15, 30, 45):
            t.ekle(kare, kutular)
        assert t.benzersiz_toplam == 4, f'4 nesne bekleniyordu: {t.benzersiz_sayim()}'
        assert t.benzersiz_sayim() == {'strawberry_ripe': 3, 'strawberry_unripe': 1}

    def test_olculen_gercek_durum(self):
        """Ölçülen olay: 4 nesne, 4 kare, takipsiz 11 kutu birikiyordu."""
        t = takip.Takipci(fps=30)
        kareler = [
            [K('strawberry_ripe', 0.2, 0.5), K('strawberry_ripe', 0.5, 0.5)],
            [K('strawberry_ripe', 0.2, 0.5), K('strawberry_ripe', 0.5, 0.5),
             K('strawberry_unripe', 0.8, 0.5)],
            [K('strawberry_ripe', 0.2, 0.5), K('strawberry_ripe', 0.5, 0.5),
             K('strawberry_unripe', 0.8, 0.5)],
            [K('strawberry_ripe', 0.2, 0.5), K('strawberry_ripe', 0.5, 0.5),
             K('strawberry_unripe', 0.8, 0.5)],
        ]
        toplam_kutu = 0
        for i, kutular in enumerate(kareler):
            t.ekle(i * 15, kutular)
            toplam_kutu += len(kutular)
        assert toplam_kutu == 11, 'ölçülen kutu sayısı'
        assert t.benzersiz_toplam == 3, 'gerçek nesne sayısı'


class TestHareketliKamera:
    def test_kayan_nesne_takip_edilir(self):
        """Yürüyerek çekimde nesne kadrajda kayar ama aynı nesnedir."""
        t = takip.Takipci(fps=30)
        for i, x in enumerate([0.5, 0.44, 0.38, 0.32]):
            t.ekle(i * 15, [K('strawberry_ripe', x, 0.5)])
        assert t.benzersiz_toplam == 1, t.benzersiz_sayim()

    def test_cok_uzaga_atlayan_YENI_nesnedir(self):
        """Kadrajın öbür ucundaki kutu aynı nesne olamaz."""
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('strawberry_ripe', 0.1, 0.1)])
        t.ekle(15, [K('strawberry_ripe', 0.9, 0.9)])
        assert t.benzersiz_toplam == 2

    def test_arama_penceresi_sureyle_buyur(self):
        """2 saniye sonra görülen nesne daha uzağa gitmiş olabilir."""
        yakin = takip.Takipci(fps=30)
        yakin.ekle(0, [K('strawberry_ripe', 0.5, 0.5)])
        yakin.ekle(6, [K('strawberry_ripe', 0.62, 0.5)])     # 0,2 sn
        assert yakin.benzersiz_toplam == 2, 'kısa sürede bu kadar kayamaz'

        uzun = takip.Takipci(fps=30)
        uzun.ekle(0, [K('strawberry_ripe', 0.5, 0.5)])
        uzun.ekle(30, [K('strawberry_ripe', 0.62, 0.5)])     # 1,0 sn
        assert uzun.benzersiz_toplam == 1, 'uzun sürede kayabilir'


class TestSinifAyrimi:
    def test_farkli_sinif_birlestirilmez(self):
        """Olgun ve olgunlaşmamış çilek aynı nesne sayılmamalı."""
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('strawberry_ripe', 0.5, 0.5)])
        t.ekle(15, [K('strawberry_unripe', 0.5, 0.5)])
        assert t.benzersiz_toplam == 2

    def test_ayni_konumda_iki_sinif_ayri_sayilir(self):
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('Gray Mold', 0.5, 0.5), K('strawberry_ripe', 0.5, 0.5)])
        assert t.benzersiz_toplam == 2


class TestKayipTolerans:
    def test_kisa_sure_kaybolan_ayni_nesnedir(self):
        """Yaprak arkasına giren meyve iki ayrı nesne sayılmamalı."""
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('strawberry_ripe', 0.5, 0.5)])
        t.ekle(15, [])                                  # kayboldu
        t.ekle(30, [K('strawberry_ripe', 0.5, 0.5)])    # 1 sn sonra geri geldi
        assert t.benzersiz_toplam == 1

    def test_uzun_sure_kaybolan_YENI_nesnedir(self):
        t = takip.Takipci(fps=30, kayip_tolerans_sn=0.5)
        t.ekle(0, [K('strawberry_ripe', 0.5, 0.5)])
        t.ekle(90, [K('strawberry_ripe', 0.5, 0.5)])    # 3 sn sonra
        assert t.benzersiz_toplam == 2


class TestDayaniklilik:
    def test_bos_kare_cokmez(self):
        t = takip.Takipci(fps=30)
        assert t.ekle(0, []) == []
        assert t.benzersiz_toplam == 0

    def test_bir_kutu_bir_ize_baglanir(self):
        """İki kutu aynı ize bağlanmamalı — sayım eksik çıkardı."""
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('strawberry_ripe', 0.5, 0.5)])
        kimlikler = t.ekle(15, [K('strawberry_ripe', 0.5, 0.5),
                                K('strawberry_ripe', 0.52, 0.5)])
        assert len(set(kimlikler)) == 2, 'iki kutu aynı kimliği almış'

    def test_fps_sifir_ise_varsayilan(self):
        t = takip.Takipci(fps=0)
        assert t.fps > 0

    def test_ozet_kaydedilebilir(self):
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('strawberry_ripe', 0.5, 0.5)])
        o = t.ozet()
        assert o['benzersiz'] == 1
        assert o['sinif'] == {'strawberry_ripe': 1}
        assert o['fps'] == 30.0


class TestCizgiSayaci:
    """Sanal çizgi geçiş sayımı — drone/transekt için.

    KULLANICININ FİKRİ, ve doğru bir sezgiyle "bu drone/video için" dedi.
    Uzun taramalarda benzersiz-iz sayımından sağlamdır: iz kopup yeniden
    kurulsa bile nesne çizgiyi bir kez geçmiştir. Ama SABİT çekimde
    hiçbir şey geçmez ve sayı 0 kalır — bu yüzden varsayılan kapalı.
    """

    def test_gecen_nesne_sayilir(self):
        c = takip.CizgiSayaci(eksen='x', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        # Hareket takipçinin izin verdiği pencere içinde olmalı (0,5 sn ×
        # 0,35 = 0,175); daha uzağa atlayan kutu AYRI nesne sayılır.
        t.ekle(0, [K('strawberry_ripe', 0.42, 0.5)])
        t.ekle(15, [K('strawberry_ripe', 0.58, 0.5)])     # çizgiyi geçti
        assert c.toplam == 1
        assert c.ileri == {'strawberry_ripe': 1}

    def test_gecmeyen_nesne_sayilmaz(self):
        c = takip.CizgiSayaci(eksen='x', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        t.ekle(0, [K('strawberry_ripe', 0.2, 0.5)])
        t.ekle(15, [K('strawberry_ripe', 0.3, 0.5)])
        assert c.toplam == 0

    def test_ayni_iz_IKI_KEZ_sayilmaz(self):
        """Dur-kalk yürüyüşte nesne çizgi üstünde titreyebilir."""
        c = takip.CizgiSayaci(eksen='x', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        t.ekle(0, [K('strawberry_ripe', 0.45, 0.5)])
        t.ekle(15, [K('strawberry_ripe', 0.55, 0.5)])    # geçti
        t.ekle(30, [K('strawberry_ripe', 0.45, 0.5)])    # geri döndü
        t.ekle(45, [K('strawberry_ripe', 0.55, 0.5)])    # yine geçti
        assert c.toplam == 1, 'aynı nesne bir kez sayılmalı'

    def test_yon_ayirt_edilir(self):
        c = takip.CizgiSayaci(eksen='x', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        t.ekle(0, [K('a', 0.58, 0.5)])
        t.ekle(15, [K('a', 0.42, 0.5)])                   # sağdan sola
        assert c.geri == {'a': 1} and not c.ileri

    def test_y_ekseni(self):
        c = takip.CizgiSayaci(eksen='y', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        t.ekle(0, [K('a', 0.5, 0.42)])
        t.ekle(15, [K('a', 0.5, 0.58)])
        assert c.toplam == 1

    def test_SABIT_cekimde_sayim_sifir(self):
        """Bu sınırın testi: kamera durunca çizgi sayımı 0 verir.

        Kadrajda 3 meyve olsa bile. Bu yüzden çizgi sayımı benzersiz-iz
        sayımının YERİNE değil YANINDA kullanılmalı.
        """
        c = takip.CizgiSayaci(eksen='x', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        kutular = [K('a', 0.2, 0.5), K('a', 0.5, 0.5), K('a', 0.8, 0.5)]
        for kare in (0, 15, 30, 45):
            t.ekle(kare, kutular)
        assert c.toplam == 0, 'sabit sahnede geçiş olmaz'
        assert t.benzersiz_toplam == 3, 'ama nesneler orada — iz sayımı doğru'

    def test_varsayilan_kapali(self):
        """Çizgi sayacı yalnızca istenirse çalışmalı."""
        t = takip.Takipci(fps=30)
        assert t.cizgi is None
        assert 'cizgi' not in t.ozet()

    def test_gecersiz_eksen_reddedilir(self):
        with pytest.raises(ValueError):
            takip.CizgiSayaci(eksen='z')

    def test_ozet_kaydedilebilir(self):
        c = takip.CizgiSayaci(eksen='x', konum=0.4)
        t = takip.Takipci(fps=30, cizgi=c)
        t.ekle(0, [K('a', 0.33, 0.5)])
        t.ekle(15, [K('a', 0.47, 0.5)])
        o = t.ozet()['cizgi']
        assert o['eksen'] == 'x' and o['konum'] == 0.4 and o['toplam'] == 1


class TestHareketOlcumu:
    """SABİT mi HAREKETLİ mi — kullanıcıya sormadan ÖLÇÜLÜR.

    Çizgi sayımı yalnızca kamera ilerlerken anlamlıdır. "Video mu sabit mi"
    diye sormak yerine izlerin kayması ölçülür: tutarlı kayma varsa kamera
    ilerliyordur.
    """

    def test_sabit_sahne_sabit_bildirilir(self):
        t = takip.Takipci(fps=30)
        kutular = [K('a', 0.3, 0.5), K('a', 0.7, 0.5)]
        for kare in (0, 15, 30, 45):
            t.ekle(kare, kutular)
        o = t.sayim_onerisi()
        assert o['kamera'] == 'sabit'
        assert o['onerilen'] == 'benzersiz'
        assert o['kayma'] < t.HAREKET_ESIGI

    def test_kayan_sahne_hareketli_bildirilir(self):
        t = takip.Takipci(fps=30)
        for i, x in enumerate([0.70, 0.62, 0.54, 0.46, 0.38]):
            t.ekle(i * 15, [K('a', x, 0.5)])
        o = t.sayim_onerisi()
        assert o['kamera'] == 'hareketli', o
        assert o['kayma'] >= t.HAREKET_ESIGI

    def test_hareketli_ve_cizgi_varsa_cizgi_onerilir(self):
        c = takip.CizgiSayaci(eksen='x', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        for i, x in enumerate([0.70, 0.62, 0.54, 0.46, 0.38]):
            t.ekle(i * 15, [K('a', x, 0.5)])
        assert t.sayim_onerisi()['onerilen'] == 'cizgi'

    def test_sabit_sahnede_cizgi_ONERILMEZ(self):
        """ASIL AYRIM: sabit çekimde çizgi 0 verir, önerilmemeli."""
        c = takip.CizgiSayaci(eksen='x', konum=0.5)
        t = takip.Takipci(fps=30, cizgi=c)
        for kare in (0, 15, 30, 45):
            t.ekle(kare, [K('a', 0.3, 0.5), K('a', 0.7, 0.5)])
        o = t.sayim_onerisi()
        assert o['kamera'] == 'sabit'
        assert o['onerilen'] == 'benzersiz', 'sabit çekimde çizgi önerilmemeli'
        assert o['cizgi'] == 0

    def test_tek_karelik_iz_hareket_olcumune_girmez(self):
        """Bir kez görülen izin kayması ölçülemez."""
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('a', 0.5, 0.5)])
        assert t.ortalama_kayma == 0.0

    def test_ozette_oneri_var(self):
        t = takip.Takipci(fps=30)
        t.ekle(0, [K('a', 0.5, 0.5)])
        assert 'oneri' in t.ozet()


class TestZamanliEkleme:
    """Canlı akış: kareler DÜZENSİZ aralıklarla gelir."""

    def test_gercek_zaman_kullanilir(self):
        t = takip.Takipci(fps=1.0)
        t.ekle_zamanli(0.0, [K('a', 0.5, 0.5)])
        t.ekle_zamanli(0.2, [K('a', 0.53, 0.5)])
        assert t.benzersiz_toplam == 1

    def test_uzun_bosluk_yeni_nesne(self):
        t = takip.Takipci(fps=1.0, kayip_tolerans_sn=1.0)
        t.ekle_zamanli(0.0, [K('a', 0.5, 0.5)])
        t.ekle_zamanli(5.0, [K('a', 0.5, 0.5)])
        assert t.benzersiz_toplam == 2


class TestIou:
    def test_tam_ortusme(self):
        a = K('x', 0.5, 0.5, 0.2, 0.2)
        assert takip._iou(a, a) == pytest.approx(1.0)

    def test_ayrik_kutular(self):
        assert takip._iou(K('x', 0.1, 0.1), K('x', 0.9, 0.9)) == 0.0

    def test_kismi_ortusme(self):
        a = K('x', 0.5, 0.5, 0.2, 0.2)
        b = K('x', 0.6, 0.5, 0.2, 0.2)
        assert 0 < takip._iou(a, b) < 1
