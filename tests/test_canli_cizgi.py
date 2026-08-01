"""Canlı akışta sanal çizgi sayacı — arayüzden açılabilir.

NEDEN AYRIM ÖNEMLİ?
    Çizgi sayımı YALNIZCA kamera tek yönde ilerlerken anlamlıdır (drone
    transekti, sıra boyunca yürüyüş). SABİT çekimde hiçbir şey çizgiyi
    geçmez ve sayı 0 kalır — kadrajda 5 meyve olsa bile.

    Bu yüzden: kendiliğinden açılmaz, kullanıcı bilerek açar; sistem ayrıca
    hareketi ÖLÇER ve sabit çekimde uyarır. Sessizce 0 göstermek, kullanıcıya
    "hiç meyve yok" dedirtirdi.
"""

from dataclasses import dataclass

import pytest

from app.moduller.canli import servis


@dataclass
class K:
    sinif_adi: str
    x: float
    y: float
    w: float = 0.1
    h: float = 0.1
    guven: float = 0.8
    sinif_id: int = 0


class TestOturumCizgiAyari:
    def test_varsayilan_kapali(self):
        """Sabit çekimde 0 vereceği için kendiliğinden açılmamalı."""
        o = servis.OturumKaydi()
        assert o.takipci.cizgi is None
        assert o.cizgi_ozet == {}

    def test_acilabilir(self):
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True, 'x', 0.5)
        assert o.takipci.cizgi is not None
        assert o.cizgi_ozet['eksen'] == 'x'
        assert o.cizgi_ozet['konum'] == 0.5

    def test_kapatilabilir(self):
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True)
        o.cizgi_ayarla(False)
        assert o.takipci.cizgi is None

    def test_ayar_degisince_sayac_sifirlanir(self):
        """Yarı yolda çizgi taşınırsa önceki sayım yeni çizgiye ait değildir."""
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True, 'x', 0.5)
        o.kareyi_izle([K('a', 0.42, 0.5)], 0.0)
        o.kareyi_izle([K('a', 0.58, 0.5)], 0.5)
        assert o.cizgi_ozet['toplam'] == 1

        o.cizgi_ayarla(True, 'x', 0.8)          # çizgi taşındı
        assert o.cizgi_ozet['toplam'] == 0

    def test_konum_sinirlanir(self):
        """Kenardaki çizgiyi hiçbir şey geçemez; kullanılabilir aralıkta tut."""
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True, 'x', 0.0)
        assert o.cizgi_ozet['konum'] >= 0.05
        o.cizgi_ayarla(True, 'x', 1.5)
        assert o.cizgi_ozet['konum'] <= 0.95

    def test_gecersiz_eksen_x_e_duser(self):
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True, 'z', 0.5)
        assert o.cizgi_ozet['eksen'] == 'x'


class TestSayim:
    def test_gecen_nesne_sayilir(self):
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True, 'x', 0.5)
        o.kareyi_izle([K('strawberry_ripe', 0.42, 0.5)], 0.0)
        o.kareyi_izle([K('strawberry_ripe', 0.58, 0.5)], 0.5)
        assert o.cizgi_ozet['toplam'] == 1

    def test_SABIT_cekimde_sifir_kalir(self):
        """Asıl sınır: kamera durunca çizgi sayımı 0 verir."""
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True, 'x', 0.5)
        kutular = [K('a', 0.3, 0.5), K('a', 0.7, 0.5)]
        for i in range(5):
            o.kareyi_izle(kutular, i * 0.5)
        assert o.cizgi_ozet['toplam'] == 0
        assert o.benzersiz == 2, 'ama nesneler orada — iz sayımı doğru'
        assert o.sayim_onerisi()['kamera'] == 'sabit'
        assert o.sayim_onerisi()['onerilen'] == 'benzersiz'

    def test_hareketli_cekimde_cizgi_onerilir(self):
        o = servis.OturumKaydi()
        o.cizgi_ayarla(True, 'x', 0.5)
        for i, x in enumerate([0.70, 0.62, 0.54, 0.46, 0.38]):
            o.kareyi_izle([K('a', x, 0.5)], i * 0.5)
        oneri = o.sayim_onerisi()
        assert oneri['kamera'] == 'hareketli'
        assert oneri['onerilen'] == 'cizgi'

    def test_cizgi_kapaliyken_benzersiz_onerilir(self):
        o = servis.OturumKaydi()
        for i, x in enumerate([0.70, 0.62, 0.54, 0.46, 0.38]):
            o.kareyi_izle([K('a', x, 0.5)], i * 0.5)
        assert o.sayim_onerisi()['onerilen'] == 'benzersiz'


class TestArayuz:
    def test_kontroller_sayfada(self):
        from pathlib import Path
        kok = Path(__file__).resolve().parent.parent
        html = (kok / 'app' / 'moduller' / 'canli' / 'templates' / 'canli'
                / 'izle.html').read_text(encoding='utf-8')
        assert 'cizgiAcik' in html
        assert 'cizgiEksen' in html and 'cizgiKonum' in html
        assert 'Sabit çekimde işe yaramaz' in html, 'sınır yazılı olmalı'

    def test_js_cizgiyi_tuvale_cizer(self):
        """Kullanıcı nesnenin nereyi geçince sayıldığını GÖRMELİ."""
        from pathlib import Path
        kok = Path(__file__).resolve().parent.parent
        js = (kok / 'app' / 'moduller' / 'canli' / 'static'
              / 'cizim.js').read_text(encoding='utf-8')
        assert 'cizgiGuncelle' in js
        assert 'setLineDash' in js, 'çizgi görsel olarak ayırt edilmeli'

    def test_js_sabit_kamerada_uyarir(self):
        from pathlib import Path
        kok = Path(__file__).resolve().parent.parent
        js = (kok / 'app' / 'moduller' / 'canli' / 'static'
              / 'izle.js').read_text(encoding='utf-8')
        assert 'kamera sabit' in js, 'sabit çekimde uyarı gösterilmeli'

    def test_websocket_cizgi_ayarini_tasir(self):
        from pathlib import Path
        kok = Path(__file__).resolve().parent.parent
        js = (kok / 'app' / 'moduller' / 'canli' / 'static'
              / 'akis.js').read_text(encoding='utf-8')
        assert 'cizgiSec' in js and 'cizgi: this.cizgi' in js

    def test_sunucu_ayari_isliyor(self):
        from pathlib import Path
        kok = Path(__file__).resolve().parent.parent
        py = (kok / 'app' / 'moduller' / 'canli'
              / 'rotalar.py').read_text(encoding='utf-8')
        assert "if 'cizgi' in veri" in py
        assert 'cizgi_ayarla' in py
