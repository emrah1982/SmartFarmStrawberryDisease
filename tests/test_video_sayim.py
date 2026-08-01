"""Videoda SAYIM — kutu sayısı değil, BENZERSİZ nesne sayısı.

NEDEN TEST?
    Video işlenirken her örneklenen karenin kutuları biriktiriliyor ve
    kareler arası eşleştirme yoktu. Ölçüldü:

        4 meyveli SABİT sahne, 4 kare örneklendi → 11 kutu

    Kullanıcı bunu "11 hastalıklı meyve" diye okursa yanlış tarımsal karar
    verir: gereksiz ilaçlama, yanlış hasat planı, boşuna imha.

    Artık app/takip.py kareler arasında eşleştirme yapıyor; arayüz kutu
    sayısını değil benzersiz nesne sayısını öne çıkarıyor.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.detector import Kutu, Sonuc
from tests.test_app import SahteDetector


class VideoDetector(SahteDetector):
    """3 kare × 4 kutu = 12 kutu, ama gerçekte 4 nesne."""

    def video(self, kaynak, cikti):
        from pathlib import Path
        Path(cikti).write_bytes(b'sahte-jpeg')
        kutular = []
        for kare in (0, 15, 30):
            for i in range(4):
                kutular.append(Kutu(9, 'strawberry_ripe', 0.8,
                                    0.2 + i * 0.15, 0.5, 0.1, 0.1, kare=kare))
        return Sonuc(kutular=kutular, sonuc_yolu=str(cikti), islenen_kare=3,
                     sure_ms=10, kare_basina_en_cok=4, benzersiz_sayi=4,
                     takip_izi={'benzersiz': 4, 'sinif': {'strawberry_ripe': 4},
                                'fps': 30.0})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, 'detector', VideoDetector())
    with TestClient(main.app) as c:
        yield c


def _video_yukle(client):
    return client.post('/analiz/dosya',
                       files={'dosyalar': ('a.mp4', b'x', 'video/mp4')},
                       follow_redirects=True)


class TestBenzersizSayim:
    def test_benzersiz_sayi_one_cikar(self, client):
        r = _video_yukle(client)
        assert r.status_code == 200
        assert 'Kaç ayrı nesne var' in r.text
        assert '4 ayrı nesne' in r.text

    def test_kutu_sayisiyla_farki_aciklanir(self, client):
        """Kullanıcı 12 kutu görüp şaşırmamalı; sebebi yazılı olmalı."""
        r = _video_yukle(client)
        assert '12 kutu' in r.text
        assert 'aynı nesnenin farklı karelerde' in r.text

    def test_takibin_siniri_soyleniyor(self, client):
        """Takip kesin değil; hızlı kamerada sapabileceği yazılı olmalı."""
        r = _video_yukle(client)
        assert 'sapabilir' in r.text

    def test_fotografta_kart_YOK(self, client):
        """Fotoğrafta her nesne bir kez sayılır; kart kafa karıştırır."""
        r = client.post('/analiz/dosya',
                        files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                        follow_redirects=True)
        assert 'Kaç ayrı nesne var' not in r.text

    def test_veritabanina_yaziliyor(self, client):
        from app.database import Analiz, SessionLocal
        _video_yukle(client)
        db = SessionLocal()
        try:
            a = db.query(Analiz).order_by(Analiz.id.desc()).first()
            assert a.tespit_sayisi == 12, 'ham kutu sayısı korunmalı'
            assert a.benzersiz_sayi == 4, 'benzersiz sayı kaydedilmeli'
            assert a.kare_basina_en_cok == 4
        finally:
            db.close()


class TestGercekVideoAkisi:
    def test_sonuc_alanlari_var(self):
        s = Sonuc()
        assert s.benzersiz_sayi == 0
        assert s.takip_izi == {}
        assert s.kare_basina_en_cok == 0

    def test_video_kodu_takipci_kullaniyor(self):
        from pathlib import Path
        kaynak = (Path(__file__).resolve().parent.parent
                  / 'app' / 'detector.py').read_text(encoding='utf-8')
        assert 'takipci.ekle(idx, kare_kutulari)' in kaynak
        assert 'benzersiz_sayi=benzersiz' in kaynak

    def test_ornekleme_fpse_gore(self):
        """Sabit kare adımı yerine süreye göre örnekleme."""
        from pathlib import Path
        kaynak = (Path(__file__).resolve().parent.parent
                  / 'app' / 'detector.py').read_text(encoding='utf-8')
        assert 'ornekleme_adimi(fps' in kaynak
        assert 'idx % adim == 0' in kaynak
