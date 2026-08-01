"""Videoda SAYIM — kutu sayısı nesne sayısı değildir.

NEDEN TEST?
    Video işlenirken her örneklenen karenin kutuları birikitiriliyor,
    kareler arası eşleştirme (takip) yapılmıyor. Sonuç: aynı meyve her
    karede yeniden sayılıyor. Ölçüldü:

        4 meyveli SABİT sahne, 4 kare örneklendi → 11 kutu

    Kullanıcı bunu "11 hastalıklı meyve" diye okursa yanlış tarımsal karar
    verir (ilaçlama, hasat, imha). Sayı düzeltilemiyorsa bile NE ANLAMA
    GELDİĞİ söylenmeli — sessizce yanlış sayı vermek en kötüsü.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.detector import Kutu, Sonuc
from tests.test_app import SahteDetector


class VideoDetector(SahteDetector):
    """Videoda birikmiş kutuları taklit eder: 3 kare × 4 nesne."""

    def video(self, kaynak, cikti):
        from pathlib import Path
        Path(cikti).write_bytes(b'sahte-jpeg')
        kutular = []
        for kare in (0, 15, 30):
            for i in range(4):
                kutular.append(Kutu(9, 'strawberry_ripe', 0.8,
                                    0.2 + i * 0.15, 0.5, 0.1, 0.1, kare=kare))
        return Sonuc(kutular=kutular, sonuc_yolu=str(cikti), islenen_kare=3,
                     sure_ms=10, kare_basina_en_cok=4)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, 'detector', VideoDetector())
    with TestClient(main.app) as c:
        yield c


class TestSayimUyarisi:
    def test_video_kaydinda_uyari_gosterilir(self, client):
        r = client.post('/analiz/dosya',
                        files={'dosyalar': ('a.mp4', b'x', 'video/mp4')},
                        follow_redirects=True)
        assert r.status_code == 200
        assert 'kaç nesne var demek değil' in r.text

    def test_alt_sinir_yazili(self, client):
        """Kullanıcı en azından güvenilir bir alt sınır görmeli."""
        r = client.post('/analiz/dosya',
                        files={'dosyalar': ('a.mp4', b'x', 'video/mp4')},
                        follow_redirects=True)
        assert 'en az o kadar' in r.text
        assert '>4<' in r.text, 'kare başına en çok değeri gösterilmeli'

    def test_kutu_sayisi_da_yazili(self, client):
        r = client.post('/analiz/dosya',
                        files={'dosyalar': ('a.mp4', b'x', 'video/mp4')},
                        follow_redirects=True)
        assert '12 kutu' in r.text, '3 kare × 4 nesne = 12 kutu'

    def test_fotografta_uyari_YOK(self, client):
        """Fotoğrafta her nesne bir kez sayılır; uyarı kafa karıştırır."""
        r = client.post('/analiz/dosya',
                        files={'dosyalar': ('a.jpg', b'x', 'image/jpeg')},
                        follow_redirects=True)
        assert 'kaç nesne var demek değil' not in r.text

    def test_deger_veritabanina_yaziliyor(self, client):
        from app.database import Analiz, SessionLocal
        client.post('/analiz/dosya', files={'dosyalar': ('a.mp4', b'x', 'video/mp4')},
                    follow_redirects=True)
        db = SessionLocal()
        try:
            a = db.query(Analiz).order_by(Analiz.id.desc()).first()
            assert a.kare_basina_en_cok == 4
            assert a.tespit_sayisi == 12
        finally:
            db.close()


class TestGercekVideoAkisi:
    """detector.video() gerçekten alt sınırı hesaplıyor mu?"""

    def test_kare_basina_en_cok_alani_var(self):
        s = Sonuc()
        assert hasattr(s, 'kare_basina_en_cok')
        assert s.kare_basina_en_cok == 0

    def test_video_kodu_uyari_metni_uretiyor(self):
        """Not metni detector içinde kuruluyor — sessizce kaybolmamalı."""
        from pathlib import Path
        kaynak = (Path(__file__).resolve().parent.parent
                  / 'app' / 'detector.py').read_text(encoding='utf-8')
        assert 'her karede yeniden sayıldı' in kaynak
        assert 'kare_basina_en_cok=en_cok' in kaynak
