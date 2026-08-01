"""Telefondan bağlanma sayfası ve ağ adresi tespiti.

NEDEN?
    Router DHCP ile sürekli yeni IP veriyordu: .103 → .101 → .104. Kullanıcı
    her seferinde eski adresi yazıp "bu siteye ulaşılamıyor" alıyordu; doğru
    adresi bulsa bile sertifika eski IP'ye göre üretildiği için güvenlik
    uyarısı çıkıyordu.

    KALICI ÇÖZÜM: makine adı. Windows kendi adını mDNS ile yayınlar,
    telefonlar `<ad>.local` adresini çözer, IP değişse de ad değişmez.
    Sayfa bunu ÖNCE gösterir ve sertifikanın kapsayıp kapsamadığını
    kontrol eder — kapsamıyorsa kalıcılık işe yaramaz.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app import ag, main


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def temiz_kayit(tmp_path, monkeypatch):
    """ag.json'u geçici dizine al — gerçek dosyayı bozmayalım."""
    monkeypatch.setattr(ag, '_bilgi_dosyasi', lambda: tmp_path / 'ag.json')
    return tmp_path / 'ag.json'


class TestAdresler:
    def test_makine_adi_ONCE_ve_kalici(self, temiz_kayit):
        ag.bilgi_yaz('SERA-PC', ['192.168.1.50'])
        a = ag.adresler()
        assert a[0].deger == 'SERA-PC.local'
        assert a[0].kalici is True
        assert a[1].kalici is False, 'IP kalıcı sayılmamalı'

    def test_sanal_ag_adresleri_elenir(self, temiz_kayit):
        """Docker/WSL adreslerine telefon erişemez; göstermek yanıltır."""
        ag.bilgi_yaz('PC', ['192.168.1.50', '172.23.112.1', '172.17.0.2',
                            '169.254.1.1', '127.0.0.1'])
        ipler = ag.yerel_ipler()
        assert ipler == ['192.168.1.50']

    def test_kayit_yoksa_kendi_tespit_eder(self, temiz_kayit):
        """Dosya yoksa çökmemeli — konteynerin kendi adını verir."""
        assert isinstance(ag.makine_adi(), str)
        assert isinstance(ag.yerel_ipler(), list)

    def test_bozuk_kayit_cokmez(self, temiz_kayit):
        temiz_kayit.write_text('bu json degil {{', encoding='utf-8')
        assert ag.kayitli_bilgi() == {}

    def test_bilgi_yaz_oku_dongusu(self, temiz_kayit):
        ag.bilgi_yaz('X-PC', ['10.0.0.5'])
        k = json.loads(temiz_kayit.read_text(encoding='utf-8'))
        assert k['makine_adi'] == 'X-PC' and k['ipler'] == ['10.0.0.5']
        assert 'zaman' in k, 'bilginin ne zaman yazıldığı da saklanmalı'

    def test_url_https_ve_port(self):
        from app import config
        u = ag.url('PC.local')
        assert u.startswith('https://PC.local:')
        assert str(config.HTTPS_PORT) in u

    def test_sertifika_kapsami_makine_adini_icerir(self, temiz_kayit):
        """Ad kapsanmazsa kalıcı adres her seferinde uyarı verir."""
        ag.bilgi_yaz('SERA-PC', ['192.168.1.50'])
        k = ag.sertifika_kapsami()
        assert 'SERA-PC' in k and 'SERA-PC.local' in k
        assert '192.168.1.50' in k


class TestSayfa:
    def test_sayfa_acilir(self, client):
        r = client.get('/baglan')
        assert r.status_code == 200
        assert 'Telefondan Bağlan' in r.text

    def test_qr_uretiliyor(self, client):
        """QR olmadan kullanıcı uzun https adresini elle yazmak zorunda."""
        pytest.importorskip('segno', reason='QR kütüphanesi kurulu değil')
        r = client.get('/baglan')
        assert '<svg' in r.text, 'QR kodu üretilmemiş'

    def test_qr_svg_gomulu_dis_istek_yok(self):
        """QR satır içi SVG olmalı: sera ağında internet olmayabilir.

        `xmlns` bir ad alanı bildirimidir, ağ isteği DEĞİLDİR — aranan şey
        dışarıdan dosya çeken öğeler.
        """
        pytest.importorskip('segno')
        q = main._qr('https://ornek.local:8443/')
        assert q.startswith('<svg')
        for dis in ('<image', 'href=', 'src=', '<script', '@import'):
            assert dis not in q, f'SVG dışarıdan kaynak çekiyor: {dis}'

    def test_kalici_adres_one_cikarilir(self, client):
        r = client.get('/baglan')
        assert 'Bunu kullanın' in r.text
        assert 'IP değişse bile çalışır' in r.text

    def test_sertifika_uyumsuzlugu_bildirilir(self, client, monkeypatch):
        """Sertifika güncel adresi kapsamıyorsa kullanıcı sebebi görmeli."""
        monkeypatch.setattr(main, '_sertifika_adlari', lambda: {'localhost'})
        r = client.get('/baglan')
        assert 'Sertifika güncel değil' in r.text
        assert 'https_sertifika.py' in r.text

    def test_anasayfadan_baglanti_var(self, client):
        assert '/baglan' in client.get('/').text

    def test_qr_uretilemezse_sayfa_yine_acilir(self, client, monkeypatch):
        monkeypatch.setattr(main, '_qr', lambda v: '')
        assert client.get('/baglan').status_code == 200


class TestSertifikaBetigi:
    def test_makine_adlari_uretilir(self):
        import importlib.util
        import sys
        from pathlib import Path
        kok = Path(__file__).resolve().parent.parent
        s = importlib.util.spec_from_file_location(
            'https_sertifika', kok / 'scripts' / 'https_sertifika.py')
        m = importlib.util.module_from_spec(s)
        sys.modules['https_sertifika'] = m
        s.loader.exec_module(m)

        adlar = m.makine_adlari()
        assert adlar, 'makine adı bulunamadı'
        assert any(a.endswith('.local') for a in adlar), 'mDNS adı eksik'
        assert 'localhost' not in adlar
