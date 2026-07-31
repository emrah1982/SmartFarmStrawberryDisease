"""HTTPS sertifikası üretimi — adres süzme.

NEDEN TEST?
    Adresler `sys.argv[1:]` ile doğrudan alınıyordu. Biri
    `python scripts/https_sertifika.py --help` yazınca `--help` bir SAN
    girdisi oldu (DNS:--help) ve otomatik IP tespiti HİÇ çalışmadı.

    Üretilen sertifika yalnızca `localhost` kapsıyordu. Sonuç: telefondan
    her bağlantıda "Bağlantınız gizli değil" uyarısı — ve sebebi görünmüyordu,
    çünkü betik hata vermeden "✅ Hazır" diyordu. Sessiz hata.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    'https_sertifika', KOK / 'scripts' / 'https_sertifika.py')
sert = importlib.util.module_from_spec(_spec)
sys.modules['https_sertifika'] = sert
_spec.loader.exec_module(sert)


class TestAdresDogrula:
    @pytest.mark.parametrize('bayrak', ['--help', '-h', '--adres', '-v'])
    def test_bayraklar_sertifikaya_girmez(self, bayrak):
        """ASIL HATA: '--help' sertifikaya DNS adı olarak yazılmıştı."""
        gecerli, atilan = sert._adres_dogrula([bayrak])
        assert gecerli == []
        assert bayrak in atilan

    def test_ip_kabul_edilir(self):
        gecerli, atilan = sert._adres_dogrula(['192.168.1.101', '10.0.0.5'])
        assert gecerli == ['192.168.1.101', '10.0.0.5']
        assert atilan == []

    def test_makine_adi_kabul_edilir(self):
        gecerli, _ = sert._adres_dogrula(['cilek-sunucu', 'sera.local'])
        assert gecerli == ['cilek-sunucu', 'sera.local']

    @pytest.mark.parametrize('kotu', ['', '   ', 'boşluklu ad', 'ad/slash',
                                      'a"tırnak', '.baştan-nokta'])
    def test_gecersiz_ad_elenir(self, kotu):
        gecerli, atilan = sert._adres_dogrula([kotu])
        assert gecerli == []
        assert atilan

    def test_gecerli_ve_gecersiz_karisik(self):
        gecerli, atilan = sert._adres_dogrula(['--help', '192.168.1.101', ''])
        assert gecerli == ['192.168.1.101']
        assert len(atilan) == 2


class TestSanUretimi:
    def test_ip_ve_dns_ayrilir(self):
        s = sert._san(['192.168.1.101', 'sera.local'])
        assert 'IP:192.168.1.101' in s
        assert 'DNS:sera.local' in s
        assert 'DNS:localhost' in s

    def test_localhost_her_zaman_var(self):
        """Bilgisayardan localhost ile bağlanmak uyarısız çalışmalı."""
        assert 'DNS:localhost' in sert._san([])


class TestYerelIpler:
    def test_en_az_loopback_doner(self):
        assert '127.0.0.1' in sert.yerel_ipler()

    def test_hepsi_gecerli_ip(self):
        import ipaddress
        for a in sert.yerel_ipler():
            ipaddress.ip_address(a)          # ValueError atmamalı


class TestUretilmisSertifika:
    """Depodaki sertifika gerçekten bu makinenin adresini kapsıyor mu?"""

    @pytest.fixture
    def san(self):
        import base64
        import ssl
        import tempfile
        pem_yolu = KOK / 'certs' / 'sunucu.crt'
        if not pem_yolu.exists():
            pytest.skip('sertifika üretilmemiş')
        pem = pem_yolu.read_text(encoding='utf-8', errors='ignore')
        gecici = Path(tempfile.mkdtemp()) / 'c.pem'
        gecici.write_text(pem, encoding='utf-8')
        try:
            return dict(ssl._ssl._test_decode_cert(str(gecici))).get(
                'subjectAltName', ())
        except Exception as e:
            pytest.skip(f'sertifika çözülemedi: {e}')

    def test_bayrak_sizmamis(self, san):
        for tur, deger in san:
            assert not deger.startswith('-'), f'SAN içinde bayrak: {deger}'

    def test_en_az_bir_ip_kapsiyor(self, san):
        """Yalnızca localhost kapsayan sertifika telefonda hep uyarı verir."""
        assert any(t == 'IP Address' for t, _ in san), (
            f'sertifika hiçbir IP kapsamıyor: {san} — '
            'telefondan bağlanınca "Bağlantınız gizli değil" çıkar')

    def test_localhost_kapsiyor(self, san):
        assert ('DNS', 'localhost') in san
