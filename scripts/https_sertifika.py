"""Yerel ağ için kendinden imzalı HTTPS sertifikası üretir.

NEDEN GEREKLİ?
    Tarayıcılar kamerayı (getUserMedia) yalnızca "güvenli bağlam"da verir:
    https:// veya localhost. Telefondan http://192.168.x.x:8000 ile
    bağlandığınızda CANLI KAMERA AÇILMAZ — tarayıcı engeller, uygulamanın
    yapabileceği bir şey yoktur. Çözüm sunucuyu sertifikayla başlatmaktır.

    Sertifika kendinden imzalı olduğu için telefon ilk açılışta
    "Bağlantınız gizli değil" uyarısı gösterir; "Gelişmiş → Yine de devam et"
    denir. Kendi ağınızdaki kendi sunucunuz olduğu için bu güvenlidir; internete
    açık bir kurulumda gerçek sertifika (Let's Encrypt) kullanın.

KULLANIM
    python scripts/https_sertifika.py            # yerel IP'ler otomatik bulunur
    python scripts/https_sertifika.py 192.168.1.42 cilek.local

    Ardından:
    - Yerelde : python -m app.main
    - Docker  : docker compose up -d      (certs/ klasörü konteynere bağlıdır)
"""

import ipaddress
import socket
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
CERT_DIR = KOK / 'certs'
CRT = CERT_DIR / 'sunucu.crt'
KEY = CERT_DIR / 'sunucu.key'


def yerel_ipler():
    """Bu makinenin yerel ağ adreslerini bulur (telefon bu adrese bağlanacak)."""
    adresler = {'127.0.0.1'}
    try:
        for bilgi in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            adresler.add(bilgi[4][0])
    except OSError:
        pass
    try:                                   # dışarı çıkan arayüzün adresi
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        adresler.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(adresler)


def _san(adlar):
    """subjectAltName listesi — sertifika hangi adreslerde geçerli olacak."""
    parcalar = ['DNS:localhost']
    for a in adlar:
        try:
            ipaddress.ip_address(a)
            parcalar.append(f'IP:{a}')
        except ValueError:
            parcalar.append(f'DNS:{a}')
    return ','.join(parcalar)


def openssl_ile(adlar) -> bool:
    """openssl varsa onunla üret (Git for Windows ile birlikte gelir)."""
    try:
        subprocess.run(['openssl', 'version'], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False

    komut = [
        'openssl', 'req', '-x509', '-nodes', '-newkey', 'rsa:2048',
        '-keyout', str(KEY), '-out', str(CRT), '-days', '825',
        '-subj', '/CN=Cilek Tespit Yerel Sunucu',
        '-addext', f'subjectAltName={_san(adlar)}',
        # CA:true → sertifika telefona "güvenilir kök" olarak kurulabilir.
        # Kurulduktan sonra tarayıcı hiç uyarı vermez.
        '-addext', 'basicConstraints=critical,CA:true',
        '-addext', 'keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign',
    ]
    sonuc = subprocess.run(komut, capture_output=True, text=True)
    if sonuc.returncode != 0:
        print(sonuc.stderr.strip()[:500])
        return False
    return True


def cryptography_ile(adlar) -> bool:
    """openssl yoksa saf Python ile üret (cryptography paketi gerekir)."""
    try:
        from datetime import datetime, timedelta, timezone

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return False

    anahtar = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ad = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                       'Cilek Tespit Yerel Sunucu')])
    alt = []
    for a in adlar:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(a)))
        except ValueError:
            alt.append(x509.DNSName(a))
    alt.append(x509.DNSName('localhost'))

    simdi = datetime.now(timezone.utc)
    sertifika = (x509.CertificateBuilder()
                 .subject_name(ad).issuer_name(ad)
                 .public_key(anahtar.public_key())
                 .serial_number(x509.random_serial_number())
                 .not_valid_before(simdi - timedelta(days=1))
                 .not_valid_after(simdi + timedelta(days=825))
                 .add_extension(x509.SubjectAlternativeName(alt), critical=False)
                 # CA:true → telefona güvenilir kök olarak kurulabilsin
                 .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                                critical=True)
                 .add_extension(x509.KeyUsage(
                     digital_signature=True, key_encipherment=True, key_cert_sign=True,
                     content_commitment=False, data_encipherment=False,
                     key_agreement=False, crl_sign=False,
                     encipher_only=False, decipher_only=False), critical=True)
                 .sign(anahtar, hashes.SHA256()))

    KEY.write_bytes(anahtar.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    CRT.write_bytes(sertifika.public_bytes(serialization.Encoding.PEM))
    return True


def main():
    adlar = sys.argv[1:] or yerel_ipler()
    CERT_DIR.mkdir(exist_ok=True)

    print('Sertifika şu adresler için üretiliyor:', ', '.join(adlar))
    if not (openssl_ile(adlar) or cryptography_ile(adlar)):
        print('\n❌ Sertifika üretilemedi.\n'
              '   openssl bulunamadı ve cryptography paketi kurulu değil.\n'
              '   Çözüm:  pip install cryptography   → betiği tekrar çalıştırın.')
        return 1

    print(f'\n✅ Hazır:\n   {CRT}\n   {KEY}')
    print('\nBaşlatma:')
    print('   Yerel : python -m app.main')
    print('   Docker: docker compose up -d --build')
    print('\n💻 Bilgisayarda sertifikaya GEREK YOK: http://localhost:8000/canli')
    print('   (localhost tarayıcılarca zaten güvenli bağlam sayılır, kamera açılır)')
    print(f'\n📱 Telefondan: https://{adlar[-1]}:8443/canli')
    print('   "Bağlantınız gizli değil" uyarısı çıkacak. İki seçenek:')
    print('     1) Hızlı  : Gelişmiş → "Yine de devam et"  (her tarayıcıda bir kez)')
    print(f'     2) Kalıcı : https://{adlar[-1]}:8443/canli/sertifika adresini açıp')
    print('                 sertifikayı kurun — sonrasında hiç uyarı çıkmaz.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
