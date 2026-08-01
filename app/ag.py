"""Sunucuya hangi adreslerden ulaşılır — tek kaynak.

NEDEN AYRI MODÜL?
    Telefon her seferinde farklı bir adrese bağlanmak zorunda kalıyordu:
    router DHCP ile yeni IP veriyor (.103 → .101 → .104) ve kullanıcı eski
    adresi yazınca "bu siteye ulaşılamıyor" alıyordu. Sertifika da eski
    IP'ye göre üretildiği için, doğru adresi bulsa bile güvenlik uyarısı
    çıkıyordu.

    KALICI ÇÖZÜM: makine ADI. Windows 10+ kendi adını mDNS ile yayınlar;
    telefonlar `<ad>.local` adresini çözer. IP değişse de ad değişmez.

    Bu modül adresleri tek yerden üretir: bağlantı sayfası, başlangıç
    günlüğü ve sertifika üreticisi aynı listeyi kullansın diye.
"""

import socket
from dataclasses import dataclass
from typing import List

# Bağlantı için işe yaramayan adres blokları:
#   127.  → yalnızca bu makine
#   169.254 → APIPA (DHCP alınamamış, ağ yok demektir)
#   172.17-31 / 172.23 → Docker ve WSL sanal ağları; telefon bunlara erişemez
ISE_YARAMAZ_ONEK = ('127.', '169.254.', '172.17.', '172.18.', '172.19.',
                    '172.20.', '172.21.', '172.22.', '172.23.', '172.24.',
                    '172.25.', '172.26.', '172.27.', '172.28.', '172.29.',
                    '172.30.', '172.31.')


@dataclass
class Adres:
    deger: str               # 192.168.1.104 veya LAPTOP-X.local
    tur: str                 # 'ad' | 'ip'
    kalici: bool             # IP değişince geçerliliğini korur mu?

    @property
    def aciklama(self) -> str:
        return ('IP değişse bile çalışır' if self.kalici
                else 'router yeni IP verince değişir')


# Uygulama Docker İÇİNDE çalışır ve orada `gethostname()` konteynerin adını,
# arayüz listesi de konteynerin sanal IP'sini verir — telefonun bağlanacağı
# adresler bunlar DEĞİLDİR. Host'un gerçek kimliği bu dosyaya yazılır
# (scripts/https_sertifika.py çalışırken üretir); storage/ konteynere zaten
# bağlı olduğu için ekstra bağlama gerekmez.
def _bilgi_dosyasi():
    from app import config
    return config.STORAGE_DIR / 'ag.json'


def kayitli_bilgi() -> dict:
    import json
    try:
        p = _bilgi_dosyasi()
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def bilgi_yaz(ad: str, ipler: List[str]) -> None:
    """Host tarafında çağrılır; uygulama bunu okur."""
    import json
    from datetime import datetime, timezone
    p = _bilgi_dosyasi()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        'makine_adi': ad, 'ipler': list(ipler),
        'zaman': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }, ensure_ascii=False, indent=1), encoding='utf-8')


def makine_adi() -> str:
    kayit = kayitli_bilgi().get('makine_adi')
    if kayit:
        return kayit
    try:
        return socket.gethostname()
    except OSError:
        return ''


def yerel_ipler() -> List[str]:
    """Telefonun erişebileceği LAN adresleri."""
    kayit = kayitli_bilgi().get('ipler')
    if kayit:
        return [a for a in kayit if not a.startswith(ISE_YARAMAZ_ONEK)]
    bulunan = set()
    try:
        for bilgi in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            bulunan.add(bilgi[4][0])
    except OSError:
        pass
    try:                                   # dışarı çıkan arayüzün adresi
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        bulunan.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(a for a in bulunan if not a.startswith(ISE_YARAMAZ_ONEK))


def adresler() -> List[Adres]:
    """Bağlanılabilecek adresler — KALICI olan başta.

    Makine adı önce gelir çünkü kullanıcının aradığı şey bu: bir kez
    yazıp bir daha uğraşmamak.
    """
    out = []
    ad = makine_adi()
    if ad:
        # mDNS: Windows 10+ ve çoğu telefon `.local` sonekini çözer
        out.append(Adres(f'{ad}.local', 'ad', True))
    out += [Adres(ip, 'ip', False) for ip in yerel_ipler()]
    return out


def url(adres: str, https: bool = True, port: int = None) -> str:
    from app import config
    if https:
        return f'https://{adres}:{port or config.HTTPS_PORT}/'
    return f'http://{adres}:{port or 8000}/'


def sertifika_kapsami() -> List[str]:
    """Sertifikanın kapsaması gereken adlar.

    Makine adı DA girer: `.local` adresi kapsanmazsa kalıcı adres kullanan
    kullanıcı her seferinde güvenlik uyarısı görür ve kalıcılık işe yaramaz.
    """
    ad = makine_adi()
    adlar = [a for a in (ad, f'{ad}.local' if ad else '') if a]
    return adlar + yerel_ipler()
