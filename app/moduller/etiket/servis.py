"""Aday etiket inceleme — çekirdek mantık.

KATMAN KURALI: bu dosya fastapi / sqlalchemy / jinja2 / app.main
IMPORT ETMEZ. Yalnızca dosya sistemi ve saf veri. Böylece rotalardan
bağımsız test edilir ve başka bir arayüze (CLI, betik) takılabilir.

NE İŞE YARAR?
    Otomatik ön-etiketleme (Grounding DINO / DINOv2 / CLIP) ADAY kutular
    üretir. Bu kutular insan onayından geçmeden eğitime verilmez —
    projede tekrar tekrar yakalanan "sessiz hata" deseni budur: model
    hata vermez, yanlışı öğrenir.

    Bu modül o onay adımını arayüze taşır: kutuyu gör, düzelt, onayla.

ONAY DURUMU NEREDE SAKLANIR?
    Paketin kendi klasöründe `inceleme.json`. Veritabanında DEĞİL, çünkü:
      - dataset kendi kendine yeter, Drive'a kopyalanınca durum da gider
      - şema göçü gerekmez
      - iki farklı kurulum aynı paketi paylaşabilir

ÜÇ AYRI ETİKET KLASÖRÜ, KARIŞTIRILMAMALI
    labels_aday/     ham otomatik çıktı — ASLA doğrudan eğitime girmez
    labels/          insan onayından geçmiş — eğitime giren budur
    labels_hastalik/ ayrı katman (hastalıklı organ), organ katmanına karışmaz

    Düzenleme `labels_aday/` üzerinde yapılır; onaylanan kare
    `labels/` altına yazılır. Böylece "ham otomatik ne demişti"
    bilgisi kaybolmaz ve gerekirse karşılaştırılır.
"""

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DURUM_DOSYASI = 'inceleme.json'

# Aday etiketlerin bulunabileceği klasör adları, ÖNCELİK SIRASIYLA.
ADAY_KLASORLERI = ('labels_aday', 'labels_organ')
ONAYLI_KLASOR = 'labels'
HASTALIK_KLASOR = 'labels_hastalik'

# Kutunun kısa kenarı bunun altındaysa uyarılır: YOLO tespit ızgarası
# 8 piksel adımlıdır, altındaki nesne zaten öğrenilemez.
EN_KUCUK_KENAR_PX = 16


# ─────────────────────────────────────────────────────────────────────────
# Güvenlik — dışarıdan gelen ad ile dosya yolu kurmak
# ─────────────────────────────────────────────────────────────────────────

_GUVENLI_AD = re.compile(r'^[A-Za-z0-9._ ()\-]+$')


def guvenli_mi(ad: str) -> bool:
    """Bu ad bir dosya adı olarak kullanılabilir mi?

    Kare adı ve paket adı URL'den gelir. Denetlenmezse `../../etc/passwd`
    gibi bir değer paket dışına yazma/okuma yapar. Beyaz liste kullanılır:
    neyin YASAK olduğunu saymak yerine neyin SERBEST olduğunu sayarız —
    kara liste her zaman eksik kalır.
    """
    if not ad or ad in ('.', '..'):
        return False
    if '/' in ad or '\\' in ad or '\x00' in ad:
        return False
    return bool(_GUVENLI_AD.match(ad))


def _icinde_mi(yol: Path, kok: Path) -> bool:
    """yol gerçekten kok'un altında mı? (sembolik bağ dahil çözülür)"""
    try:
        yol.resolve().relative_to(kok.resolve())
        return True
    except (ValueError, OSError):
        return False


# ─────────────────────────────────────────────────────────────────────────
# Veri yapıları
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Kutu:
    sinif: int
    cx: float
    cy: float
    w: float
    h: float
    guven: float = 0.0

    def gecerli_mi(self, sinif_sayisi: int) -> bool:
        if not 0 <= self.sinif < sinif_sayisi:
            return False
        if not (self.w > 0 and self.h > 0):
            return False
        return all(0.0 <= v <= 1.0 for v in (self.cx, self.cy, self.w, self.h))

    def satir(self) -> str:
        return '%d %.6f %.6f %.6f %.6f' % (self.sinif, self.cx, self.cy,
                                           self.w, self.h)


@dataclass
class Kare:
    ad: str                      # dosya adı (uzantısız değil, tam)
    onayli: bool = False
    kutu_sayisi: int = 0
    en_dusuk_guven: float = 1.0
    hastalik_kutusu: int = 0


@dataclass
class Paket:
    urun: str
    ad: str
    yol: Path
    siniflar: List[str] = field(default_factory=list)
    aday_klasor: str = ''
    goruntu_sayisi: int = 0
    kutu_sayisi: int = 0
    onayli_sayisi: int = 0
    hastalik_var: bool = False

    @property
    def anahtar(self) -> str:
        return f'{self.urun}/{self.ad}'

    @property
    def ilerleme(self) -> int:
        if not self.goruntu_sayisi:
            return 0
        return int(round(100 * self.onayli_sayisi / self.goruntu_sayisi))


# ─────────────────────────────────────────────────────────────────────────
# Etiket dosyası okuma/yazma
# ─────────────────────────────────────────────────────────────────────────

def satir_coz(parcalar) -> Optional[Kutu]:
    """YOLO satırı → Kutu. Poligon biçimi de kabul edilir.

    Poligon (segmentasyon) satırı 7+ alan taşır. Ultralytics tespit
    eğitiminde onu kutuya çevirir; burada da aynısını yaparız, yoksa
    2. noktanın koordinatları genişlik/yükseklik sanılır.
    Bkz. docs/HATA-YONETIMI.md § 2.6b
    """
    if len(parcalar) < 5:
        return None
    try:
        sinif = int(parcalar[0])
        sayilar = [float(x) for x in parcalar[1:]]
    except ValueError:
        return None
    if len(sayilar) == 4:
        cx, cy, w, h = sayilar
    elif len(sayilar) >= 6 and len(sayilar) % 2 == 0:
        xs, ys = sayilar[0::2], sayilar[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    else:
        return None
    if w <= 0 or h <= 0:
        return None
    return Kutu(sinif, cx, cy, w, h)


def kutulari_oku(yol: Path) -> List[Kutu]:
    if not yol.exists():
        return []
    out = []
    for satir in yol.read_text(encoding='utf-8', errors='ignore').splitlines():
        k = satir_coz(satir.split())
        if k is not None:
            out.append(k)
    return out


def kutulari_yaz(yol: Path, kutular: List[Kutu]):
    """Etiket dosyasını ATOMİK yazar.

    Doğrudan yazarken süreç kesilirse dosya yarım kalır ve o kare sessizce
    bozulur. Geçici dosyaya yazıp taşımak, ya eski ya yeni içeriği garanti
    eder — arada bir hal olmaz.
    """
    yol.parent.mkdir(parents=True, exist_ok=True)
    gecici = yol.with_suffix(yol.suffix + '.tmp')
    metin = '\n'.join(k.satir() for k in kutular)
    gecici.write_text(metin + ('\n' if metin else ''), encoding='utf-8')
    os.replace(gecici, yol)


# ─────────────────────────────────────────────────────────────────────────
# Paket keşfi
# ─────────────────────────────────────────────────────────────────────────

def _siniflari_oku(paket_yolu: Path) -> List[str]:
    """data.yaml'daki sınıf sırası — kutulardaki ID'lerin anlamı budur."""
    y = paket_yolu / 'data.yaml'
    if not y.exists():
        return []
    try:
        import yaml
        d = yaml.safe_load(y.read_text(encoding='utf-8')) or {}
    except Exception as e:
        logger.error(f'{y} okunamadı: {e}')
        return []
    n = d.get('names')
    if isinstance(n, dict):
        return [n[k] for k in sorted(n, key=lambda x: int(x))]
    return list(n or [])


def _aday_klasoru(paket_yolu: Path) -> str:
    for ad in ADAY_KLASORLERI:
        if (paket_yolu / ad).is_dir():
            return ad
    return ''


def paketleri_bul(dataset_kok: Path) -> List[Paket]:
    """İnceleme bekleyen paketleri bulur.

    Bir klasör paket sayılır: images/ ve (labels_aday/ veya labels_organ/)
    içeriyorsa. Ürün adı bir üst klasördür; kapsamsız kökte duran paketin
    ürünü boş kalır (ortak varlıklar).
    """
    out = []
    if not dataset_kok.is_dir():
        return out
    for urun_dizini in sorted(p for p in dataset_kok.iterdir() if p.is_dir()):
        adaylar = [urun_dizini] + sorted(
            p for p in urun_dizini.iterdir() if p.is_dir())
        for p in adaylar:
            aday = _aday_klasoru(p)
            if not aday or not (p / 'images').is_dir():
                continue
            urun = urun_dizini.name if p is not urun_dizini else ''
            out.append(_paket_kur(urun, p, aday))
    return out


def _paket_kur(urun: str, yol: Path, aday: str) -> Paket:
    goruntuler = _goruntuler(yol)
    durum = durum_oku(yol)
    kutu_toplam = 0
    for g in goruntuler:
        kutu_toplam += len(kutulari_oku(yol / aday / (g.stem + '.txt')))
    return Paket(
        urun=urun, ad=yol.name, yol=yol,
        siniflar=_siniflari_oku(yol), aday_klasor=aday,
        goruntu_sayisi=len(goruntuler), kutu_sayisi=kutu_toplam,
        onayli_sayisi=sum(1 for a in durum.get('onayli', []) if a),
        hastalik_var=(yol / HASTALIK_KLASOR).is_dir(),
    )


def paket_bul(dataset_kok: Path, urun: str, ad: str) -> Optional[Paket]:
    """Tek paketi ada göre çözer — yol enjeksiyonuna kapalı."""
    if not guvenli_mi(ad) or (urun and not guvenli_mi(urun)):
        return None
    yol = (dataset_kok / urun / ad) if urun else (dataset_kok / ad)
    if not _icinde_mi(yol, dataset_kok) or not yol.is_dir():
        return None
    aday = _aday_klasoru(yol)
    if not aday or not (yol / 'images').is_dir():
        return None
    return _paket_kur(urun, yol, aday)


def _goruntuler(yol: Path) -> List[Path]:
    d = yol / 'images'
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in GORUNTU_UZANTI)


# ─────────────────────────────────────────────────────────────────────────
# İnceleme durumu (paket klasöründe JSON)
# ─────────────────────────────────────────────────────────────────────────

def durum_oku(paket_yolu: Path) -> dict:
    p = paket_yolu / DURUM_DOSYASI
    if not p.exists():
        return {'onayli': {}}
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as e:
        # Bozuk durum dosyası inceleme akışını KESMEMELİ; en kötü ihtimalle
        # onay bilgisi kaybolur, etiketler yerinde durur.
        logger.error(f'{p} okunamadı, sıfırdan başlanıyor: {e}')
        return {'onayli': {}}
    d.setdefault('onayli', {})
    return d


def durum_yaz(paket_yolu: Path, durum: dict):
    p = paket_yolu / DURUM_DOSYASI
    gecici = p.with_suffix('.json.tmp')
    gecici.write_text(json.dumps(durum, ensure_ascii=False, indent=1),
                      encoding='utf-8')
    os.replace(gecici, p)


def _guven_haritasi(paket_yolu: Path) -> Dict[str, float]:
    """INCELEME.csv varsa kare başına EN DÜŞÜK güveni okur.

    Sıralama için: hata en düşük güvenli karelerde birikir, inceleme
    oradan başlamalı. CSV yoksa sıralama dosya adına düşer.
    """
    p = paket_yolu / 'INCELEME.csv'
    if not p.exists():
        return {}
    out: Dict[str, float] = {}
    try:
        import csv
        with p.open(encoding='utf-8') as f:
            for satir in csv.DictReader(f):
                ad = satir.get('goruntu')
                try:
                    g = float(satir.get('guven', 1.0))
                except (TypeError, ValueError):
                    continue
                if ad:
                    out[ad] = min(out.get(ad, 1.0), g)
    except (OSError, csv.Error) as e:
        logger.error(f'{p} okunamadı: {e}')
    return out


def kareler(paket: Paket, sirala: str = 'guven') -> List[Kare]:
    """Paketin kareleri + inceleme durumu.

    sirala='guven'  → en düşük güvenli önce (hata orada birikir)
    sirala='ad'     → dosya adına göre
    """
    durum = durum_oku(paket.yol)
    onayli = durum.get('onayli', {})
    guven = _guven_haritasi(paket.yol)
    out = []
    for g in _goruntuler(paket.yol):
        kutular = kutulari_oku(paket.yol / paket.aday_klasor / (g.stem + '.txt'))
        h = 0
        if paket.hastalik_var:
            h = len(kutulari_oku(paket.yol / HASTALIK_KLASOR / (g.stem + '.txt')))
        out.append(Kare(
            ad=g.name,
            onayli=bool(onayli.get(g.name)),
            kutu_sayisi=len(kutular),
            en_dusuk_guven=guven.get(g.name, 1.0),
            hastalik_kutusu=h,
        ))
    if sirala == 'guven':
        out.sort(key=lambda k: (k.onayli, k.en_dusuk_guven, k.ad))
    else:
        out.sort(key=lambda k: k.ad)
    return out


def kare_yolu(paket: Paket, ad: str) -> Optional[Path]:
    """Kare adını dosya yoluna çevirir — paket dışına çıkmaya kapalı."""
    if not guvenli_mi(ad):
        return None
    p = paket.yol / 'images' / ad
    if not _icinde_mi(p, paket.yol / 'images') or not p.is_file():
        return None
    return p


def kare_kutulari(paket: Paket, ad: str, katman: str = 'organ') -> List[Kutu]:
    if kare_yolu(paket, ad) is None:
        return []
    klasor = paket.aday_klasor if katman == 'organ' else HASTALIK_KLASOR
    return kutulari_oku(paket.yol / klasor / (Path(ad).stem + '.txt'))


def kare_kaydet(paket: Paket, ad: str, kutular: List[Kutu],
                katman: str = 'organ') -> tuple:
    """Düzeltilmiş kutuları yazar. Dönen: (basarili, mesaj).

    GEÇERSİZ KUTU SESSİZCE ATILMAZ — kaç tanesinin neden elendiği
    çağırana bildirilir. Sessiz eleme, kullanıcının çizdiği kutunun
    kaybolduğunu fark etmemesine yol açar.
    """
    if kare_yolu(paket, ad) is None:
        return False, 'Kare bulunamadı'
    n = len(paket.siniflar)
    gecerli = [k for k in kutular if k.gecerli_mi(n)]
    elenen = len(kutular) - len(gecerli)
    klasor = paket.aday_klasor if katman == 'organ' else HASTALIK_KLASOR
    try:
        kutulari_yaz(paket.yol / klasor / (Path(ad).stem + '.txt'), gecerli)
    except OSError as e:
        logger.error(f'{ad} yazılamadı: {e}')
        return False, f'Yazılamadı: {e}'
    if elenen:
        return True, f'{len(gecerli)} kutu kaydedildi, {elenen} geçersiz kutu atıldı'
    return True, f'{len(gecerli)} kutu kaydedildi'


def onay_ver(paket: Paket, ad: str, onayli: bool = True) -> bool:
    if kare_yolu(paket, ad) is None:
        return False
    durum = durum_oku(paket.yol)
    if onayli:
        durum['onayli'][ad] = True
    else:
        durum['onayli'].pop(ad, None)
    durum_yaz(paket.yol, durum)
    return True


def sonraki_kare(paket: Paket, ad: str, sirala: str = 'guven') -> Optional[str]:
    """Sıradaki İNCELENMEMİŞ kare. Kalmadıysa None."""
    liste = kareler(paket, sirala)
    adlar = [k.ad for k in liste]
    if ad in adlar:
        kalan = liste[adlar.index(ad) + 1:]
    else:
        kalan = liste
    for k in kalan:
        if not k.onayli:
            return k.ad
    for k in liste:                      # başa sar
        if not k.onayli and k.ad != ad:
            return k.ad
    return None


# ─────────────────────────────────────────────────────────────────────────
# Dışa aktarma
# ─────────────────────────────────────────────────────────────────────────

def disa_aktar(paket: Paket, yalniz_onayli: bool = True) -> dict:
    """Onaylanan kareleri `labels/` altına yazar.

    ADAY KLASÖRÜ SİLİNMEZ: "ham otomatik ne demişti" bilgisi kalır,
    gerekirse karşılaştırılır. Eğitim `labels/` klasörünü kullanır.
    """
    durum = durum_oku(paket.yol)
    onayli = durum.get('onayli', {})
    hedef = paket.yol / ONAYLI_KLASOR
    hedef.mkdir(parents=True, exist_ok=True)
    yazilan = atlanan = bos = 0
    for g in _goruntuler(paket.yol):
        if yalniz_onayli and not onayli.get(g.name):
            atlanan += 1
            continue
        kutular = kutulari_oku(paket.yol / paket.aday_klasor / (g.stem + '.txt'))
        kutulari_yaz(hedef / (g.stem + '.txt'), kutular)
        yazilan += 1
        if not kutular:
            bos += 1
    return {'yazilan': yazilan, 'atlanan': atlanan, 'bos': bos,
            'hedef': str(hedef)}


def kalite_denetimi(paket: Paket) -> List[dict]:
    """Eğitime vermeden önceki uyarılar.

    Ölçülen iki gerçek hatayı arar (docs/HATA-YONETIMI.md § 2.6c):
      1. Bir sınıfın bütün kutuları AYNI değerde → sabit damga, bilgi yok
      2. Tam kadrajı kaplayan kutular → görüntü düzeyi etiketin kutuya
         çevrilmiş hali; ROI kırpmayı işlevsizleştirir
    """
    from collections import Counter, defaultdict
    benzersiz = defaultdict(set)
    say = Counter()
    tam_kadraj = 0
    toplam = 0
    for g in _goruntuler(paket.yol):
        for k in kutulari_oku(paket.yol / paket.aday_klasor / (g.stem + '.txt')):
            toplam += 1
            ad = (paket.siniflar[k.sinif] if k.sinif < len(paket.siniflar)
                  else str(k.sinif))
            say[ad] += 1
            benzersiz[ad].add((round(k.cx, 4), round(k.cy, 4),
                               round(k.w, 4), round(k.h, 4)))
            if k.w > 0.99 and k.h > 0.99:
                tam_kadraj += 1

    uyarilar = []
    for ad, n in say.items():
        u = len(benzersiz[ad])
        if n >= 10 and u < max(2, n * 0.5):
            uyarilar.append({
                'tur': 'sabit_kutu', 'sinif': ad,
                'mesaj': f'`{ad}`: {n} kutu ama yalnızca {u} benzersiz değer. '
                         'Bu sınıf konum bilgisi taşımıyor olabilir — '
                         'kutular sabit damga mı, kontrol edin.',
            })
    if toplam and tam_kadraj / toplam > 0.01:
        uyarilar.append({
            'tur': 'tam_kadraj', 'sinif': '',
            'mesaj': f'{tam_kadraj} kutu tüm kareyi kaplıyor '
                     f'(%{100 * tam_kadraj / toplam:.1f}). Bunlar tespit değil, '
                     'görüntü düzeyi etiketin kutuya çevrilmiş hali olabilir; '
                     'ROI kırpmayı işlevsiz bırakır.',
        })
    return uyarilar
