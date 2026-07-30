"""Drive'daki eğitim koşularını izler: ilerleme, kesinti, ölü zaman, ETA.

NEDEN BETİK?
    Colab oturumu düzensiz aralıklarla ölüyor. Eğitim döngüsünün kendisinde
    duraklama YOK — ölçüldü, epoch süreleri %5'ten az sapıyor. Pahalı olan
    şey oturumun ölmesi değil, ÖLDÜĞÜNÜN FARK EDİLMEMESİ:

        organ_detection-2: epoch 190'a 23:55'te geldi, 195'te öldü,
        08:35'e kadar öyle kaldı. Son 5 epoch 15 dakika sürdü.
        Kayıp: 8,5 saat GPU zamanı, tamamen boşa.

    Bu betik Drive'a bakarak koşunun canlı olup olmadığını söyler. Colab'e
    girmeye gerek yok; Drive masaüstü uygulaması bağlıysa yerelden çalışır.

KULLANIM
    python scripts/egitim_izle.py                      # tüm koşuların özeti
    python scripts/egitim_izle.py organ_detection-2    # tek koşu, ayrıntılı
    python scripts/egitim_izle.py --bekle              # ölünce haber ver
    python scripts/egitim_izle.py --drive "G:/Drive'ım/SmartFarmStrawberryDisease"

NASIL ANLAR?
    Canlılık  : results.csv'nin son yazım zamanı, epoch süresinin 2,5 katından
                yeniyse koşu çalışıyor demektir.
    Kesinti   : results.csv'deki `time` kolonu her YENİ SÜREÇTE sıfırlanır.
                Değerin geriye gitmesi, o epoch'ta yeniden başlatıldığını gösterir.
    Ölü zaman : Ardışık epoch checkpoint'lerinin dosya zamanları arasındaki
                fark, beklenenden fazlaysa aradaki süre kayıptır.
"""

import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path

VARSAYILAN_DRIVE = [
    "G:/Drive'ım/SmartFarmStrawberryDisease",
    'G:/My Drive/SmartFarmStrawberryDisease',
    '/content/drive/MyDrive/SmartFarmStrawberryDisease',
]


def drive_bul(verilen=None) -> Path:
    if verilen:
        return Path(verilen)
    ortam = os.environ.get('DRIVE_KOK')
    if ortam:
        return Path(ortam)
    for a in VARSAYILAN_DRIVE:
        if Path(a).is_dir():
            return Path(a)
    raise SystemExit('Drive klasörü bulunamadı — --drive ile yolu verin.')


def _sayi(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def egri(csv_yolu: Path):
    """results.csv → satır sözlükleri. Bozuk satırlar atlanır."""
    with io.open(csv_yolu, encoding='utf-8', errors='ignore') as f:
        return [s for s in csv.DictReader(f) if s]


def hedef_epoch(kosu: Path, varsayilan=200) -> int:
    a = kosu / 'args.yaml'
    if not a.exists():
        return varsayilan
    for satir in a.read_text(encoding='utf-8', errors='ignore').splitlines():
        if satir.startswith('epochs:'):
            v = _sayi(satir.split(':', 1)[1])
            if v:
                return int(v)
    return varsayilan


def olcum(kosu: Path) -> dict:
    """Bir koşunun durumu — tek yerden hesaplanır ki rapor ve --bekle aynı şeyi görsün."""
    c = kosu / 'results.csv'
    if not c.exists():
        return {'ad': kosu.name, 'durum': 'checkpoint yok', 'epoch': 0}

    sat = egri(c)
    if not sat:
        return {'ad': kosu.name, 'durum': 'boş results.csv', 'epoch': 0}

    hedef = hedef_epoch(kosu)
    t = [_sayi(s.get('time')) for s in sat]
    t = [v for v in t if v is not None]

    # Süreç sınırları: `time` geriye gittiği yerlerde yeniden başlatılmış.
    kesinti = [i for i in range(1, len(t)) if t[i] < t[i - 1]]
    sinir = [0] + kesinti + [len(t)]
    gpu_sn = 0.0
    epoch_sn = None
    for a, b in zip(sinir, sinir[1:]):
        if b - a < 2:
            continue
        dilim = t[a:b]
        gpu_sn += dilim[-1] - dilim[0]
        sureler = sorted(dilim[i] - dilim[i - 1] for i in range(1, len(dilim)))
        orta = sureler[len(sureler) // 2]
        epoch_sn = orta if epoch_sn is None else min(epoch_sn, orta)

    # mAP: kolon adı sürümle değişebiliyor, içeriğinden bul
    anahtar = next((k for k in sat[0] if 'mAP50-95' in k), None)
    en_iyi = en_iyi_ep = None
    if anahtar:
        degerler = [(_sayi(s.get(anahtar)), i + 1) for i, s in enumerate(sat)]
        degerler = [(v, i) for v, i in degerler if v is not None]
        if degerler:
            en_iyi, en_iyi_ep = max(degerler)

    yas = time.time() - c.stat().st_mtime
    canli = epoch_sn is not None and yas < epoch_sn * 2.5

    return {
        'ad': kosu.name, 'epoch': len(sat), 'hedef': hedef,
        'kesinti': len(kesinti), 'kesinti_epoch': [i + 1 for i in kesinti],
        'gpu_saat': gpu_sn / 3600, 'epoch_sn': epoch_sn,
        'en_iyi': en_iyi, 'en_iyi_epoch': en_iyi_ep,
        'yas_sn': yas, 'canli': canli,
        'durum': 'ÇALIŞIYOR' if canli else ('bitti' if len(sat) >= hedef else 'DURDU'),
    }


def olu_zaman(kosu: Path, epoch_sn):
    """epoch*.pt dosya zamanlarından kayıp süreyi çıkarır.

    Ultralytics her save_period epoch'ta bir anlık görüntü yazar. İki
    ardışık görüntü arasındaki süre beklenenden fazlaysa fark kayıptır.
    """
    if not epoch_sn:
        return []
    w = kosu / 'weights'
    if not w.is_dir():
        return []
    anlar = []
    for f in w.glob('epoch*.pt'):
        n = _sayi(f.stem.replace('epoch', ''))
        if n is not None:
            anlar.append((int(n), f.stat().st_mtime))
    anlar.sort()
    kayip = []
    for (e1, t1), (e2, t2) in zip(anlar, anlar[1:]):
        beklenen = (e2 - e1) * epoch_sn
        fark = (t2 - t1) - beklenen
        if fark > max(120, beklenen * 0.25):
            kayip.append((e1, e2, fark))
    return kayip


def yazdir(kosu: Path, ayrintili=False):
    o = olcum(kosu)
    if o.get('epoch', 0) == 0:
        print(f"  {o['ad']:<26} {o['durum']}")
        return o

    isaret = {'ÇALIŞIYOR': '🟢', 'DURDU': '🔴', 'bitti': '✅'}[o['durum']]
    print(f"  {isaret} {o['ad']:<24} {o['epoch']:>4}/{o['hedef']:<4} epoch  "
          f"{o['epoch_sn'] or 0:>4.0f} sn/epoch  "
          f"mAP50-95 {o['en_iyi'] or 0:.4f}  "
          f"{o['gpu_saat']:.1f} sa GPU"
          + (f"  ⚠️ {o['kesinti']} kesinti" if o['kesinti'] else ''))

    if o['durum'] == 'DURDU':
        print(f"      son yazım {o['yas_sn'] / 60:.0f} dakika önce — "
              'eğitim ölmüş, Colab\'de eğitim hücresini yeniden çalıştırın')
        print("      (MOD='otomatik' kaldığı epoch'tan devam eder, ilerleme kaybolmaz)")
    elif o['durum'] == 'ÇALIŞIYOR' and o['epoch_sn']:
        kalan = (o['hedef'] - o['epoch']) * o['epoch_sn']
        print(f"      kalan ~{kalan / 3600:.1f} saat ({o['hedef'] - o['epoch']} epoch)")

    if ayrintili:
        if o['kesinti']:
            print(f"      kesinti epoch'ları: {o['kesinti_epoch']}")
        kayip = olu_zaman(kosu, o['epoch_sn'])
        if kayip:
            toplam = sum(f for _, _, f in kayip)
            print(f'      ÖLÜ ZAMAN: {toplam / 3600:.1f} saat '
                  f'(fark edilmeyen ölümler)')
            for e1, e2, f in kayip:
                print(f'        epoch {e1}→{e2}: +{f / 60:.0f} dakika')
    return o


def main():
    ap = argparse.ArgumentParser(description="Drive'daki eğitim koşularını izler")
    ap.add_argument('kosu', nargs='?', help='Tek koşu adı (boş = hepsi)')
    ap.add_argument('--drive', default=None, help='Drive proje klasörü')
    ap.add_argument('--bekle', action='store_true',
                    help='Koşu ölünce/bitince haber ver (sürekli izler)')
    ap.add_argument('--aralik', type=int, default=300, help='--bekle yoklama aralığı (sn)')
    a = ap.parse_args()

    kok = drive_bul(a.drive) / 'results'
    if not kok.is_dir():
        raise SystemExit(f'Koşu klasörü yok: {kok}')

    def kosular():
        d = [p for p in kok.iterdir() if p.is_dir()]
        if a.kosu:
            d = [p for p in d if p.name == a.kosu]
            if not d:
                raise SystemExit(f'Koşu bulunamadı: {a.kosu}\n'
                                 f"Mevcut: {sorted(p.name for p in kok.iterdir() if p.is_dir())}")
        return sorted(d, key=lambda p: p.stat().st_mtime, reverse=True)

    if not a.bekle:
        print(f'📂 {kok}\n')
        for k in kosular():
            yazdir(k, ayrintili=bool(a.kosu))
        return 0

    # --bekle: yalnızca DURUM DEĞİŞİNCE yaz, yoksa ekranı kirletir
    onceki = {}
    print(f'👁️  İzleniyor (her {a.aralik} sn) — Ctrl+C ile çık\n')
    while True:
        for k in kosular():
            o = olcum(k)
            imza = (o['durum'], o.get('epoch'))
            if onceki.get(k.name) == imza:
                continue
            eski = onceki.get(k.name)
            onceki[k.name] = imza
            if eski is None:
                yazdir(k)
                continue
            if eski[0] != o['durum']:
                print(f"\n{'=' * 62}")
                print(f"⚠️  {k.name}: {eski[0]} → {o['durum']}  "
                      f"({time.strftime('%H:%M:%S')})")
                print('=' * 62)
                yazdir(k)
            sys.stdout.flush()
        time.sleep(a.aralik)


if __name__ == '__main__':
    raise SystemExit(main())
