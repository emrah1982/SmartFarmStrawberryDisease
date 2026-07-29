"""YOLO etiketlerindeki bozuk kutuları onarır: taşanları kırpar, geçersizleri atar.

SORUN
    Roboflow dışa aktarımlarında kutuların bir kısmı görüntü sınırlarının
    DIŞINA taşar (x-w/2 < 0 gibi). Etiket dosyasındaki her sayı 0-1 aralığında
    olduğu için hiçbir doğrulayıcı uyarmaz — hata ancak kutu çizilince görülür.
    Genelde kaynak görüntü kırpılıp/yeniden boyutlandırılıp kutular buna göre
    güncellenmediğinde oluşur.

NEDEN ÖNEMLİ
    Model, nesnenin GÖRÜNEN kısmına bakıp görüntü dışına taşan bir kutu
    tahmin etmeyi öğrenir. Merkez ve boyut hedefleri sistematik olarak kayar;
    en çok mAP50-95 (konumlandırma hassasiyeti) zarar görür. Kırpma, hedefi
    nesnenin gerçekten görünen kısmına oturtur.

    Ayrıca genişliği/yüksekliği sıfır olan kutular hiçbir şey öğretmez ve bazı
    artırma (augmentation) kütüphanelerinde hata verir.

KULLANIM
    python scripts/etiket_temizle.py --kuru            # rapor, hiçbir şey yazmaz
    python scripts/etiket_temizle.py                   # yapılandırmadaki tüm kaynaklar
    python scripts/etiket_temizle.py --inputs "dataset/X" --kok "G:/.../dataset"

Yedek: her etiket klasörünün yanına `labels_temizlik_oncesi/` yazılır (yalnızca
ilk çalıştırmada). Kırpma işlemi kendi kendine tekrarlanabilir — betiği iki kez
çalıştırmak veriyi bozmaz.
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parent.parent
VARSAYILAN_YAPILANDIRMA = None      # ürüne göre çözülür (bkz. veri_yapilandirmasi)

# --- Ürün kapsamı (çok bitkili kurulum) -------------------------------------
# Eğitim yapılandırması her ürünün kendi klasöründedir:
#   configs/urunler/<urun>/veri.yaml
# Eski kurulumlarda configs/strawberry_data.yaml'a düşülür.
def veri_yapilandirmasi(urun: str = None) -> Path:
    import os
    urun = urun or os.environ.get('VARSAYILAN_URUN', 'cilek')
    yeni = KOK / 'configs' / 'urunler' / urun / 'veri.yaml'
    return yeni if yeni.exists() else KOK / 'configs' / 'strawberry_data.yaml'


# Kırpma sonrası bu değerden ince kalan kutu atılır (görüntü kenarının binde biri).
EN_KUCUK_KENAR = 0.001


def kirp(x, y, w, h):
    """Kutuyu [0,1] karesine kırpar ve merkez/boyutu yeniden hesaplar.

    Returns: (x, y, w, h) veya None (kutu tamamen dışarıda/çok ince kaldıysa)
    """
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    yw, yh = x2 - x1, y2 - y1
    if yw <= EN_KUCUK_KENAR or yh <= EN_KUCUK_KENAR:
        return None
    return (x1 + yw / 2, y1 + yh / 2, yw, yh)


def dosya_temizle(satirlar):
    """Returns: (yeni_satırlar, kırpılan, atılan)"""
    yeni, kirpilan, atilan = [], 0, 0
    for satir in satirlar:
        p = satir.split()
        if len(p) < 5:
            atilan += 1          # eksik/bozuk satır da rapora girsin
            continue
        try:
            c = int(float(p[0]))
            x, y, w, h = (float(v) for v in p[1:5])
        except ValueError:
            atilan += 1
            continue

        if w <= 0 or h <= 0:
            atilan += 1
            continue

        tasma = max(0 - (x - w / 2), 0 - (y - h / 2), (x + w / 2) - 1, (y + h / 2) - 1)
        if tasma <= 1e-6:
            yeni.append(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
            continue

        onarilan = kirp(x, y, w, h)
        if onarilan is None:
            atilan += 1
            continue
        kirpilan += 1
        yeni.append('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(c, *onarilan))
    return yeni, kirpilan, atilan


def etiket_dizinleri(yapilandirma: Path, kok: Path):
    """strawberry_data.yaml'daki görüntü dizinlerinden etiket dizinlerini bulur."""
    cfg = yaml.safe_load(yapilandirma.read_text(encoding='utf-8')) or {}
    dizinler = []
    for bolum in ('train', 'val', 'test'):
        for yol in cfg.get(bolum) or []:
            d = (kok / yol.replace('../dataset/', '')).parent / 'labels'
            if d.is_dir() and d not in dizinler:
                dizinler.append(d)
    return dizinler


def calistir(dizinler, kuru: bool) -> int:
    toplam = Counter()
    for d in dizinler:
        yedek = d.parent / 'labels_temizlik_oncesi'
        if not yedek.exists() and not kuru:
            shutil.copytree(d, yedek)

        dosya = kirpilan = atilan = degisen = 0
        for f in sorted(d.glob('*.txt')):
            eski = [s for s in f.read_text(encoding='utf-8').splitlines() if s.strip()]
            yeni, k, a = dosya_temizle(eski)
            dosya += 1
            kirpilan += k
            atilan += a
            if k or a:
                degisen += 1
                if not kuru:
                    f.write_text('\n'.join(yeni) + ('\n' if yeni else ''), encoding='utf-8')

        toplam['dosya'] += dosya
        toplam['kirpilan'] += kirpilan
        toplam['atilan'] += atilan
        toplam['degisen'] += degisen
        ad = d.parent.parent.name if d.parent.parent.name != 'dataset' else d.parent.name
        print(f'  {ad[:40]:42} {dosya:>5} dosya  {kirpilan:>5} kırpıldı  {atilan:>4} atıldı')

    print(f"\n{'[KURU] ' if kuru else ''}TOPLAM: {toplam['dosya']} dosya · "
          f"{toplam['kirpilan']} kutu kırpıldı · {toplam['atilan']} kutu atıldı · "
          f"{toplam['degisen']} dosya değişti")
    if kuru:
        print('Hiçbir dosya yazılmadı. Uygulamak için --kuru olmadan çalıştırın.')
    return 0


def main():
    ap = argparse.ArgumentParser(description='YOLO etiketlerindeki taşan/bozuk kutuları onarır')
    ap.add_argument('--inputs', nargs='*', default=[],
                    help='Etiket dizinleri veya dataset kökleri (boşsa yapılandırmadan alınır)')
    ap.add_argument('--urun', default=None)
    ap.add_argument('--yapilandirma', default=None)
    ap.add_argument('--kok', default='', help='Dataset kökü (yapılandırmadaki ../dataset/ yerine)')
    ap.add_argument('--kuru', action='store_true', help='Yazmadan raporla')
    a = ap.parse_args()

    if a.inputs:
        dizinler = []
        for g in a.inputs:
            p = Path(g)
            dizinler += [p] if p.name == 'labels' else [
                d for d in sorted(p.rglob('labels')) if d.is_dir()]
    else:
        kok = Path(a.kok) if a.kok else KOK / 'dataset'
        yap = Path(a.yapilandirma) if a.yapilandirma else veri_yapilandirmasi(a.urun)
        dizinler = etiket_dizinleri(yap, kok)

    if not dizinler:
        print('Etiket dizini bulunamadı.')
        return 1
    print(f'{len(dizinler)} etiket dizini işlenecek:\n')
    return calistir(dizinler, a.kuru)


if __name__ == '__main__':
    sys.exit(main())
