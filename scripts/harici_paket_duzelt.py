"""Dışarıdan indirilen (Roboflow vb.) dataset paketini proje standardına çevirir.

ÜÇ SORUNU BİRDEN ÇÖZER

1. SIZINTI  (en tehlikelisi, sessizdir)
    Roboflow artırımı tek kaynak fotoğraftan birkaç kopya üretir; bölme
    artırımdan SONRA yapıldıysa aynı fotoğrafın kopyaları hem train hem
    valid'e düşer. Model doğrulama setini zaten görmüştür: mAP yüksek
    çıkar, sahada çöker. Ölçülen örnek (pest_detection):

        valid'in %99,6'sı train görüntülerinin kopyasıydı
        (aynı fotoğraf, biri 90° döndürülmüş)

    Bu betik kopyaları KAYNAĞA göre gruplar ve bölmeyi grup düzeyinde
    yapar — bir kaynağın bütün kopyaları aynı bölmede kalır.

2. YOL BİÇİMİ
    Roboflow `train: ../train/images` yazar. Bizim data.yaml'ımız kendi
    klasörünün içini gösterir (`train: train/images`) ve `path` anahtarı
    BİLEREK yazılmaz — böylece klasör taşınsa da, Colab'de açılsa da
    çalışır. Ayrıntı: datasets/*/data.yaml başlığı.

3. OLMAYAN BÖLÜM
    `test: ../test/images` yazılıp test klasörü konmamış olabilir;
    Ultralytics test istediğinde hata verir.

KULLANIM
    python scripts/harici_paket_duzelt.py datasets/cilek/pest_detection.zip \\
        --ad bocek_teshis \\
        --siniflar "Army Worm,Black Cutworm,Grub,Mole Cricket,Peach Borer,Red Spider Mite"

    python scripts/harici_paket_duzelt.py <zip> --kuru      # yalnızca rapor

SINIF ADI DEĞİŞTİRME
    --siniflar SIRAYA göre eşleşir; ID'ler değişmez, yalnızca görünen ad
    değişir. Sıra yanlış verilirse etiketler sessizce yanlış sınıfa kayar,
    bu yüzden betik eski ve yeni adı yan yana yazıp onay ister.
"""

import argparse
import io
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parent.parent

# Roboflow dosya adi: '<n>_<m>_<KAYNAK>_jpg.rf.<karma>.jpg'
# Kaynagi bulmak icin artirim karmasini, sayisal oneki ve '_jpg' ekini atariz.
RF_KARMA = re.compile(r'\.rf\.[0-9a-f]+', re.I)
RF_ONEK = re.compile(r'^\d+_\d+_')
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def kaynak_kimligi(dosya_adi: str) -> str:
    """Artırılmış kopyaların ORTAK kaynağını verir.

    '605_2_Black_Cutworm--1-_jpg.rf.2d51...jpg' → 'black_cutworm--1-'
    Aynı kaynağın bütün kopyaları aynı dizeyi üretir.
    """
    t = Path(dosya_adi).stem
    t = RF_KARMA.sub('', t)
    t = RF_ONEK.sub('', t)
    t = re.sub(r'_jpg$', '', t)
    return t.lower()


def paketi_ac(kaynak: Path, hedef: Path) -> Path:
    """Zip veya klasörü geçici dizine alır, dataset kökünü döner."""
    if kaynak.is_dir():
        return kaynak
    with zipfile.ZipFile(kaynak) as zf:
        zf.extractall(hedef)
    # data.yaml'in bulundugu EN SIG dizin kok kabul edilir
    adaylar = sorted(hedef.rglob('data.yaml'), key=lambda p: len(p.parts))
    if adaylar:
        return adaylar[0].parent
    # data.yaml yoksa train/ iceren dizini ara
    for d in [hedef, *hedef.iterdir()]:
        if d.is_dir() and (d / 'train').is_dir():
            return d
    return hedef


def bolumleri_topla(kok: Path):
    """{bölüm: [(görüntü, etiket|None)]} — mevcut bölmeyi olduğu gibi okur."""
    out = {}
    for b in ('train', 'valid', 'val', 'test'):
        img_d = kok / b / 'images'
        if not img_d.is_dir():
            continue
        ciftler = []
        for g in sorted(img_d.iterdir()):
            if g.suffix.lower() not in GORUNTU_UZANTI:
                continue
            e = kok / b / 'labels' / (g.stem + '.txt')
            ciftler.append((g, e if e.exists() else None))
        if ciftler:
            out['valid' if b == 'val' else b] = ciftler
    return out


def sizinti_olc(bolumler):
    """Bölmeler arasında kaç kaynak paylaşılıyor?"""
    kume = {b: {kaynak_kimligi(g.name) for g, _ in c} for b, c in bolumler.items()}
    rapor = []
    adlar = list(kume)
    for i, a in enumerate(adlar):
        for b in adlar[i + 1:]:
            ortak = kume[a] & kume[b]
            if ortak:
                kucuk = min(len(kume[a]), len(kume[b]))
                rapor.append((a, b, len(ortak), 100 * len(ortak) / kucuk))
    return rapor


def yeniden_bol(bolumler, oranlar=(0.80, 0.15, 0.05), tohum=0):
    """Kaynak grubu düzeyinde böler — bir kaynağın kopyaları hep aynı bölmede.

    Sınıf dengesi korunsun diye gruplar BASKIN SINIFA göre sıralanıp
    sırayla dağıtılır; rastgele bölme küçük sınıfları bir bölmede yığabilir.
    """
    grup = defaultdict(list)
    for ciftler in bolumler.values():
        for g, e in ciftler:
            grup[kaynak_kimligi(g.name)].append((g, e))

    def baskin_sinif(ogeler):
        say = Counter()
        for _, e in ogeler:
            if e and e.exists():
                for satir in e.read_text(encoding='utf-8', errors='ignore').splitlines():
                    p = satir.split()
                    if p:
                        say[p[0]] += 1
        return say.most_common(1)[0][0] if say else '?'

    sinifa_gore = defaultdict(list)
    for k, ogeler in grup.items():
        sinifa_gore[baskin_sinif(ogeler)].append(k)

    yeni = {'train': [], 'valid': [], 'test': []}
    rng = random.Random(tohum)
    for _, gruplar in sorted(sinifa_gore.items()):
        gruplar = sorted(gruplar)
        rng.shuffle(gruplar)
        n = len(gruplar)
        n_tr = int(n * oranlar[0])
        n_va = int(n * oranlar[1])
        # Kucuk siniflarda valid/test bos kalmasin
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(n_va, 1)
        for i, k in enumerate(gruplar):
            b = 'train' if i < n_tr else ('valid' if i < n_tr + n_va else 'test')
            yeni[b].extend(grup[k])
    return yeni, len(grup)


def siniflari_say(bolumler, n_sinif):
    say = {b: Counter() for b in bolumler}
    for b, ciftler in bolumler.items():
        for _, e in ciftler:
            if e and e.exists():
                for satir in e.read_text(encoding='utf-8', errors='ignore').splitlines():
                    p = satir.split()
                    if p:
                        say[b][int(p[0])] += 1
    return say


def yaz(yeni_bolumler, hedef: Path, siniflar):
    if hedef.exists():
        shutil.rmtree(hedef)
    for b, ciftler in yeni_bolumler.items():
        if not ciftler:
            continue
        (hedef / b / 'images').mkdir(parents=True, exist_ok=True)
        (hedef / b / 'labels').mkdir(parents=True, exist_ok=True)
        for g, e in ciftler:
            shutil.copy2(g, hedef / b / 'images' / g.name)
            if e and e.exists():
                shutil.copy2(e, hedef / b / 'labels' / e.name)

    basli = (
        "# Bu dataset TASINABILIR: 'path' anahtari BILEREK YAZILMAZ.\n"
        '#\n'
        '# Ultralytics kok dizini soyle cozer (data/utils.py):\n'
        "#   path = data.get('path') or Path(data['yaml_file']).parent\n"
        "# 'path' yoksa yaml'in KENDI klasoru kok olur. Roboflow'un yazdigi\n"
        '# "../train/images" bicimi bir ust dizine cikar ve yanlis yeri gosterir.\n'
        '#\n'
        '# Bolme KAYNAK GRUBUNA gore yapildi: ayni fotografin artirilmis\n'
        '# kopyalari hep ayni bolmededir, train/valid sizintisi yoktur.\n'
    )
    govde = {}
    for b, k in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
        if yeni_bolumler.get(b):
            govde[k] = f'{b}/images'
    govde['nc'] = len(siniflar)
    govde['names'] = {i: s for i, s in enumerate(siniflar)}
    (hedef / 'data.yaml').write_text(
        basli + yaml.dump(govde, allow_unicode=True, sort_keys=False), encoding='utf-8')


def paketle(dizin: Path) -> Path:
    zip_yolu = dizin.parent / (dizin.name + '.zip')
    if zip_yolu.exists():
        zip_yolu.unlink()
    with zipfile.ZipFile(zip_yolu, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for p in sorted(dizin.rglob('*')):
            if p.is_file():
                zf.write(p, str(Path(dizin.name) / p.relative_to(dizin)).replace('\\', '/'))
    return zip_yolu


def main():
    ap = argparse.ArgumentParser(description='Harici dataset paketini standarda çevirir')
    ap.add_argument('paket', help='Zip dosyası veya açılmış klasör')
    ap.add_argument('--ad', default=None, help='Hedef dataset adı (varsayılan: paketin adı)')
    ap.add_argument('--siniflar', default=None,
                    help='Yeni sınıf adları, SIRAYLA, virgülle ayrılmış')
    ap.add_argument('--urun', default='cilek')
    ap.add_argument('--oran', default='0.80,0.15,0.05', help='train,valid,test')
    ap.add_argument('--tohum', type=int, default=0)
    ap.add_argument('--kuru', action='store_true', help='Yalnızca rapor, yazma')
    ap.add_argument('--paketle', action='store_true', help='Sonucu zip olarak da üret')
    a = ap.parse_args()

    kaynak = Path(a.paket)
    if not kaynak.exists():
        print(f'❌ Yok: {kaynak}')
        return 1
    ad = a.ad or kaynak.stem
    gecici = Path(tempfile.mkdtemp(prefix='harici_'))
    try:
        kok = paketi_ac(kaynak, gecici)
        cfg = {}
        y = kok / 'data.yaml'
        if y.exists():
            cfg = yaml.safe_load(y.read_text(encoding='utf-8')) or {}
        n = cfg.get('names') or []
        eski_siniflar = [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)

        bolumler = bolumleri_topla(kok)
        if not bolumler:
            print(f'❌ {kok} içinde <bölüm>/images bulunamadı.')
            return 1

        print('=' * 72)
        print(f'KAYNAK: {kaynak}')
        print('=' * 72)
        for b, c in bolumler.items():
            eksik = sum(1 for _, e in c if e is None)
            print(f'  {b:6} {len(c):>6} görüntü' + (f'  ({eksik} etiketsiz)' if eksik else ''))
        for k in ('train', 'val', 'test'):
            if k in cfg:
                print(f"  data.yaml {k:5} = {cfg[k]!r}"
                      + ('   ⚠️ üst dizine çıkıyor' if str(cfg[k]).startswith('..') else ''))
        if cfg.get('path'):
            print(f"  data.yaml path  = {cfg['path']!r}   ⚠️ taşınabilirliği bozar")

        # --- Sızıntı ------------------------------------------------------
        print('\n--- SIZINTI KONTROLÜ (kaynak fotoğraf düzeyinde) ---')
        rapor = sizinti_olc(bolumler)
        if rapor:
            for x, z, adet, yuzde in rapor:
                print(f'  ⛔ {x} ↔ {z}: {adet} ortak kaynak (küçük bölmenin %{yuzde:.1f}\'i)')
            print('     Doğrulama skoru ANLAMSIZ olurdu; bölme yeniden yapılacak.')
        else:
            print('  ✅ bölmeler arası ortak kaynak yok')

        # --- Sınıf adları --------------------------------------------------
        siniflar = eski_siniflar
        if a.siniflar:
            yeni = [s.strip() for s in a.siniflar.split(',') if s.strip()]
            if len(yeni) != len(eski_siniflar):
                print(f'\n⛔ Sınıf sayısı uyuşmuyor: datasette {len(eski_siniflar)}, '
                      f'verilen {len(yeni)}')
                return 1
            print('\n--- SINIF ADI DEĞİŞİKLİĞİ (sıra korunur, ID değişmez) ---')
            for i, (e, y2) in enumerate(zip(eski_siniflar, yeni)):
                isaret = '' if e == y2 else '  ←'
                print(f'  {i}: {e:<22} → {y2}{isaret}')
            siniflar = yeni

        # --- Yeniden bölme -------------------------------------------------
        oranlar = tuple(float(x) for x in a.oran.split(','))
        yeni_bolumler, grup_sayisi = yeniden_bol(bolumler, oranlar, a.tohum)
        print(f'\n--- YENİDEN BÖLME ({grup_sayisi} kaynak grubu) ---')
        say = siniflari_say(yeni_bolumler, len(siniflar))
        basliklar = [b for b in ('train', 'valid', 'test') if yeni_bolumler.get(b)]
        print('  %-24s %s' % ('sınıf', ''.join(f'{b:>9}' for b in basliklar)))
        for i, s in enumerate(siniflar):
            print('  %-24s %s' % (s, ''.join(f'{say[b][i]:>9}' for b in basliklar)))
        print('  %-24s %s' % ('GÖRÜNTÜ',
                              ''.join(f'{len(yeni_bolumler[b]):>9}' for b in basliklar)))

        kontrol = sizinti_olc({b: yeni_bolumler[b] for b in basliklar})
        print('  sızıntı sonrası: ' + ('⛔ HÂLÂ VAR' if kontrol else '✅ temiz'))
        if kontrol:
            return 1

        if a.kuru:
            print('\n(--kuru: hiçbir şey yazılmadı)')
            return 0

        hedef = KOK / 'datasets' / a.urun / ad
        yaz(yeni_bolumler, hedef, siniflar)
        print(f'\n✅ Yazıldı: {hedef}')
        if a.paketle:
            z = paketle(hedef)
            print(f'📦 Paket:   {z}  ({z.stat().st_size / 1e6:.0f} MB)')
        print('\nSıradaki adımlar:')
        print(f'  1. {ad}.zip dosyasını Drive\'da datasets/{a.urun}/ altına yükleyin')
        print(f"  2. Notebook 4️⃣ hücresi: EGITILECEK = '{ad}'")
        return 0
    finally:
        shutil.rmtree(gecici, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
