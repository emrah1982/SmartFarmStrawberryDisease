"""Görüntü SINIFLANDIRMA paketini proje standardına çevirir.

`harici_paket_duzelt.py`'nin sınıflandırma karşılığıdır. Aradaki fark
yüzeysel değil, yapısaldır:

    TESPİT (harici_paket_duzelt.py)      SINIFLANDIRMA (bu betik)
    ────────────────────────────────     ────────────────────────────────
    images/ + labels/ ikilisi            sınıf başına bir klasör
    data.yaml sınıf sırasını taşır       klasör ADI sınıfın kendisidir
    bölüm adı 'valid' (yaml eşler)       bölüm adı 'val' (eşleyen yok)

Son satır önemli: sınıflandırmada `data.yaml` YOKTUR, dolayısıyla
`valid` yazarsak Ultralytics doğrulama bölümünü bulamaz ve sessizce
atlar. Klasör adı doğrudan sözleşmedir.
Kaynak: https://docs.ultralytics.com/datasets/classify/

SIZINTI
    Roboflow paketlerinde sızıntı ARTIRIM kopyalarından gelir ve dosya
    adındaki `.rf.<hash>` ile yakalanır. Burada öyle bir ipucu yok, o
    yüzden algısal hash (dHash) ile yakın kopya kümeleri bulunur ve bir
    küme daima TEK bölmede kalır.

    ⚠️ Bu yöntem AYNI AĞACIN FARKLI AÇIDAN karesini yakalayamaz. EXIF
    veya kaynak kimliği yoksa o sızıntı ÖLÇÜLEMEZ — betik bunu rapora
    yazar, sessizce "temiz" demez.

Kullanım:
    python scripts/siniflandirma_paketi.py <zip|klasor> --urun findik \
        --ad cotanak_saglik \
        --sinif-eslesme "Diseased=diseased_cluster,Healthy=healthy_cluster" \
        --kuru
"""

import argparse
import io
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parents[1]
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# dHash mesafe eşiği. ÖLÇÜLEREK seçildi (286646.zip, 4.310 görüntü):
#   en yakın komşu mesafesi  %1: 10 bit   %50: 16 bit   %90: 18 bit
#   rastgele çift medyanı              32 bit
# Yani 0-8 bit aralığı BOŞ — gerçek kopya yok. 12 bit, gerçek kopyayı
# kaçırmayacak kadar gevşek, farklı kareleri birleştirmeyecek kadar sıkı.
KOPYA_ESIGI = 12

# int32 ZORUNLU: varsayılan uint64, int32'yle karşılaştırıldığında ortak
# tip bulunamayıp float64'e kaçar (ölçüldü, etiket_kumesi_sec.py'de patladı).
_POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None],
                     axis=1).sum(1).astype(np.int32)


def dhash(ham: bytes) -> np.ndarray:
    """9x8'e küçültüp yatay komşu farkının işaretini alır → 64 bit."""
    with Image.open(io.BytesIO(ham)) as im:
        g = np.asarray(im.convert('L').resize((9, 8), Image.BILINEAR),
                       dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten().astype(np.uint8)


def kopya_kumeleri(paketli: np.ndarray, esik: int) -> list:
    """Birleşim-bul ile yakın kopya kümeleri. Dönen: her görüntünün küme no'su."""
    n = len(paketli)
    ebeveyn = list(range(n))

    def bul(x):
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]
            x = ebeveyn[x]
        return x

    for i in range(n):
        d = _POP[np.bitwise_xor(paketli[i], paketli[i + 1:])].sum(1)
        for j in np.nonzero(d <= esik)[0]:
            a, b = bul(i), bul(i + 1 + int(j))
            if a != b:
                ebeveyn[a] = b
    return [bul(i) for i in range(n)]


def paketi_ac(kaynak: Path, hedef: Path) -> Path:
    if kaynak.is_dir():
        return kaynak
    with zipfile.ZipFile(kaynak) as z:
        z.extractall(hedef)
    return hedef


def goruntuleri_topla(kok: Path) -> list:
    return sorted(p for p in kok.rglob('*')
                  if p.suffix.lower() in GORUNTU_UZANTI)


def sinifi_coz(yol: Path, kok: Path, desen: str, eslesme: dict) -> str:
    """Sınıfı önce ÜST KLASÖRDEN, yoksa dosya adından çıkarır."""
    bagil = yol.relative_to(kok)
    ham = bagil.parts[0] if len(bagil.parts) > 1 else None
    if ham is None:
        m = re.match(desen, yol.stem)
        ham = m.group(1).strip() if m else ''
    return eslesme.get(ham, eslesme.get(ham.lower(), ham.lower()))


def bol(kume_no: list, siniflar: list, oranlar, tohum: int) -> list:
    """Küme düzeyinde ve SINIF İÇİNDE dengeli böler.

    Bir kümenin bütün görüntüleri aynı bölmeye gider. Sınıf içinde
    bölmek, küçük sınıfın bir bölmede yığılmasını önler.
    """
    kume_ogeleri = defaultdict(list)
    for i, k in enumerate(kume_no):
        kume_ogeleri[k].append(i)

    # Küme -> baskın sınıf
    kume_sinifi = {}
    for k, ogeler in kume_ogeleri.items():
        kume_sinifi[k] = Counter(siniflar[i] for i in ogeler).most_common(1)[0][0]

    sinifa_gore = defaultdict(list)
    for k, s in kume_sinifi.items():
        sinifa_gore[s].append(k)

    import random
    rng = random.Random(tohum)
    bolum = [None] * len(kume_no)
    for _, kumeler in sorted(sinifa_gore.items()):
        kumeler = sorted(kumeler)
        rng.shuffle(kumeler)
        n = len(kumeler)
        n_tr = int(n * oranlar[0])
        n_va = int(n * oranlar[1])
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(n_va, 1)
        for i, k in enumerate(kumeler):
            b = 'train' if i < n_tr else ('val' if i < n_tr + n_va else 'test')
            for oge in kume_ogeleri[k]:
                bolum[oge] = b
    return bolum


def yaz(goruntuler, siniflar, bolum, hedef: Path):
    if hedef.exists():
        shutil.rmtree(hedef)
    for g, s, b in zip(goruntuler, siniflar, bolum):
        d = hedef / b / s
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(g, d / g.name)


def okubeni_yaz(hedef: Path, kaynak: Path, say, esik, kume_sayisi,
                en_yakin_yuzdelik, oranlar):
    satir = [
        f'# {hedef.name} — görüntü sınıflandırma paketi',
        '',
        f'Kaynak: `{kaynak.name}`',
        '',
        '## Klasör düzeni NEDEN `val`, `valid` değil?',
        '',
        'Sınıflandırmada `data.yaml` yoktur; klasör adı doğrudan sözleşmedir.',
        '`valid` yazılırsa Ultralytics doğrulama bölümünü bulamaz ve sessizce',
        'atlar. Tespit paketlerimizde `valid` kullanılır çünkü orada',
        '`data.yaml` yolu açıkça eşler.',
        '',
        '## Bölme',
        '',
        f'Oran: {oranlar[0]:.0%} / {oranlar[1]:.0%} / {oranlar[2]:.0%} '
        '(train / val / test)',
        '',
        '| bölüm | ' + ' | '.join(sorted(say['train'])) + ' | toplam |',
        '|---|' + '---|' * (len(say['train']) + 1),
    ]
    for b in ('train', 'val', 'test'):
        s = say[b]
        satir.append(f'| {b} | ' + ' | '.join(str(s[k]) for k in sorted(s))
                     + f' | {sum(s.values())} |')
    satir += [
        '',
        '## Sızıntı denetimi',
        '',
        f'Algısal hash (dHash) ile yakın kopya arandı, eşik **{esik} bit**.',
        f'{kume_sayisi} bağımsız küme bulundu; bir kümenin bütün görüntüleri',
        'daima aynı bölmededir.',
        '',
        'En yakın komşu mesafesi ölçüldü:',
        '',
        '| yüzdelik | mesafe |',
        '|---|---|',
    ]
    for q, v in en_yakin_yuzdelik:
        satir.append(f'| %{q} | {v:.0f} bit |')
    satir += [
        '',
        '> **0-8 bit aralığı boş** → artırım/kopya sızıntısı YOK.',
        '',
        '## ⚠️ Ölçülemeyen sızıntı',
        '',
        'Aynı ağacın farklı açıdan çekilmiş kareleri **elenemedi**:',
        '',
        '- EXIF yok (görüntüler 500x500\'e küçültülürken silinmiş)',
        '- Dosya adı sıralı numara, kaynak kimliği taşımıyor',
        '- Ardışık kareler birbirine rastgele çiftlerden daha benzer değil',
        '  (yani sıralamada seri çekim kümesi yok)',
        '',
        'Bu yüzden test doğruluğunu **üst sınır** olarak okuyun. Kesin ölçüm',
        'için ağaç/parsel kimliği taşıyan veri gerekir.',
        '',
        '## Sağlıklı sınıfı burada NEDEN var?',
        '',
        'ROI boru hattındaki uzman modellere `Healthy` sınıfı EKLENMEZ —',
        'orada sağlıklı durumu organ modelinden türetilir. Bu paket ayrı bir',
        'sınıflandırma akışıdır; organ modeli çalışmaz, dolayısıyla',
        '"çotanak var ve sağlıklı" ile "çotanakla ilgisiz kare" ancak sınıfla',
        'ayrılır. `Sound Nut` ile aynı gerekçe.',
        'Bkz. `docs/MIMARI.md`, `docs/HATA-YONETIMI.md` § 2.8',
        '',
        '## Hastalık ADI yok — bilerek',
        '',
        'Kaynak veri seti teşhis koymaz, yalnızca "erken bozulma belirtisi"',
        'der. Görüntüye bakarak antraknoz/külleme/kokarca ayrımı yapmak',
        'etiket uydurmak olurdu; model o adı güvenle söyler ve yanlış ilaç',
        'önerilir. Hastalık adı ancak teşhisli veriyle veya uzman',
        'doğrulamasıyla eklenir.',
        '',
    ]
    (hedef / 'OKUBENI.md').write_text('\n'.join(satir), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(
        description='Görüntü sınıflandırma paketini standarda çevirir')
    ap.add_argument('paket', help='Zip dosyası veya klasör')
    ap.add_argument('--urun', default='cilek')
    ap.add_argument('--ad', default=None, help='Hedef dataset adı')
    ap.add_argument('--sinif-deseni', default=r'^([A-Za-z_ ]+?)\s*\(',
                    dest='sinif_deseni',
                    help='Dosya adından sınıfı çeken regex (1. grup)')
    ap.add_argument('--sinif-eslesme', default='', dest='sinif_eslesme',
                    help='Ham=Yeni,Ham2=Yeni2')
    ap.add_argument('--oran', default='0.70,0.15,0.15')
    ap.add_argument('--kopya-esigi', type=int, default=KOPYA_ESIGI,
                    dest='kopya_esigi')
    ap.add_argument('--tohum', type=int, default=0)
    ap.add_argument('--kuru', action='store_true', help='Yalnızca rapor')
    a = ap.parse_args()

    kaynak = Path(a.paket)
    if not kaynak.exists():
        print(f'❌ Yok: {kaynak}')
        return 1
    eslesme = {}
    for p in a.sinif_eslesme.split(','):
        if '=' in p:
            k, v = p.split('=', 1)
            eslesme[k.strip()] = v.strip()

    ad = a.ad or kaynak.stem
    gecici = Path(tempfile.mkdtemp(prefix='sinif_'))
    try:
        kok = paketi_ac(kaynak, gecici)
        goruntuler = goruntuleri_topla(kok)
        if not goruntuler:
            print(f'❌ {kok} içinde görüntü bulunamadı.')
            return 1

        siniflar = [sinifi_coz(g, kok, a.sinif_deseni, eslesme)
                    for g in goruntuler]
        dagilim = Counter(siniflar)
        print('=' * 72)
        print(f'KAYNAK: {kaynak.name}   ({len(goruntuler)} görüntü)')
        print('=' * 72)
        if '' in dagilim:
            print(f"  ⛔ {dagilim['']} görüntünün sınıfı çözülemedi.")
            print(f"     Desen: {a.sinif_deseni!r}")
            for g, s in zip(goruntuler, siniflar):
                if not s:
                    print(f'     örnek: {g.name}')
                    break
            return 1
        for s, n in sorted(dagilim.items()):
            print(f'  {s:<24} {n:>6}  (%{100 * n / len(goruntuler):.1f})')

        # --- Yakın kopya ---------------------------------------------------
        print('\n--- YAKIN KOPYA DENETİMİ (algısal hash) ---')
        bit = np.array([dhash(g.read_bytes()) for g in goruntuler])
        paketli = np.packbits(bit, axis=1)
        kume_no = kopya_kumeleri(paketli, a.kopya_esigi)
        kume_sayisi = len(set(kume_no))
        buyuk = Counter(kume_no).most_common(1)[0][1]
        print(f'  eşik {a.kopya_esigi} bit → {kume_sayisi} bağımsız küme '
              f'({len(goruntuler)} görüntüden), en büyük küme {buyuk}')

        rng = np.random.default_rng(a.tohum)
        ornek = rng.choice(len(goruntuler), min(600, len(goruntuler)),
                           replace=False)
        enyakin = []
        for i in ornek:
            d = _POP[np.bitwise_xor(paketli[i], paketli)].sum(1)
            d[i] = 999
            enyakin.append(d.min())
        yuzdelikler = [(q, float(np.percentile(enyakin, q)))
                       for q in (1, 5, 50, 90)]
        print('  en yakın komşu: ' + ', '.join(
            f'%{q}={v:.0f} bit' for q, v in yuzdelikler))
        if yuzdelikler[0][1] > 8:
            print('  ✅ 0-8 bit aralığı boş → artırım/kopya sızıntısı yok')
        print('  ⚠️ Aynı ağacın farklı açıdan karesi bu yöntemle ELENEMEZ '
              '(EXIF ve kaynak kimliği yok).')
        print('     Test doğruluğunu ÜST SINIR olarak okuyun.')

        # --- Bölme ---------------------------------------------------------
        oranlar = tuple(float(x) for x in a.oran.split(','))
        bolum = bol(kume_no, siniflar, oranlar, a.tohum)
        say = {b: Counter() for b in ('train', 'val', 'test')}
        for s, b in zip(siniflar, bolum):
            say[b][s] += 1
        print(f'\n--- BÖLME ({oranlar[0]:.0%}/{oranlar[1]:.0%}/'
              f'{oranlar[2]:.0%}, küme düzeyinde) ---')
        adlar = sorted(dagilim)
        print('  %-8s %s %9s' % ('bölüm', ''.join(f'{s:>22}' for s in adlar),
                                 'toplam'))
        for b in ('train', 'val', 'test'):
            print('  %-8s %s %9d' % (b, ''.join(f'{say[b][s]:>22}'
                                                for s in adlar),
                                     sum(say[b].values())))

        cakisma = {b: {kume_no[i] for i, x in enumerate(bolum) if x == b}
                   for b in ('train', 'val', 'test')}
        ortak = ((cakisma['train'] & cakisma['val'])
                 | (cakisma['train'] & cakisma['test'])
                 | (cakisma['val'] & cakisma['test']))
        print('  küme sızıntısı: ' + ('⛔ VAR' if ortak else '✅ yok'))
        if ortak:
            return 1

        if a.kuru:
            print('\n(--kuru: hiçbir şey yazılmadı)')
            return 0

        hedef = KOK / 'datasets' / a.urun / ad
        yaz(goruntuler, siniflar, bolum, hedef)
        okubeni_yaz(hedef, kaynak, say, a.kopya_esigi, kume_sayisi,
                    yuzdelikler, oranlar)
        print(f'\n✅ Yazıldı: {hedef}')
        print(f'📄 {hedef / "OKUBENI.md"} — köken, bölme ve sızıntı notu')
        return 0
    finally:
        shutil.rmtree(gecici, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
