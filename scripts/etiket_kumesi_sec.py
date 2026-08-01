"""Elle etiketlenecek ÇEŞİTLİ bir alt küme seçer.

NEDEN RASTGELE SEÇMİYORUZ?
    Elle etiketleme en pahalı adımdır; 400 kareye bütçe ayırıyorsan o 400
    kare varyasyonu KAPSAMALI. Rastgele seçim, veri setinde ne çoksa onu
    tekrar getirir — 400 benzer kare etiketlemiş olursun, model dar bir
    koşul aralığı öğrenir.

    Bunun yerine EN UZAK NOKTA ÖRNEKLEMESİ (farthest-point sampling):
    her adımda, seçilmişlere EN BENZEMEYEN kare eklenir. Sonuç, hash
    uzayına yayılmış bir küme olur.

ÇIKTI, ETİKETLEMEYE HAZIRDIR
    images/   seçilen kareler
    labels/   BOŞ — etiketleyici dolduracak
    data.yaml sınıf listesi ÖNCEDEN sabitlenmiş

    data.yaml'ı şimdi yazmak şart: sınıf SIRASI etiket ID'lerini belirler
    ve etiketleme başladıktan sonra değiştirilemez. Sıra
    configs/urunler/<urun>/modeller.yaml'daki organ modelinden alınır ki
    scripts/model_kur.py sonradan kurulumu reddetmesin.

Kullanım:
    python scripts/etiket_kumesi_sec.py datasets/findik/cotanak_saglik \
        --urun findik --ad organ_etiketleme --adet 400 --kuru
"""

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

KOK = Path(__file__).resolve().parents[1]
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# int32 ZORUNLU: varsayılan uint64 kalırsa np.minimum(int32, uint64)
# ortak tip bulamayıp float64 üretir ve `out=` yazımı patlar.
_POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None],
                     axis=1).sum(1).astype(np.int32)

# Netliği bu yüzdeliğin altında kalan kareler seçime GİRMEZ. Ölçüldü
# (cotanak_saglik, 800 örnek): Laplace varyansı medyanı 869, alt %10
# eşiği 455. En bulanık kare bile etiketlenebilir durumdaydı, o yüzden
# filtre agresif değil — yalnızca en zorları geri plana atar.
NETLIK_ALT_YUZDELIK = 10


def dhash(yol: Path) -> np.ndarray:
    with Image.open(yol) as im:
        g = np.asarray(im.convert('L').resize((9, 8), Image.BILINEAR),
                       dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten().astype(np.uint8)


def netlik(yol: Path) -> float:
    """Laplace varyansı — düşükse bulanık."""
    from numpy.lib.stride_tricks import sliding_window_view
    with Image.open(yol) as im:
        g = np.asarray(im.convert('L'), dtype=float)[::2, ::2]
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    return float((sliding_window_view(g, (3, 3)) * k).sum((-1, -2)).var())


def en_uzak_ornekle(paketli: np.ndarray, adet: int, tohum: int) -> list:
    """Farthest-point sampling: her adımda en 'yeni' kareyi ekler."""
    n = len(paketli)
    if adet >= n:
        return list(range(n))
    rng = np.random.default_rng(tohum)
    ilk = int(rng.integers(n))
    secili = [ilk]
    # en_yakin[i] = i'nin seçililere olan EN KÜÇÜK mesafesi
    en_yakin = _POP[np.bitwise_xor(paketli[ilk], paketli)].sum(1).astype(np.int32)
    en_yakin[ilk] = -1
    for _ in range(adet - 1):
        i = int(np.argmax(en_yakin))
        secili.append(i)
        d = _POP[np.bitwise_xor(paketli[i], paketli)].sum(1)
        np.minimum(en_yakin, d, out=en_yakin)
        en_yakin[i] = -1
    return secili


def organ_siniflari(urun: str) -> list:
    """Sınıf sırası modeller.yaml'daki organ modelinden gelir."""
    p = KOK / 'configs' / 'urunler' / urun / 'modeller.yaml'
    if not p.exists():
        return []
    kutuk = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    for tanim in kutuk.values():
        if (tanim or {}).get('rol') == 'organ':
            return list(tanim.get('siniflar') or [])
    return []


def kilavuz_yaz(hedef: Path, siniflar, kaynak, secim, toplam, netlik_esigi):
    s = [
        '# Organ etiketleme kılavuzu',
        '',
        f'Kaynak: `{kaynak}` ({toplam} kareden {len(secim)} seçildi)',
        '',
        '## Neden bu kareler?',
        '',
        'En uzak nokta örneklemesi: her adımda seçilmişlere EN BENZEMEYEN',
        'kare eklendi. Amaç, etiketleme bütçesini varyasyona harcamak —',
        'rastgele seçim benzer kareleri tekrar getirirdi.',
        f'Netliği alt %{NETLIK_ALT_YUZDELIK} dilimde kalan kareler '
        f'(Laplace varyansı < {netlik_esigi:.0f}) seçime girmedi.',
        '',
        '## Sınıflar — SIRA DEĞİŞTİRİLEMEZ',
        '',
        'Sıra etiket ID\'lerini belirler ve `configs/urunler/<urun>/'
        'modeller.yaml` içindeki organ modelinden alınmıştır. Değiştirilirse',
        '`scripts/model_kur.py` kurulumu reddeder.',
        '',
        '| ID | sınıf | ne kutulanır |',
        '|---|---|---|',
    ]
    aciklama = {
        'Leaf': 'Tek yaprak. Üst üste binen yapraklarda **görünen kısmı** '
                'kutula, tahmini tam şeklini değil.',
        'Nut': 'Fındığın kendisi (kabuk). Zuruf içinde görünüyorsa yalnızca '
               'fındığın görünen kısmı.',
        'Husk': 'Fındığı saran yeşil zuruf. Zuruf + içindeki fındık birlikte '
                'görünüyorsa İKİSİNİ AYRI kutula.',
        'Branch': 'Dal ve odunsu sürgün. İnce yaprak sapını kutulama.',
        'Flower': 'Erkek tırfıl (püskül) veya dişi çiçek.',
    }
    for i, ad in enumerate(siniflar):
        s.append(f'| {i} | `{ad}` | {aciklama.get(ad, "—")} |')
    s += [
        '',
        '## Kurallar',
        '',
        '1. **Kutu, görünen piksele oturur.** Yaprağın arkada kalan kısmını',
        '   tahmin edip kutuyu büyütme — model görmediği şeyi öğrenemez.',
        '2. **Çok küçükse etiketleme.** Kısa kenarı ~16 pikselden küçük',
        '   nesneyi atla; YOLO\'nun tespit ızgarası 8 piksel adımlıdır,',
        '   altındaki nesne zaten öğrenilemez.',
        '3. **Arka plandaki bulanık organları etiketleme.** Odak dışı',
        '   yapraklar gürültü katar.',
        '4. **Emin değilsen boş bırak.** Yanlış kutu, eksik kutudan pahalıdır.',
        '5. **Hastalık etiketlemiyorsun.** Bu aşamada yalnızca ORGAN var.',
        '   Sağlıklı/hastalıklı ayrımı organ modelinin işi değildir —',
        '   sağlıklı organ da aynı sınıfla kutulanır.',
        '',
        '## Sağlıklı organ da etiketlenir',
        '',
        'Bu mimaride `Healthy` diye bir sınıf yoktur: organ modeli yaprağı',
        'bulur, uzman model orada bulgu çıkarmazsa yaprak sağlıklıdır. Yani',
        'sağlıklı yaprak da hastalıklı yaprak da `Leaf` kutusudur.',
        'Bkz. `docs/MIMARI.md` § "Üçüncü fayda".',
        '',
        '## Etiketleme sonrası',
        '',
        '```bash',
        '# 1. Sızıntı ve bölme denetimi',
        'python scripts/harici_paket_duzelt.py <bu_klasor> \\',
        '    --urun <urun> --ad organ_detection --kuru',
        '',
        '# 2. Alan tespiti — saha verisi mi?',
        'python scripts/imgsz_oner.py datasets/<urun>/organ_detection',
        '```',
        '',
        '## Kaynak eşleşmesi',
        '',
        '`secim_kaydi.csv` her seçilen karenin kaynak yolunu ve sağlık',
        'sınıfını tutar. Organ etiketlemesi bittikten sonra bu sütun,',
        'çotanak sağlık modelinin doğrulamasında kullanılabilir.',
        '',
    ]
    (hedef / 'KILAVUZ.md').write_text('\n'.join(s), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(
        description='Elle etiketlenecek çeşitli alt küme seçer')
    ap.add_argument('kaynak', help='Görüntü klasörü (alt klasörler taranır)')
    ap.add_argument('--urun', default='cilek')
    ap.add_argument('--ad', default='organ_etiketleme')
    ap.add_argument('--adet', type=int, default=400)
    ap.add_argument('--netlik-alt', type=float, default=NETLIK_ALT_YUZDELIK,
                    dest='netlik_alt', help='Bu yüzdeliğin altı seçime girmez')
    ap.add_argument('--tohum', type=int, default=0)
    ap.add_argument('--kuru', action='store_true')
    a = ap.parse_args()

    kaynak = Path(a.kaynak)
    if not kaynak.exists():
        print(f'❌ Yok: {kaynak}')
        return 1

    siniflar = organ_siniflari(a.urun)
    if not siniflar:
        print(f"⛔ configs/urunler/{a.urun}/modeller.yaml içinde "
              "rol: organ olan model yok. Sınıf sırası oradan alınır.")
        return 1

    yollar = sorted(p for p in kaynak.rglob('*')
                    if p.suffix.lower() in GORUNTU_UZANTI)
    if not yollar:
        print(f'❌ {kaynak} içinde görüntü yok.')
        return 1

    print('=' * 72)
    print(f'KAYNAK: {kaynak}   ({len(yollar)} görüntü)')
    print('=' * 72)
    print(f'  organ sınıfları ({a.urun}): ' +
          ', '.join(f'{i}:{s}' for i, s in enumerate(siniflar)))

    # Kaynak klasör adı = sağlık sınıfı (varsa) — seçimde denge için
    etiket = [p.parent.name for p in yollar]

    print('\n--- NETLİK ---')
    n = np.array([netlik(p) for p in yollar])
    esik = float(np.percentile(n, a.netlik_alt))
    uygun = np.nonzero(n >= esik)[0]
    print(f'  Laplace varyansı: medyan {np.median(n):.0f}, '
          f'alt %{a.netlik_alt:.0f} eşiği {esik:.0f}')
    print(f'  seçime giren: {len(uygun)} / {len(yollar)}')

    print('\n--- ÇEŞİTLİLİK SEÇİMİ (en uzak nokta) ---')
    bit = np.array([dhash(yollar[i]) for i in uygun])
    paketli = np.packbits(bit, axis=1)

    # Sınıf başına orantılı kota — küçük sınıf ezilmesin
    gruplar = defaultdict(list)
    for yerel, kures in enumerate(uygun):
        gruplar[etiket[kures]].append(yerel)
    secili_yerel = []
    for ad, uyeler in sorted(gruplar.items()):
        kota = max(1, round(a.adet * len(uyeler) / len(uygun)))
        alt = np.array(uyeler)
        sec = en_uzak_ornekle(paketli[alt], min(kota, len(alt)), a.tohum)
        secili_yerel += [int(alt[i]) for i in sec]
        print(f'  {ad:<24} {len(sec):>4} / {len(uyeler)} kare')
    secim = [int(uygun[i]) for i in secili_yerel]

    d = _POP[np.bitwise_xor(paketli[secili_yerel][:, None, :],
                            paketli[secili_yerel][None, :, :])].sum(2)
    np.fill_diagonal(d, 999)
    tum = _POP[np.bitwise_xor(paketli[0], paketli)].sum(1)
    print(f'  seçilenler arası en yakın mesafe: medyan {np.median(d.min(1)):.0f} bit')
    print(f'  (tüm veride rastgele çift medyanı ≈ {np.median(tum):.0f} bit)')

    if a.kuru:
        print('\n(--kuru: hiçbir şey yazılmadı)')
        return 0

    hedef = KOK / 'datasets' / a.urun / a.ad
    if hedef.exists():
        shutil.rmtree(hedef)
    (hedef / 'images').mkdir(parents=True)
    (hedef / 'labels').mkdir(parents=True)
    with (hedef / 'secim_kaydi.csv').open('w', newline='', encoding='utf-8') as f:
        y = csv.writer(f)
        y.writerow(['dosya', 'kaynak_yol', 'saglik_sinifi', 'netlik'])
        for i in secim:
            p = yollar[i]
            shutil.copy2(p, hedef / 'images' / p.name)
            # Kaynak göreli verilmiş olabilir; KOK'e göreliye çevirmeyi dene
            try:
                kayit = str(p.resolve().relative_to(KOK))
            except ValueError:
                kayit = str(p)
            y.writerow([p.name, kayit, etiket[i], f'{n[i]:.0f}'])

    (hedef / 'data.yaml').write_text(
        '# Organ etiketleme — SINIF SIRASI DEGISTIRILEMEZ.\n'
        f'# Sira configs/urunler/{a.urun}/modeller.yaml organ modelinden alindi;\n'
        '# degistirilirse scripts/model_kur.py kurulumu reddeder.\n'
        '#\n'
        "# 'path' anahtari BILEREK YAZILMAZ: yoksa yaml'in kendi klasoru kok\n"
        '# olur ve dataset tasinabilir kalir.\n'
        + yaml.dump({'train': 'images', 'nc': len(siniflar),
                     'names': {i: s for i, s in enumerate(siniflar)}},
                    allow_unicode=True, sort_keys=False),
        encoding='utf-8')
    kilavuz_yaz(hedef, siniflar, kaynak, secim, len(yollar), esik)

    print(f'\n✅ Yazıldı: {hedef}')
    print(f'   images/  {len(secim)} kare')
    print('   labels/  BOŞ — etiketleyici dolduracak')
    print(f'📄 {hedef / "KILAVUZ.md"} — sınıf tanımları ve kurallar')
    return 0


if __name__ == '__main__':
    sys.exit(main())
