"""Dataset'i ölçüp eğitim çözünürlüğünü (imgsz) önerir.

NEDEN GEREKLİ?
    imgsz her dataset için 1024 yazılıydı. Böcek dataset'i 416x416 makro
    fotoğraftan oluşuyor: 1024'e BÜYÜTÜLEREK eğitiliyordu. Büyütme bilgi
    eklemez — sadece 6 kat hesap, 6 kat RAM önbelleği ve 6 kat süre.
    Ölçülen: epoch başına 37 sn; 416 ile ~9 sn olurdu.

    Ters yön daha tehlikeli: saha fotoğrafı 4000x3000 ise ve lezyon
    görüntünün %1'i ise, 640'a küçültünce lezyon 6 piksele iner ve model
    onu ASLA öğrenemez. Eğitim sorunsuz görünür, mAP düşük çıkar, sebebi
    anlaşılmaz.

ASIL ÖLÇÜT: NESNENİN PİKSEL BOYU
    "Görüntü kaç piksel" yanlış sorudur. Doğru soru: bu imgsz'de en küçük
    nesneler kaç piksel kalıyor?

    YOLO'nun en ince tespit katmanı 8 piksel adımlıdır (P3). Bir nesnenin
    güvenilir bulunabilmesi için o ızgarada en az ~2 hücre kaplaması,
    yani ~16 piksel olması gerekir. Altındaki nesneler öğrenilemez.

    Bu yüzden betik etiket kutularının GERÇEK piksel boyunu hesaplar ve
    "küçük nesnelerin %90'ı 16 pikselin üstünde kalsın" kuralıyla en KÜÇÜK
    yeterli imgsz'i seçer. Küçük imgsz = hızlı eğitim + az RAM.

KULLANIM
    python scripts/imgsz_oner.py datasets/cilek/bocek_teshis
    python scripts/imgsz_oner.py --hepsi              # tüm ürün dataset'leri
    python scripts/imgsz_oner.py <dizin> --ornek 500  # daha çok örnekle
"""

import argparse
import random
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent

# YOLO imgsz'i 32'nin katı olmalı (5 kez yarılanan omurga).
ADAYLAR = (320, 416, 512, 640, 768, 896, 1024, 1280, 1536)

# P3 katmanı 8 piksel adımlıdır; nesne o ızgarada en az ~2 hücre kaplamalı.
EN_KUCUK_NESNE_PX = 16

# Nesnelerin bu oranı eşiğin üstünde kalmalı. %100 istemek en uçtaki birkaç
# minik kutu yüzünden gereksiz yere devasa imgsz seçtirirdi.
KAPSAMA = 0.90

GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _boyut_oku(yol: Path):
    """Görüntünün (genişlik, yükseklik) değeri — mümkünse tam çözmeden."""
    try:
        from PIL import Image
        with Image.open(yol) as g:
            return g.size
    except Exception:
        pass
    try:
        import cv2
        import numpy as np
        veri = np.fromfile(str(yol), dtype=np.uint8)   # Türkçe yol güvenli
        g = cv2.imdecode(veri, cv2.IMREAD_REDUCED_COLOR_8)
        if g is None:
            return None
        y, x = g.shape[:2]
        return x * 8, y * 8            # 1/8 okundu, geri ölçekle
    except Exception:
        return None


def olc(kok: Path, ornek: int = 300, tohum: int = 0) -> dict:
    """Dataset'i örnekleyerek görüntü ve kutu boyutlarını ölçer."""
    goruntuler = []
    for bolum in ('train', 'valid', 'val', 'test'):
        d = kok / bolum / 'images'
        if d.is_dir():
            goruntuler += [p for p in d.iterdir()
                           if p.suffix.lower() in GORUNTU_UZANTI]
    if not goruntuler:
        return {}

    rng = random.Random(tohum)
    secilen = rng.sample(goruntuler, min(ornek, len(goruntuler)))

    uzun_kenarlar, kutu_paylari, kutu_px = [], [], []
    okunamayan = 0
    for g in secilen:
        boyut = _boyut_oku(g)
        if not boyut:
            okunamayan += 1
            continue
        gen, yuk = boyut
        uzun_kenarlar.append(max(gen, yuk))

        etiket = Path(str(g.parent).replace('images', 'labels')) / (g.stem + '.txt')
        if not etiket.exists():
            continue
        for satir in etiket.read_text(encoding='utf-8', errors='ignore').splitlines():
            p = satir.split()
            if len(p) < 5:
                continue
            try:
                w, h = float(p[3]), float(p[4])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            # Kutunun KISA kenarı belirleyicidir: ince uzun bir lezyon
            # kısa kenarı görünmez olduğunda tespit edilemez.
            kutu_paylari.append(min(w, h))
            kutu_px.append(min(w * gen, h * yuk))

    return {
        'goruntu_sayisi': len(goruntuler),
        'orneklenen': len(secilen),
        'okunamayan': okunamayan,
        'uzun_kenar': sorted(uzun_kenarlar),
        'kutu_payi': sorted(kutu_paylari),
        'kutu_px': sorted(kutu_px),
    }


def _yuzdelik(dizi, oran):
    if not dizi:
        return 0.0
    i = min(len(dizi) - 1, max(0, int(round(oran * (len(dizi) - 1)))))
    return dizi[i]


def oner(o: dict) -> dict:
    """Ölçümden imgsz önerir."""
    if not o or not o['uzun_kenar']:
        return {}

    kaynak = _yuzdelik(o['uzun_kenar'], 0.5)          # tipik uzun kenar
    kaynak_min = o['uzun_kenar'][0]

    # Küçük nesnelerin sınırı: en küçük %10'un üst ucu
    kucuk_pay = _yuzdelik(o['kutu_payi'], 1 - KAPSAMA) if o['kutu_payi'] else 0.0

    yeterli = []
    for a in ADAYLAR:
        px = kucuk_pay * a
        yeterli.append({'imgsz': a, 'kucuk_nesne_px': px,
                        'yeterli': px >= EN_KUCUK_NESNE_PX,
                        'buyutme': a > kaynak})

    # En KÜÇÜK yeterli aday; yoksa en büyüğü
    uygun = [x for x in yeterli if x['yeterli']]
    secim = uygun[0]['imgsz'] if uygun else ADAYLAR[-1]

    # Kaynağın üstüne çıkma: büyütmek bilgi eklemez
    tavan = max(320, int(kaynak // 32) * 32)
    kirpildi = secim > tavan
    if kirpildi:
        secim = tavan

    # MUHAFAZAKÂR seçenek: kaynak çözünürlüğün kendisi.
    # Yukarıdaki kural EN KÜÇÜK yeterli boyu seçer; hızlıdır ama küçük
    # nesnelerde payı dardır. Kaynak çözünürlükte eğitmek hiçbir detayı
    # atmaz — yalnızca daha yavaştır. Hangisinin daha iyi mAP verdiği
    # dataset'e bağlıdır; ikisini de gösterip kararı kullanıcıya bırakıyoruz.
    guvenli = min(tavan, 1024)

    # Kaynak GÖRÜNTÜDE nesne zaten küçükse sorun imgsz değil VERİDİR.
    kaynak_nesne_px = _yuzdelik(o['kutu_px'], 1 - KAPSAMA) if o['kutu_px'] else 0.0

    return {
        'imgsz': secim,
        'guvenli': guvenli,
        'kaynak_medyan': kaynak,
        'kaynak_min': kaynak_min,
        'kucuk_nesne_payi': kucuk_pay,
        'kucuk_nesne_px': kucuk_pay * secim,
        'kaynak_nesne_px': kaynak_nesne_px,
        'veri_sorunu': 0 < kaynak_nesne_px < EN_KUCUK_NESNE_PX,
        'tablo': yeterli,
        'kaynakla_sinirlandi': kirpildi,
        'nesne_yok': not o['kutu_payi'],
    }


def ram_gb(goruntu_sayisi: int, imgsz: int) -> float:
    """RAM önbelleği tahmini — görüntüler AÇILMIŞ halde tutulur."""
    return goruntu_sayisi * imgsz * imgsz * 3 / 1e9


def rapor(ad: str, o: dict, s: dict, mevcut: int = None):
    print('=' * 74)
    print(f'{ad}   ({o["goruntu_sayisi"]} görüntü, {o["orneklenen"]} örneklendi)')
    print('=' * 74)
    if o['okunamayan']:
        print(f'  ⚠️ {o["okunamayan"]} görüntü okunamadı, ölçüme katılmadı')

    uk = o['uzun_kenar']
    print(f'  Görüntü uzun kenarı : medyan {_yuzdelik(uk, 0.5):.0f} px '
          f'(en küçük {uk[0]:.0f}, en büyük {uk[-1]:.0f})')

    if s.get('nesne_yok'):
        print('  ⚠️ Etiket kutusu bulunamadı — öneri yalnızca görüntü boyutuna dayanıyor.')
    else:
        kp = o['kutu_px']
        print(f'  Kutu kısa kenarı    : medyan {_yuzdelik(kp, 0.5):.0f} px '
              f'(kaynak çözünürlükte)')
        print(f'  En küçük %10 nesne  : görüntünün %{s["kucuk_nesne_payi"] * 100:.1f}\'i '
              f'= kaynakta {s["kaynak_nesne_px"]:.0f} px')

    print()
    print(f'  {"imgsz":>6}  {"küçük nesne":>12}  {"durum":<26} {"RAM önbellek":>12}')
    print('  ' + '-' * 66)
    for x in s['tablo']:
        if x['buyutme']:
            durum = 'büyütme — bilgi eklemez'
        elif x['yeterli']:
            durum = 'yeterli'
        else:
            durum = f'küçük nesneler kaybolur (<{EN_KUCUK_NESNE_PX}px)'
        isaret = ' ←' if x['imgsz'] == s['imgsz'] else '  '
        print(f'  {x["imgsz"]:>6}  {x["kucuk_nesne_px"]:>9.0f} px  {durum:<26} '
              f'{ram_gb(o["goruntu_sayisi"], x["imgsz"]):>9.1f} GB{isaret}')

    print()
    print(f'  ÖNERİ  (hızlı)   : imgsz = {s["imgsz"]}')
    if s['guvenli'] != s['imgsz']:
        print(f'  ALTERNATİF (güvenli): imgsz = {s["guvenli"]}  '
              f'— kaynak çözünürlük, hiçbir detay atılmaz')
        print(f'     Hızlı seçenek ~{(s["guvenli"] / s["imgsz"]) ** 2:.1f} kat daha çabuk '
              'ama küçük nesnelerde payı dardır.')
        print('     Kararsızsanız: ilk eğitimi HIZLI ile yapın, mAP yetersizse')
        print('     GÜVENLİ ile tekrarlayın — fark ölçüyle görülür, tahminle değil.')
    if s['kaynakla_sinirlandi']:
        print(f'     (kaynak çözünürlük {s["kaynak_medyan"]:.0f} px ile sınırlandı — '
              'daha büyüğü büyütme olurdu)')

    if mevcut and mevcut != s['imgsz']:
        oran = (mevcut / s['imgsz']) ** 2
        if mevcut > s['imgsz']:
            print(f'     Şu anki {mevcut} → {s["imgsz"]}: ~{oran:.1f} kat daha hızlı, '
                  f'RAM {ram_gb(o["goruntu_sayisi"], mevcut):.1f} → '
                  f'{ram_gb(o["goruntu_sayisi"], s["imgsz"]):.1f} GB')
            if mevcut > s['kaynak_medyan']:
                print(f'     ⚠️ {mevcut}, kaynak çözünürlüğün ({s["kaynak_medyan"]:.0f} px) '
                      'ÜSTÜNDE — görüntüler büyütülüyor, bilgi eklenmiyor.')
        else:
            print(f'     ⚠️ Şu anki {mevcut} DÜŞÜK: küçük nesneler '
                  f'{s["kucuk_nesne_payi"] * mevcut:.0f} piksele iner '
                  f'(sınır {EN_KUCUK_NESNE_PX}) — model onları öğrenemez.')
    if s.get('veri_sorunu'):
        print()
        print(f'  ⛔ SORUN imgsz DEĞİL, VERİDE: en küçük %10 kutu KAYNAK görüntüde '
              f'zaten {s["kaynak_nesne_px"]:.0f} piksel.')
        print(f'     Sınır {EN_KUCUK_NESNE_PX} px. Görüntüyü büyütmek bu kutuları')
        print('     büyütmez, yalnızca bulanıklaştırır. Üç olasılık:')
        print('       1. Bu kutular hatalı/artık etiket — temizlenmeli')
        print('          (kontrol: python scripts/etiket_temizle.py)')
        print('       2. Lezyonlar gerçekten çok küçük — daha YAKIN çekim gerekir')
        print('       3. Kaynak görüntüler zaten küçültülmüş — orijinalinden')
        print('          yeniden dışa aktarın')
    print()


def main():
    ap = argparse.ArgumentParser(description="Dataset'i ölçüp imgsz önerir")
    ap.add_argument('dizin', nargs='?', help='Dataset kökü (data.yaml burada)')
    ap.add_argument('--hepsi', action='store_true',
                    help='datasets/<urun>/ altındaki tüm dataset\'ler')
    ap.add_argument('--urun', default='cilek')
    ap.add_argument('--ornek', type=int, default=300, help='Örneklenecek görüntü')
    ap.add_argument('--mevcut', type=int, default=None,
                    help='Şu anki imgsz (karşılaştırma için)')
    a = ap.parse_args()

    if a.hepsi:
        kok = KOK / 'datasets' / a.urun
        hedefler = [d for d in sorted(kok.iterdir())
                    if d.is_dir() and (d / 'data.yaml').exists()] if kok.is_dir() else []
        if not hedefler:
            print(f'❌ {kok} altında dataset yok.')
            return 1
    elif a.dizin:
        hedefler = [Path(a.dizin)]
    else:
        ap.print_help()
        return 1

    mevcut = a.mevcut
    if mevcut is None:
        try:
            import yaml
            cfg = yaml.safe_load(
                (KOK / 'configs' / 'train_config.yaml').read_text(encoding='utf-8'))
            mevcut = int(cfg.get('imgsz', 0)) or None
        except Exception:
            mevcut = None

    for d in hedefler:
        if not d.is_dir():
            print(f'❌ Yok: {d}')
            continue
        o = olc(d, a.ornek)
        if not o:
            print(f'❌ {d} içinde <bölüm>/images bulunamadı.')
            continue
        rapor(d.name, o, oner(o), mevcut)
    return 0


if __name__ == '__main__':
    sys.exit(main())
