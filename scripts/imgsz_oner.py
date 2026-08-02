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


def etiket_satiri(parcalar):
    """YOLO etiket satırı → (sinif_id, cx, cy, w, h) veya None.

    İKİ BİÇİM VARDIR ve karıştırılırsa ölçüm sessizce çöp üretir:

        kutu     : sinif cx cy w h                      (5 alan)
        poligon  : sinif x1 y1 x2 y2 x3 y3 ...          (7+ alan, TEK)

    Poligon biçimi Ultralytics'in segmentasyon etiketidir. Tespit
    eğitiminde Ultralytics onu KUTUYA ÇEVİRİR (segments2boxes), o yüzden
    eğitim doğru çalışır — ama satırı 5 alan varsayan bir ölçüm, 2.
    noktanın koordinatlarını genişlik/yükseklik sanır.

    ÖLÇÜLDÜ: datasets/cilek/organ_detection satırlarının %68'i poligon.
    Bu fonksiyon eklenmeden önce o dataset'in bütün kutu ölçümleri
    yanlıştı (docs'a yazılan sayılar dahil).
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
        return (sinif, cx, cy, w, h) if w > 0 and h > 0 else None
    if len(sayilar) >= 6 and len(sayilar) % 2 == 0:
        xs, ys = sayilar[0::2], sayilar[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w, h = x1 - x0, y1 - y0
        return (sinif, (x0 + x1) / 2, (y0 + y1) / 2, w, h) if w > 0 and h > 0 \
            else None
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
    # ALAN TESPİTİ için: bu dataset saha görüntüsü mü, stüdyo/makro tek
    # nesne çekimi mi? Karar boru hattına bağlanıp bağlanmayacağını belirler
    # (bkz. docs/HATA-YONETIMI.md § 2.6).
    kutu_sayilari, merkez_kacikligi, kutu_alanlari = [], [], []
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
        bu_goruntude = 0
        for satir in etiket.read_text(encoding='utf-8', errors='ignore').splitlines():
            cozum = etiket_satiri(satir.split())
            if cozum is None:
                continue
            _, cx, cy, w, h = cozum
            # Kutunun KISA kenarı belirleyicidir: ince uzun bir lezyon
            # kısa kenarı görünmez olduğunda tespit edilemez.
            kutu_paylari.append(min(w, h))
            kutu_px.append(min(w * gen, h * yuk))
            bu_goruntude += 1
            # Kadraj merkezinden sapma. Stüdyo çekiminde nesne hep ortadadır;
            # sahada dağılır.
            merkez_kacikligi.append(max(abs(cx - 0.5), abs(cy - 0.5)))
            kutu_alanlari.append(w * h)
        kutu_sayilari.append(bu_goruntude)

    return {
        'goruntu_sayisi': len(goruntuler),
        'orneklenen': len(secilen),
        'okunamayan': okunamayan,
        'uzun_kenar': sorted(uzun_kenarlar),
        'kutu_payi': sorted(kutu_paylari),
        'kutu_px': sorted(kutu_px),
        'kutu_sayisi': sorted(kutu_sayilari),
        'merkez_kacikligi': sorted(merkez_kacikligi),
        'kutu_alani': sorted(kutu_alanlari),
    }


# Alan sınıflandırma eşikleri — MEVCUT DATASET'LER ÖLÇÜLEREK seçildi,
# uydurulmadı. 400 görüntü örneklenerek:
#
#   dataset               kutuMax  kutu>1%  merkez~   alan%   alan
#   findik_kalite               1       0%    0.036    4.8%   stüdyo
#   cilek/bocek_teshis         14      12%    0.089   18.7%   makro
#   ─────────────────────────────────────────────────────────────────
#   cilek/organ_detection       8      24%    0.196   32.7%   saha
#   cilek/fruit_disease         7      29%    0.171   22.3%   saha
#   cilek/leaf_disease         11      37%    0.229   17.7%   saha
#   cilek/fruit_ripeness       30      60%    0.227    2.6%   saha
#
# Ayıran iki sinyal MERKEZ SAPMASI ve ÇOK-KUTULU ORANI: ikisinde de
# boşluk geniş (0.089↔0.171 ve %12↔%24).
#
# KUTU ALANI AYIRT ETMİYOR — böcek %18.7 iken saha organ %32.7. İlk
# denemede alan eşiği kullanıldı ve organ_detection'ı yanlışlıkla "makro"
# saydı; ölçüm bunu yakaladı, eşik atıldı.
TEKIL_MERKEZ_ESIGI = 0.15   # kutular kadraj ortasına bu kadar yakınsa
TEKIL_COKLU_ORANI = 0.20    # ve görüntülerin bu kadarından azı çok kutuluysa


def alan_tespiti(o: dict) -> dict:
    """Saha görüntüsü mü, stüdyo/makro tek nesne çekimi mi?

    ROI boru hattına bağlanacak model SAHA verisiyle eğitilmelidir. Stüdyo
    verisi `rol: tekil, tetik: []` ile ayrı akışta durmalıdır — yoksa model
    hiç görmediği bir ölçekte çalıştırılır (ölçülen iki olay:
    bocek_teshis ve hazelnut detection v9; docs/HATA-YONETIMI.md § 2.6).

    Bu bir SEZGİSEL UYARIDIR, karar değil: sınırda kalan bir pakette
    görüntülere bakıp kendiniz karar verin.
    """
    ks, mk, ka = o.get('kutu_sayisi'), o.get('merkez_kacikligi'), o.get('kutu_alani')
    if not ks or not mk:
        return {}
    kutu_maks = ks[-1]
    coklu_oran = sum(1 for k in ks if k > 1) / len(ks)
    merkez_med = _yuzdelik(mk, 0.5)

    # Tek kutudan fazlası HİÇ yoksa tartışma yok: her görüntüde tam bir nesne.
    tek_nesne = kutu_maks <= 1
    ortalanmis = merkez_med < TEKIL_MERKEZ_ESIGI and coklu_oran < TEKIL_COKLU_ORANI
    tekil = tek_nesne or ortalanmis
    return {
        'kutu_medyan': _yuzdelik(ks, 0.5), 'kutu_maks': kutu_maks,
        'coklu_oran': coklu_oran, 'merkez_medyan': merkez_med,
        'alan_medyan': _yuzdelik(ka, 0.5) if ka else 0.0,
        'alan': ('stüdyo/tek nesne' if tek_nesne
                 else 'makro/yakın çekim' if tekil else 'saha'),
        'boru_hattina_uygun': not tekil,
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

    a = alan_tespiti(o)
    if a:
        print()
        print('  --- ALAN TESPİTİ (boru hattına bağlanır mı?) ---')
        print(f'  Görüntü başına kutu : medyan {a["kutu_medyan"]:.0f}, '
              f'maksimum {a["kutu_maks"]}  '
              f'(%{a["coklu_oran"] * 100:.0f}\'i çok kutulu)')
        print(f'  Merkezden sapma     : medyan {a["merkez_medyan"]:.3f} '
              f'(0 = kadraj ortası, saha ≈ 0.20)')
        print(f'  Kutu alanı          : medyan karenin %{a["alan_medyan"] * 100:.1f}\'i'
              '   (ayırt edici DEĞİL, bilgi amaçlı)')
        print(f'  → ALAN: {a["alan"]}')
        if a['boru_hattina_uygun']:
            print('  ✅ Saha verisi görünüyor — ROI boru hattına bağlanabilir.')
        else:
            print('  ⛔ ROI BORU HATTINA BAĞLAMAYIN.')
            print(f'     Ölçüm "{a["alan"]}" diyor: nesne kadraja ortalanmış,')
            print('     görüntülerin çoğunda tek bulgu var. Boru hattı ise uzman')
            print('     modele bahçe fotoğrafından kesilmiş ROI verir — model')
            print('     hiç görmediği bir ölçekte çalışır.')
            print('     modeller.yaml → rol: tekil, tetik: []  (yapısal kilit)')
            print('     Ölçülen emsaller: docs/HATA-YONETIMI.md § 2.6')

    print()
    print(f'  {"imgsz":>6}  {"küçük nesne":>12}  {"durum":<26} {"RAM önbellek":>12}')
    print('  ' + '-' * 66)
    for x in s['tablo']:
        # DİKKAT: "büyütme" tek başına "işe yaramaz" demek DEĞİLDİR.
        # Büyütme yeni bilgi eklemez, ama YOLO'nun tespit ızgarası eğitim
        # pikselinde SABİTTİR (8 px adım). Kaynakta 3 px olan bir lezyon
        # 320'de yarım hücre, 1024'te ~1,5 hücre kaplar — ağ ikincisinde
        # görebilir. Küçük nesneli setlerde yüksek imgsz bu yüzden yaygındır.
        if x['buyutme'] and x['yeterli']:
            durum = 'büyütme — ama küçük nesneye yarar'
        elif x['buyutme']:
            durum = 'büyütme — yine de yetersiz'
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
                print(f'     ℹ️ {mevcut}, kaynak çözünürlüğün ({s["kaynak_medyan"]:.0f} px) '
                      'ÜSTÜNDE — yeni bilgi eklenmiyor, AMA nesneler ağın')
                print('        sabit ızgarasına göre büyür; çok küçük nesnelerde bu')
                print('        recall\'ı artırabilir. Kesin cevap ölçmekle bulunur:')
                print(f'        aynı dataset\'i {s["imgsz"]} ve {mevcut} ile eğitip')
                print('        mAP karşılaştırın (scripts/model_karsilastir.py).')
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
