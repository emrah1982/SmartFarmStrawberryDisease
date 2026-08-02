"""Grounding DINO ile ADAY organ etiketi (+ istege bagli SAM2 sikilastirma).

    etiketsiz gorunti
      -> Grounding DINO (metin promptu ile acik sozluklu tespit)
      -> temizlik: NMS, alan filtresi, ic ice kutu ayiklama
      -> [SAM2] kutuyu maskeye oturtup SIKILASTIR
      -> labels_organ/     organ kutulari (SAGLIK BILMEZ)
      -> labels_hastalik/  yalnizca hastalikli organlar (AYRI KATMAN)
      -> INCELEME.csv + KILAVUZ.md
      -> INSAN DUZELTMESI -> labels/

╔══════════════════════════════════════════════════════════════════════╗
║ NEDEN IKI AYRI ETIKET KATMANI?                                        ║
║                                                                       ║
║ "Hastalikli organi da etiketle" istegi dogrudur, ama ayni dosyaya     ║
║ `leaf` ve `diseased_leaf` yazilirsa:                                  ║
║   1. Ayni yaprak IKI kutuya girer, NMS'te birbirini bastirir.        ║
║   2. Hastalik karari ORGAN modeline yuklenir; uzman model            ║
║      gereksizlesir ve mimari duz tek-modele coker.                    ║
║                                                                       ║
║ Cozum: AYNI kutular, IKI katman.                                      ║
║   labels_organ/     Leaf/Nut/Husk/Branch/Flower  -> organ modeli     ║
║   labels_hastalik/  diseased_*                   -> uzman model      ║
║ Ikisi de ayni goruntuye ait, birbirine karismaz.                      ║
║ Bkz. docs/MIMARI.md § "Ucuncu fayda"                                  ║
╚══════════════════════════════════════════════════════════════════════╝

HASTALIK KATMANI NASIL KURULUR?
    Otomatik teshis GUVENILIR DEGIL (olculdu: CLIP sifir-atis ince taneli
    hastalik ayriminda tabana gore +0.03, yazi-tura. docs/EGITIM.md 2.6).
    Bu yuzden hastalik katmani IKI kosula birden baglidir:
      1. Goruntu duzeyi etiketi 'Diseased' olmali (dosya adindan gelir)
      2. Kirpintinin hastalik skoru esigi gecmeli
    'Healthy' karelerde hastalik kutusu HIC uretilmez. Uretilenler de
    adaydir; hastaligin ADI verilmez, yalnizca "bu organda bozulma var".

CALISTIRMA YERI
    torch + transformers gerekir. CPU'da grounding-dino-tiny ~5.5 sn/kare
    (olculdu) -> 4310 kare ~7 saat. Colab GPU'da ~0.1 sn/kare.
    Notebook: Findik_Otomatik_Etiketleme.ipynb

Kullanim:
    python scripts/on_etiket_gdino.py datasets/findik/cotanak_saglik \
        --urun findik --ad organ_gdino --hastalik --sinir 50
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

KOK = Path(__file__).resolve().parents[1]
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

GDINO_MODEL = 'IDEA-Research/grounding-dino-base'
SAM_MODEL = 'facebook/sam2.1-hiera-small'
CLIP_MODEL = 'openai/clip-vit-base-patch32'

# ── PROMPT ── ÖLÇÜLEREK seçildi (4 karede 3 kalıp denendi):
#   'hazelnut. leaf. branch.'                 -> etiketler PARÇALANDI
#                                                ('hazelnut leaf', 'leaf branch')
#   'a cluster of hazelnuts in green husk...'  -> daha kötü
#                                                ('a clusternut', '##s husk')
#   'green hazelnut cluster. green leaf. brown woody branch.'  -> TEMİZ ✅
#
# Grounding DINO promptu küçük harf ve NOKTA ile ayrılmalı; uzun tamlamalar
# belirteç sınırında bölünüp geri gelmeyen etiket üretiyor.
#
# Anahtar = prompt cümlesi, değer = kütükteki organ adı.
# 'green hazelnut cluster' -> Husk: bu fotoğraflarda GÖRÜNEN organ zuruftur;
# fındık kabuğu zurufun içindedir. Kabuk açığa çıkmışsa insan Nut'a çevirir.
ORGAN_PROMPT = {
    'green hazelnut cluster': 'Husk',
    'green leaf': 'Leaf',
    'brown woody branch': 'Branch',
}

# ── TEMİZLİK EŞİKLERİ ──
#
# ⚠️ ALAN EŞİĞİ ZAYIF BİR SİNYALDİR — ölçüldü, ilk tahminim yanlıştı.
#
# 21.106 gerçek saha organ kutusu (cilek/organ_detection, dejenere
# tam-kadraj olanlar dışlanarak):
#     Flower  medyan  2.9%  p90  6.0%  max 19.0%
#     Fruit   medyan 28.3%  p90 90.9%  max 98.9%
#     Leaf    medyan 16.2%  p90 66.0%  max 98.9%
#
# Yani MEŞRU bir organ kutusu kadrajın %98'ini kaplayabilir — yakın
# çekimde tek çilek ya da tek yaprak kareyi doldurur. Dolayısıyla alan
# eşiği "dejenere sabit kutu"yu "meşru yakın çekim"den AYIRAMAZ:
#     eşik %64 → gerçek kutuların %17.3'ü de atılırdı
#     eşik %50 → %22.5'i atılırdı
#
# Bu yüzden eşik yalnızca "tüm kareyi kaplayan" tespitleri eler; asıl iş
# NMS ve İÇERME denetimine bırakılır.
EN_BUYUK_ALAN = 0.90     # yalnızca kadrajın tamamını kaplayanları eler
EN_KUCUK_ALAN = 0.004    # çok küçük kutu etiketlenemez (YOLO 8px ızgara)
NMS_IOU = 0.55
ICERME_ESIGI = 0.85      # A, B'nin %85'ini kaplıyorsa ikisi aynı şeydir

HASTALIK_PROMPT = {
    'diseased': ['a diseased hazelnut organ with brown rot and dark lesions',
                 'discolored deformed plant tissue with necrotic spots'],
    'healthy': ['a healthy green plant organ with uniform color',
                'undamaged fresh green plant tissue'],
}
HASTALIK_ESIGI = 0.55    # kırpıntının 'diseased' olasılığı bunu geçmeli


def organ_siniflari(urun: str) -> list:
    """Sınıf sırası modeller.yaml organ modelinden gelir — uydurulmaz."""
    p = KOK / 'configs' / 'urunler' / urun / 'modeller.yaml'
    if not p.exists():
        return []
    kutuk = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    for t in kutuk.values():
        if (t or {}).get('rol') == 'organ':
            return list(t.get('siniflar') or [])
    return []


# ─────────────────────────────────────────────────────────────────────────
# Temizlik — model gerektirmez, test edilebilir
# ─────────────────────────────────────────────────────────────────────────

def iou(a, b):
    """a, b = (x0, y0, x1, y1) normalize."""
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    kesisim = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if kesisim <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return kesisim / (aa + ab - kesisim)


def icerme(kucuk, buyuk):
    """kucuk'un ne kadarı buyuk'un içinde? (0-1)"""
    x0 = max(kucuk[0], buyuk[0]); y0 = max(kucuk[1], buyuk[1])
    x1 = min(kucuk[2], buyuk[2]); y1 = min(kucuk[3], buyuk[3])
    kesisim = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    alan = (kucuk[2] - kucuk[0]) * (kucuk[3] - kucuk[1])
    return kesisim / alan if alan > 0 else 0.0


def temizle(kutular, en_buyuk_alan=EN_BUYUK_ALAN, en_kucuk_alan=EN_KUCUK_ALAN,
            nms_iou=NMS_IOU, icerme_esigi=ICERME_ESIGI):
    """[(xyxy, skor, sinif)] → temizlenmiş liste + atılanların gerekçesi.

    Üç ayrı iş yapar, sırası önemlidir:
      1. ALAN filtresi — yalnızca tüm kareyi kaplayan tespitleri eler.
         Kasıtlı olarak gevşektir: ölçülen gerçek organ kutuları %98'e
         kadar çıkıyor, dar eşik meşru yakın çekimleri de atardı.
      2. NMS — aynı nesneye birden çok kutu.
      3. İÇERME — NMS'in kaçırdığı iç içe kutular. Büyük kutu küçüğü
         %85 kapsıyorsa ikisi aynı nesnedir; NMS bunu yakalamaz çünkü
         IoU düşük kalır (alanlar çok farklı).
    """
    tutulan, atilan = [], []
    aday = []
    for kutu, skor, sinif in kutular:
        alan = max(0.0, kutu[2] - kutu[0]) * max(0.0, kutu[3] - kutu[1])
        if alan > en_buyuk_alan:
            atilan.append((kutu, skor, sinif, 'alan çok büyük (%.2f)' % alan))
        elif alan < en_kucuk_alan:
            atilan.append((kutu, skor, sinif, 'alan çok küçük (%.4f)' % alan))
        else:
            aday.append((kutu, skor, sinif))

    for sinif in sorted({s for _, _, s in aday}):
        grup = sorted([x for x in aday if x[2] == sinif],
                      key=lambda x: -x[1])
        secili = []
        for kutu, skor, s in grup:
            cakisan = next((t for t in secili if iou(kutu, t[0]) > nms_iou), None)
            if cakisan is not None:
                atilan.append((kutu, skor, s, 'NMS (IoU>%.2f)' % nms_iou))
                continue
            ice = next((t for t in secili
                        if icerme(kutu, t[0]) > icerme_esigi
                        or icerme(t[0], kutu) > icerme_esigi), None)
            if ice is not None:
                atilan.append((kutu, skor, s, 'iç içe kutu'))
                continue
            secili.append((kutu, skor, s))
        tutulan += secili
    return tutulan, atilan


def yolo_satiri(kutu, sinif_id):
    x0, y0, x1, y1 = kutu
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    return '%d %.6f %.6f %.6f %.6f' % (sinif_id, cx, cy, w, h)


def saglik_etiketi(dosya_adi: str):
    """Dosya adından görüntü düzeyi sağlık etiketi. Yoksa None."""
    m = re.search(r'(Diseased|Healthy)', dosya_adi, re.IGNORECASE)
    return m.group(1).lower() if m else None


# ─────────────────────────────────────────────────────────────────────────

def gdino_yukle(model_adi, aygit=None):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    isl = AutoProcessor.from_pretrained(model_adi)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_adi).eval()
    aygit = aygit or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(aygit)
    metin = '. '.join(ORGAN_PROMPT) + '.'
    print(f'  Grounding DINO: {model_adi}  aygıt: {aygit}')
    print(f'  prompt: {metin!r}')

    def bul(im, kutu_esigi, metin_esigi):
        g = isl(images=im, text=metin, return_tensors='pt').to(aygit)
        with torch.no_grad():
            cikti = model(**g)
        s = isl.post_process_grounded_object_detection(
            cikti, g.input_ids, threshold=kutu_esigi,
            text_threshold=metin_esigi, target_sizes=[im.size[::-1]])[0]
        etiketler = s.get('text_labels', s.get('labels'))
        gen, yuk = im.size
        out = []
        for kutu, e, skor in zip(s['boxes'], etiketler, s['scores']):
            organ = ORGAN_PROMPT.get(str(e).strip().lower())
            if organ is None:      # parçalanmış etiket — atılır
                continue
            x0, y0, x1, y1 = [float(v) for v in kutu]
            out.append(((max(0.0, x0 / gen), max(0.0, y0 / yuk),
                         min(1.0, x1 / gen), min(1.0, y1 / yuk)),
                        float(skor), organ))
        return out

    return bul


def sam_yukle(model_adi, aygit=None):
    """Kutuyu maskeye oturtup sıkılaştırır. Bulunamazsa None döner."""
    import torch
    try:
        from transformers import AutoProcessor, Sam2Model
    except ImportError:
        return None
    try:
        isl = AutoProcessor.from_pretrained(model_adi)
        model = Sam2Model.from_pretrained(model_adi).eval()
    except Exception as e:
        print(f'  ⚠️ SAM2 yüklenemedi ({type(e).__name__}), sıkılaştırma atlandı')
        return None
    aygit = aygit or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(aygit)
    print(f'  SAM2: {model_adi}  aygıt: {aygit}')

    def sikilastir(im, kutular):
        if not kutular:
            return kutular
        gen, yuk = im.size
        piksel = [[[k[0] * gen, k[1] * yuk, k[2] * gen, k[3] * yuk]
                   for k, _, _ in kutular]]
        try:
            g = isl(images=im, input_boxes=piksel, return_tensors='pt').to(aygit)
            with torch.no_grad():
                c = model(**g, multimask_output=False)
            maskeler = isl.post_process_masks(
                c.pred_masks.cpu(), g['original_sizes'].cpu())[0]
        except Exception:
            return kutular
        yeni = []
        for (kutu, skor, sinif), m in zip(kutular, maskeler):
            m = np.asarray(m).squeeze()
            ys, xs = np.nonzero(m)
            if len(xs) < 16:
                yeni.append((kutu, skor, sinif))
                continue
            yeni.append(((xs.min() / gen, ys.min() / yuk,
                          (xs.max() + 1) / gen, (ys.max() + 1) / yuk),
                         skor, sinif))
        return yeni

    return sikilastir


def clip_yukle(model_adi, aygit=None):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    isl = CLIPProcessor.from_pretrained(model_adi)
    model = CLIPModel.from_pretrained(model_adi).eval()
    aygit = aygit or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(aygit)

    def _al(x):
        return x.pooler_output if hasattr(x, 'pooler_output') else x

    proto = []
    for kalips in HASTALIK_PROMPT.values():
        g = isl(text=kalips, return_tensors='pt', padding=True).to(aygit)
        with torch.no_grad():
            t = _al(model.get_text_features(**g))
        t = torch.nn.functional.normalize(t, dim=-1).mean(0)
        proto.append(torch.nn.functional.normalize(t, dim=-1))
    proto = torch.stack(proto)
    print(f'  CLIP (hastalık katmanı): {model_adi}  aygıt: {aygit}')

    def skor(kirpintilar):
        if not kirpintilar:
            return np.zeros(0)
        g = isl(images=kirpintilar, return_tensors='pt').to(aygit)
        with torch.no_grad():
            v = _al(model.get_image_features(**g))
        v = torch.nn.functional.normalize(v, dim=-1)
        s = (v @ proto.T) * 100.0
        return torch.softmax(s, dim=-1)[:, 0].cpu().numpy()   # 'diseased'

    return skor


def data_yaml_yaz(yol: Path, siniflar, baslik):
    yol.write_text(
        baslik +
        "#\n# 'path' anahtari BILEREK YAZILMAZ: yoksa yaml'in kendi klasoru\n"
        '# kok olur ve dataset tasinabilir kalir (docs/VERI-ALMA.md 1).\n'
        + yaml.dump({'train': 'images', 'nc': len(siniflar),
                     'names': {i: s for i, s in enumerate(siniflar)}},
                    allow_unicode=True, sort_keys=False),
        encoding='utf-8')


def kilavuz_yaz(hedef, organ_siniflar, hastalik_siniflar, sayim, h_sayim,
                toplam, kutusuz, atilan_ozet, hastalik_acik):
    s = [
        '# Grounding DINO ön-etiketleme — düzeltme kılavuzu', '',
        '## ⚠️ Bunlar ADAY etiketlerdir', '',
        'İnsan kontrolünden geçmeden eğitime verilmez.',
        '`INCELEME.csv` en düşük güvenden sıralıdır — kontrolü oradan',
        'başlatın, hata orada birikir.', '',
        '## İki katman — ayrı tutulmalı', '',
        '| klasör | sınıflar | hangi model |',
        '|---|---|---|',
        f'| `labels_organ/` | {", ".join(organ_siniflar)} | organ modeli (YOLO26) |',
    ]
    if hastalik_acik:
        s.append(f'| `labels_hastalik/` | {", ".join(hastalik_siniflar)} '
                 '| uzman model |')
    s += [
        '',
        'Organ katmanı **sağlık bilmez**: sağlıklı yaprak da hastalıklı',
        'yaprak da `Leaf` kutusudur. Sağlık kararı uzman modelin işidir.',
        'Aynı dosyaya `leaf` ve `diseased_leaf` yazılsaydı aynı yaprak iki',
        'kutuya girer, NMS\'te birbirini bastırırdı.',
        'Bkz. `docs/MIMARI.md` § "Üçüncü fayda".', '',
        '## Üretilen kutular', '', '| sınıf | kutu |', '|---|---|',
    ]
    for ad in organ_siniflar:
        s.append(f'| `{ad}` | {sayim.get(ad, 0)} |')
    if hastalik_acik:
        s.append('')
        s.append('| hastalık katmanı | kutu |')
        s.append('|---|---|')
        for ad in hastalik_siniflar:
            s.append(f'| `{ad}` | {h_sayim.get(ad, 0)} |')
    s += ['', f'Görüntü: {toplam} · hiç kutu çıkmayan: {kutusuz}', '',
          '## Temizlikte atılanlar', '', '| gerekçe | adet |', '|---|---|']
    for g, n in atilan_ozet.most_common():
        s.append(f'| {g} | {n} |')
    s += [
        '', '## Düzeltirken', '',
        '1. **Her küme, her yaprak, her dal AYRI kutu.** Model sık sık',
        '   birkaç yaprağı tek kutuda topluyor — bölün.',
        '2. **Eksikleri ekleyin.** Bulma oranı düşüktür; kadraj kenarındaki',
        '   ve arkadaki organları model çoğu zaman atlıyor.',
        '3. **Kutu görünen piksele oturur.** Arkada kalan kısmı tahmin edip',
        '   büyütmeyin.',
        '4. **Kısa kenarı ~16 pikselden küçükse atlayın.** YOLO ızgarası',
        '   8 piksel adımlıdır, altı zaten öğrenilemez.',
        '5. **Arka plandaki bulanık organları silin.** Toprak/gökyüzü',
        '   üzerindeki kutular gürültüdür.',
        '6. **Zuruf mu fındık mı?** Kabuk görünmüyorsa `Husk`. Kabuk açığa',
        '   çıkmışsa `Nut` yapın.',
    ]
    if hastalik_acik:
        s += [
            '7. **Hastalık katmanı yalnızca `Diseased` karelerde üretildi.**',
            '   Hastalığın ADI verilmedi — yalnızca "bu organda bozulma var".',
            '   Otomatik teşhis güvenilir değil (ölçüldü: `docs/EGITIM.md` 2.6).',
        ]
    s += [
        '', '## Sonraki adım', '', '```bash',
        '# 1. labels_organ/ -> duzelt -> labels/',
        '# 2. Sizinti ve bolme denetimi',
        'python scripts/harici_paket_duzelt.py <klasor> \\',
        '    --urun findik --ad organ_detection --kuru',
        '# 3. Alan tespiti',
        'python scripts/imgsz_oner.py datasets/findik/organ_detection',
        '# 4. YOLO26 ile egit (notebook 4-6 hucreleri)',
        '```', '',
    ]
    (hedef / 'KILAVUZ.md').write_text('\n'.join(s), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(
        description='Grounding DINO ile aday organ etiketi üretir')
    ap.add_argument('kaynak', help='Görüntü klasörü veya zip açılmış dizin')
    ap.add_argument('--urun', default='findik')
    ap.add_argument('--ad', default='organ_gdino')
    ap.add_argument('--model', default=GDINO_MODEL)
    ap.add_argument('--kutu-esigi', type=float, default=0.30,
                    dest='kutu_esigi')
    ap.add_argument('--metin-esigi', type=float, default=0.25,
                    dest='metin_esigi')
    ap.add_argument('--sam', action='store_true',
                    help='SAM2 ile kutuları sıkılaştır (yavaş ama daha iyi)')
    ap.add_argument('--hastalik', action='store_true',
                    help='labels_hastalik/ katmanını da üret')
    ap.add_argument('--hastalik-esigi', type=float, default=HASTALIK_ESIGI,
                    dest='hastalik_esigi')
    ap.add_argument('--sinir', type=int, default=0,
                    help='Yalnızca ilk N görüntü (deneme için)')
    ap.add_argument('--kuru', action='store_true')
    a = ap.parse_args()

    kaynak = Path(a.kaynak)
    if not kaynak.exists():
        print(f'❌ Yok: {kaynak}')
        return 1
    organ_siniflar = organ_siniflari(a.urun)
    if not organ_siniflar:
        print(f'⛔ configs/urunler/{a.urun}/modeller.yaml içinde '
              'rol: organ olan model yok.')
        return 1
    bilinmeyen = set(ORGAN_PROMPT.values()) - set(organ_siniflar)
    if bilinmeyen:
        print(f'⛔ Prompt kütükte olmayan organ üretiyor: {bilinmeyen}')
        return 1

    goruntuler = sorted(p for p in kaynak.rglob('*')
                        if p.suffix.lower() in GORUNTU_UZANTI)
    if a.sinir:
        goruntuler = goruntuler[:a.sinir]
    if not goruntuler:
        print(f'❌ {kaynak} içinde görüntü yok.')
        return 1

    hastalik_siniflar = ['diseased_' + s.lower() for s in organ_siniflar]
    saglik = [saglik_etiketi(p.name) for p in goruntuler]
    etiketli = sum(1 for s in saglik if s)

    print('=' * 76)
    print(f'KAYNAK: {kaynak}   ({len(goruntuler)} görüntü)')
    print('=' * 76)
    print('  organ sınıfları : ' + ', '.join(
        f'{i}:{s}' for i, s in enumerate(organ_siniflar)))
    print('  prompt eşlemesi : ' + ', '.join(
        f'{k!r}→{v}' for k, v in ORGAN_PROMPT.items()))
    yok = [s for s in organ_siniflar if s not in ORGAN_PROMPT.values()]
    if yok:
        print(f'  ⚠️ prompt\'u olmayan sınıf: {yok} → kutu üretilmeyecek')
    if a.hastalik:
        print(f'  hastalık katmanı: AÇIK (eşik {a.hastalik_esigi})')
        print(f'  görüntü düzeyi sağlık etiketi bulunan: '
              f'{etiketli}/{len(goruntuler)}')
        if not etiketli:
            print('  ⛔ Dosya adlarında Diseased/Healthy yok. Hastalık katmanı')
            print('     görüntü düzeyi etikete DAYANIR; onsuz üretilmez.')
            return 1

    hedef = KOK / 'datasets' / a.urun / a.ad
    if a.kuru:
        print(f'\n  yazılacaktı: {hedef}')
        print('\n(--kuru: model yüklenmedi)')
        return 0

    from PIL import Image
    bul = gdino_yukle(a.model)
    sikilastir = sam_yukle(SAM_MODEL) if a.sam else None
    hastalik_skoru = clip_yukle(CLIP_MODEL) if a.hastalik else None

    for alt in ('images', 'labels_organ') + (
            ('labels_hastalik',) if a.hastalik else ()):
        (hedef / alt).mkdir(parents=True, exist_ok=True)

    sayim, h_sayim, atilan_ozet = Counter(), Counter(), Counter()
    kutusuz = 0
    inceleme = []
    import shutil
    import time
    t0 = time.time()

    for n, (p, sag) in enumerate(zip(goruntuler, saglik), 1):
        with Image.open(p) as im:
            im = im.convert('RGB')
            ham = bul(im, a.kutu_esigi, a.metin_esigi)
            tutulan, atilan = temizle(ham)
            for *_, gerekce in atilan:
                atilan_ozet[gerekce.split(' (')[0]] += 1
            if sikilastir and tutulan:
                tutulan = sikilastir(im, tutulan)

            h_satir = []
            if a.hastalik and sag == 'diseased' and tutulan:
                gen, yuk = im.size
                kirp = [im.crop((int(k[0] * gen), int(k[1] * yuk),
                                 max(int(k[2] * gen), int(k[0] * gen) + 1),
                                 max(int(k[3] * yuk), int(k[1] * yuk) + 1)))
                        for k, _, _ in tutulan]
                skorlar = hastalik_skoru(kirp)
                for (kutu, _, organ), sk in zip(tutulan, skorlar):
                    if sk >= a.hastalik_esigi:
                        hid = organ_siniflar.index(organ)
                        h_satir.append(yolo_satiri(kutu, hid))
                        h_sayim[hastalik_siniflar[hid]] += 1
                for k in kirp:
                    k.close()

        shutil.copy2(p, hedef / 'images' / p.name)
        satir = []
        for kutu, skor, organ in tutulan:
            sayim[organ] += 1
            satir.append(yolo_satiri(kutu, organ_siniflar.index(organ)))
            inceleme.append([p.name, organ, '%.4f' % skor, sag or '',
                             '%.4f %.4f %.4f %.4f' % kutu])
        if not satir:
            kutusuz += 1
        (hedef / 'labels_organ' / (p.stem + '.txt')).write_text(
            '\n'.join(satir) + ('\n' if satir else ''), encoding='utf-8')
        if a.hastalik:
            (hedef / 'labels_hastalik' / (p.stem + '.txt')).write_text(
                '\n'.join(h_satir) + ('\n' if h_satir else ''),
                encoding='utf-8')

        if n % 25 == 0 or n == len(goruntuler):
            hiz = (time.time() - t0) / n
            kalan = hiz * (len(goruntuler) - n)
            print('  %4d/%d  kutu %d  %.2f sn/kare  kalan ~%.0f dk'
                  % (n, len(goruntuler), sum(sayim.values()), hiz, kalan / 60))

    inceleme.sort(key=lambda r: float(r[2]))
    with (hedef / 'INCELEME.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['goruntu', 'organ', 'guven', 'saglik', 'kutu'])
        w.writerows(inceleme)

    data_yaml_yaz(hedef / 'data.yaml', organ_siniflar,
                  '# ADAY organ etiketleri (Grounding DINO). Sinif sirasi\n'
                  '# configs/urunler/%s/modeller.yaml organ modelinden gelir;\n'
                  '# degistirilirse scripts/model_kur.py kurulumu reddeder.\n'
                  % a.urun)
    if a.hastalik:
        data_yaml_yaz(hedef / 'data_hastalik.yaml', hastalik_siniflar,
                      '# ADAY hastalik katmani. ORGAN katmaniyla ayni kutular,\n'
                      '# ayri dosya. Organ modeline BU siniflar VERILMEZ.\n')
    kilavuz_yaz(hedef, organ_siniflar, hastalik_siniflar, sayim, h_sayim,
                len(goruntuler), kutusuz, atilan_ozet, a.hastalik)

    print(f'\n  organ kutuları: {sum(sayim.values())}')
    for ad in organ_siniflar:
        if sayim.get(ad):
            print(f'    {ad:<10} {sayim[ad]:>6}')
    if a.hastalik:
        print(f'  hastalık kutuları: {sum(h_sayim.values())}')
        for ad in hastalik_siniflar:
            if h_sayim.get(ad):
                print(f'    {ad:<20} {h_sayim[ad]:>6}')
    print(f'  hiç kutu çıkmayan görüntü: {kutusuz} / {len(goruntuler)}')
    print('  temizlikte atılan:')
    for g, k in atilan_ozet.most_common():
        print(f'    {g:<28} {k:>6}')
    print(f'\n✅ {hedef}')
    print(f'📄 {hedef / "KILAVUZ.md"} · {hedef / "INCELEME.csv"}')
    print('\n⚠️ ADAY etiketlerdir. İnsan kontrolünden geçmeden eğitime verilmez.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
