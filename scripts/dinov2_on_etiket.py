"""DINOv2 ile ADAY organ etiketi üretir (ön-etiketleme).

╔══════════════════════════════════════════════════════════════════════╗
║ DINOv2 KUTU ÜRETMEZ. Tespit başlığı yoktur; omurga modelidir.        ║
║ Bu betik kutuyu DINOv2'nin PATCH GÖMÜLERİNDEN türetir:               ║
║                                                                       ║
║   görüntü → ViT patch belirteçleri (14 px ızgara)                    ║
║          → her patch'i sınıf prototipine ata (kosinüs)               ║
║          → maske → bağlantılı bileşen → kutu                         ║
║                                                                       ║
║ Çıktı NİHAİ ETİKET DEĞİL, ADAYDIR. `labels/` yerine `labels_aday/`   ║
║ altına yazılır ve insan onayından geçmeden eğitime GİRMEZ.           ║
╚══════════════════════════════════════════════════════════════════════╝

NEDEN ADAY?
    Ölçüm yapılmadan otomatik etiketi eğitime vermek, bu projede
    tekrar tekrar yakaladığımız "sessiz hata" desenidir: model hata
    vermez, yalnızca yanlışı öğrenir. Aday etiketler insan tarafından
    düzeltilir; kazanç, sıfırdan çizmek yerine düzeltmektir.

İKİ KİP
    1) TOHUMSUZ (--tohum-etiket verilmezse)
       Sınıf bilgisi YOKTUR. Patch gömülerinin ilk temel bileşeni ile
       ön plan/arka plan ayrılır ve SINIFSIZ aday kutu üretilir
       (hepsi ID 0). Sınıfı insan atar.
       Kullanım: "nerede nesne var" sorusunu hızlandırır.

    2) PROTOTİPLİ (--tohum-etiket <dizin>)
       Elle etiketlenmiş birkaç düzine kareden sınıf prototipi çıkarılır
       (kutu içindeki patch gömülerinin ortalaması). Sonra her patch en
       yakın prototipe atanır → SINIFLI aday kutu.
       Önerilen: en az 20-30 tohum kare, her sınıftan örnek içermeli.

ÇALIŞTIRMA YERİ
    torch + transformers gerekir. Bu depoda kurulu DEĞİL; Colab'da veya
    Docker imajında çalıştırın:
        pip install torch transformers pillow opencv-python-headless

Kullanım:
    # tohumsuz
    python scripts/dinov2_on_etiket.py datasets/findik/organ_etiketleme \
        --urun findik

    # elle etiketlenmiş tohumlarla
    python scripts/dinov2_on_etiket.py datasets/findik/organ_etiketleme \
        --urun findik --tohum-etiket datasets/findik/organ_tohum
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

KOK = Path(__file__).resolve().parents[1]
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

VARSAYILAN_MODEL = 'facebook/dinov2-small'
# ViT-S/14 → girdi kenarı 14'ün katı olmalı. 518 = 14*37 → 37x37 patch.
# Daha büyüğü daha ince kutu verir ama bellek/süre karesel artar.
GIRDI_KENARI = 518
PATCH = 14

# Bir aday kutunun kabul edilmesi için gereken en küçük alan (kare oranı).
# YOLO tespit ızgarası 8 px adımlıdır; 518'lik girdide tek patch 14 px'e
# denk gelir, tek patch'lik bileşen gürültüdür.
EN_KUCUK_ALAN = 0.004        # karenin %0.4'ü
EN_AZ_PATCH = 3


def organ_siniflari(urun: str) -> list:
    p = KOK / 'configs' / 'urunler' / urun / 'modeller.yaml'
    if not p.exists():
        return []
    kutuk = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    for tanim in kutuk.values():
        if (tanim or {}).get('rol') == 'organ':
            return list(tanim.get('siniflar') or [])
    return []


# ─────────────────────────────────────────────────────────────────────────
# Omurga — ayrı tutuldu ki testler sahte omurga enjekte edebilsin.
# ─────────────────────────────────────────────────────────────────────────

def omurga_yukle(model_adi: str = VARSAYILAN_MODEL):
    """DINOv2'yi yükler ve (goruntuler)->patch gömüleri fonksiyonu döner."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as e:
        raise SystemExit(
            f'⛔ {e.name} kurulu değil. DINOv2 için gerekli:\n'
            '   pip install torch transformers\n'
            'Bu depoda kurulu değildir; Colab veya Docker imajında çalıştırın.')

    islemci = AutoImageProcessor.from_pretrained(model_adi)
    model = AutoModel.from_pretrained(model_adi).eval()
    aygit = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(aygit)
    print(f'  omurga: {model_adi}  aygıt: {aygit}')

    def cikar(pil_goruntuler):
        """→ (N, izgara, izgara, boyut) L2-normalize patch gömüleri."""
        girdi = islemci(images=list(pil_goruntuler), return_tensors='pt',
                        size={'height': GIRDI_KENARI, 'width': GIRDI_KENARI},
                        do_center_crop=False)
        with torch.no_grad():
            cikti = model(**{k: v.to(aygit) for k, v in girdi.items()})
        # İlk belirteç CLS'tir, patch değildir — atılır.
        h = cikti.last_hidden_state[:, 1:, :]
        n, p, d = h.shape
        izgara = int(round(p ** 0.5))
        h = torch.nn.functional.normalize(h, dim=-1)
        return h.reshape(n, izgara, izgara, d).cpu().numpy()

    return cikar


# ─────────────────────────────────────────────────────────────────────────
# Prototip çıkarma ve atama — omurgadan bağımsız, test edilebilir
# ─────────────────────────────────────────────────────────────────────────

def kutu_patchleri(kutu, izgara: int):
    """YOLO kutusu (cx,cy,w,h normalize) → patch ızgarasında dilim."""
    cx, cy, w, h = kutu
    x0 = int(np.floor((cx - w / 2) * izgara))
    x1 = int(np.ceil((cx + w / 2) * izgara))
    y0 = int(np.floor((cy - h / 2) * izgara))
    y1 = int(np.ceil((cy + h / 2) * izgara))
    return (max(0, y0), min(izgara, max(y0 + 1, y1)),
            max(0, x0), min(izgara, max(x0 + 1, x1)))


def prototip_kur(gomuler, etiketler, sinif_sayisi: int):
    """Etiketli kutulardan sınıf başına ortalama gömü.

    gomuler   : (N, izgara, izgara, boyut)
    etiketler : N uzunlukta liste; her öğe [(sinif_id, (cx,cy,w,h)), ...]
    """
    izgara, boyut = gomuler.shape[1], gomuler.shape[3]
    toplam = np.zeros((sinif_sayisi, boyut), dtype=np.float64)
    adet = np.zeros(sinif_sayisi, dtype=np.int64)
    for g, kutular in zip(gomuler, etiketler):
        for sid, kutu in kutular:
            if not 0 <= sid < sinif_sayisi:
                continue
            y0, y1, x0, x1 = kutu_patchleri(kutu, izgara)
            dilim = g[y0:y1, x0:x1].reshape(-1, boyut)
            if len(dilim):
                toplam[sid] += dilim.sum(0)
                adet[sid] += len(dilim)
    gorulen = adet > 0
    proto = np.zeros_like(toplam)
    proto[gorulen] = toplam[gorulen] / adet[gorulen][:, None]
    norm = np.linalg.norm(proto, axis=1, keepdims=True)
    proto = np.divide(proto, norm, out=np.zeros_like(proto), where=norm > 0)
    return proto, adet


def patch_ata(gomu, prototipler, en_az_benzerlik: float):
    """Her patch'i en yakın prototipe atar. Eşiğin altı -1 (arka plan)."""
    izgara, boyut = gomu.shape[0], gomu.shape[2]
    d = gomu.reshape(-1, boyut) @ prototipler.T          # kosinüs (ikisi de normalize)
    en_iyi = d.argmax(1)
    skor = d.max(1)
    en_iyi[skor < en_az_benzerlik] = -1
    return en_iyi.reshape(izgara, izgara), skor.reshape(izgara, izgara)


def on_plan_pca(gomu):
    """Tohumsuz kip: ilk temel bileşenin işareti ön/arka planı ayırır.

    DINO ailesinde bilinen davranış — patch gömülerinin baskın değişimi
    nesne/arka plan ekseninde olur. Hangi işaretin NESNE olduğu garanti
    değil; kenar patch'lerinin çoğunlukla arka plan olduğu varsayılarak
    işaret düzeltilir.
    """
    izgara, boyut = gomu.shape[0], gomu.shape[2]
    x = gomu.reshape(-1, boyut)
    x = x - x.mean(0)
    # İlk tekil vektör = ilk temel bileşen
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    skor = (x @ vt[0]).reshape(izgara, izgara)
    kenar = np.concatenate([skor[0], skor[-1], skor[:, 0], skor[:, -1]])
    if kenar.mean() > skor.mean():
        skor = -skor          # kenar arka plan olmalı → düşük skor
    return skor > 0


def bilesen_kutulari(maske: np.ndarray, en_az_patch: int = EN_AZ_PATCH):
    """İkili patch maskesi → bağlantılı bileşenlerin normalize kutuları."""
    import cv2
    m = maske.astype(np.uint8)
    adet, etiket = cv2.connectedComponents(m, connectivity=4)
    izgara = maske.shape[0]
    kutular = []
    for i in range(1, adet):
        ys, xs = np.nonzero(etiket == i)
        if len(ys) < en_az_patch:
            continue
        # Patch ızgarasından normalize görüntü koordinatına
        x0, x1 = xs.min() / izgara, (xs.max() + 1) / izgara
        y0, y1 = ys.min() / izgara, (ys.max() + 1) / izgara
        w, h = x1 - x0, y1 - y0
        if w * h < EN_KUCUK_ALAN:
            continue
        kutular.append((x0 + w / 2, y0 + h / 2, w, h, len(ys)))
    return kutular


def adaylari_uret(gomu, prototipler=None, en_az_benzerlik=0.55):
    """Bir görüntünün patch gömüsünden aday kutular.

    Dönen: [(sinif_id, cx, cy, w, h, guven)] — tohumsuz kipte sinif_id=0
    ve anlamı "sınıfsız aday"dır.
    """
    if prototipler is None:
        maske = on_plan_pca(gomu)
        return [(0, cx, cy, w, h, 0.0)
                for cx, cy, w, h, _ in bilesen_kutulari(maske)]

    atama, skor = patch_ata(gomu, prototipler, en_az_benzerlik)
    cikti = []
    for sid in range(len(prototipler)):
        maske = atama == sid
        if not maske.any():
            continue
        for cx, cy, w, h, n in bilesen_kutulari(maske):
            # Güven: bu bileşendeki patch'lerin ortalama benzerliği
            guven = float(skor[maske].mean())
            cikti.append((sid, cx, cy, w, h, guven))
    return cikti


# ─────────────────────────────────────────────────────────────────────────

def etiket_oku(yol: Path):
    kutular = []
    if not yol.exists():
        return kutular
    for satir in yol.read_text(encoding='utf-8', errors='ignore').splitlines():
        p = satir.split()
        if len(p) >= 5:
            kutular.append((int(p[0]), tuple(float(x) for x in p[1:5])))
    return kutular


def gorunutuleri_bul(kok: Path):
    d = kok / 'images' if (kok / 'images').is_dir() else kok
    return sorted(p for p in d.rglob('*') if p.suffix.lower() in GORUNTU_UZANTI)


def rapor_yaz(hedef: Path, siniflar, sayim, toplam, kip, tohum_bilgi,
              kutusuz):
    s = [
        '# DINOv2 ön-etiketleme raporu',
        '',
        f'Kip: **{kip}**',
        '',
        '## ⚠️ Bu etiketler ADAYDIR',
        '',
        'Dosyalar `labels_aday/` altındadır, `labels/` altında DEĞİL.',
        'İnsan onayından geçmeden eğitime verilmemelidir. Otomatik etiketi',
        'doğrudan eğitime vermek, bu projede tekrar tekrar yakaladığımız',
        '"sessiz hata" desenidir: model hata vermez, yanlışı öğrenir.',
        '',
        '## Üretilen aday kutular',
        '',
        '| sınıf | kutu |',
        '|---|---|',
    ]
    for i, ad in enumerate(siniflar):
        s.append(f'| `{ad}` | {sayim.get(i, 0)} |')
    s += [
        '',
        f'Görüntü: {toplam}   ·   hiç kutu çıkmayan: {kutusuz}',
        '',
    ]
    if tohum_bilgi:
        s += ['## Prototip tohumları', '',
              '| sınıf | tohum patch sayısı |', '|---|---|']
        for i, ad in enumerate(siniflar):
            n = tohum_bilgi[i] if i < len(tohum_bilgi) else 0
            uyari = '  ⛔ tohum yok' if n == 0 else ''
            s.append(f'| `{ad}` | {n}{uyari} |')
        s += ['',
              'Tohum patch sayısı düşük olan sınıfın prototipi zayıftır;',
              'o sınıfın adaylarına ayrıca dikkat edin.', '']
    else:
        s += [
            '## Sınıfsız kip',
            '',
            'Tohum etiketi verilmediği için sınıf ATANMADI — bütün kutular',
            'ID 0 ile yazıldı ve "burada bir nesne var" demektir.',
            'Sınıfı etiketleyici atar.',
            '',
            'Sınıflı aday için önce 20-30 kareyi elle etiketleyip',
            '`--tohum-etiket` ile verin.',
            '',
        ]
    s += [
        '## Sonraki adım',
        '',
        '```bash',
        '# 1. labels_aday/ dosyalarini etiketleme araciyla ac, DUZELT',
        '# 2. Onaylananlari labels/ altina tasi',
        '# 3. Sizinti ve bolme denetimi',
        'python scripts/harici_paket_duzelt.py <klasor> --urun <urun> \\',
        '    --ad organ_detection --kuru',
        '# 4. Alan tespiti — saha verisi mi?',
        'python scripts/imgsz_oner.py datasets/<urun>/organ_detection',
        '```',
        '',
    ]
    (hedef / 'ON_ETIKET_RAPORU.md').write_text('\n'.join(s), encoding='utf-8')


def main(cikar=None):
    ap = argparse.ArgumentParser(
        description='DINOv2 ile aday organ etiketi üretir')
    ap.add_argument('hedef', help='images/ içeren dataset klasörü')
    ap.add_argument('--urun', default='cilek')
    ap.add_argument('--tohum-etiket', default=None, dest='tohum_etiket',
                    help='Elle etiketlenmiş klasör (images/ + labels/)')
    ap.add_argument('--model', default=VARSAYILAN_MODEL)
    ap.add_argument('--yigin', type=int, default=8, dest='yigin')
    ap.add_argument('--en-az-benzerlik', type=float, default=0.55,
                    dest='en_az_benzerlik')
    ap.add_argument('--kuru', action='store_true')
    a = ap.parse_args()

    hedef = Path(a.hedef)
    if not hedef.exists():
        print(f'❌ Yok: {hedef}')
        return 1
    siniflar = organ_siniflari(a.urun)
    if not siniflar:
        print(f'⛔ configs/urunler/{a.urun}/modeller.yaml içinde '
              'rol: organ olan model yok.')
        return 1

    goruntuler = gorunutuleri_bul(hedef)
    if not goruntuler:
        print(f'❌ {hedef} içinde görüntü yok.')
        return 1

    print('=' * 72)
    print(f'HEDEF: {hedef}   ({len(goruntuler)} görüntü)')
    print('=' * 72)
    print('  sınıflar: ' + ', '.join(f'{i}:{s}' for i, s in enumerate(siniflar)))

    # --kuru omurgayı YÜKLEMEZ: yapılandırmayı torch kurulu olmayan bir
    # makinede de doğrulayabilmek gerekir (bu depoda torch yok).
    if a.kuru:
        print('\n--- KURU KOŞU (model yüklenmedi) ---')
        if a.tohum_etiket:
            tk = Path(a.tohum_etiket)
            tg = gorunutuleri_bul(tk)
            dolu = sum(1 for p in tg
                       if etiket_oku(tk / 'labels' / (p.stem + '.txt')))
            print(f'  tohum: {len(tg)} kare, {dolu} tanesi etiketli')
            if not dolu:
                print('  ⛔ Etiketli tohum yok — prototip kurulamaz.')
                return 1
            print('  kip: prototipli (sınıflı aday)')
        else:
            print('  kip: tohumsuz (SINIFSIZ aday, hepsi ID 0)')
            print('  sınıflı aday için --tohum-etiket verin')
        print(f'  yazılacak: {hedef / "labels_aday"}  ({len(goruntuler)} dosya)')
        print('\n(--kuru: hiçbir şey yazılmadı)')
        return 0

    if cikar is None:
        cikar = omurga_yukle(a.model)
    from PIL import Image

    def gomu_al(yollar):
        cikti = []
        for i in range(0, len(yollar), a.yigin):
            obek = yollar[i:i + a.yigin]
            ims = [Image.open(p).convert('RGB') for p in obek]
            cikti.append(cikar(ims))
            for im in ims:
                im.close()
        return np.concatenate(cikti, 0)

    # --- Prototipler -----------------------------------------------------
    prototipler = None
    tohum_bilgi = None
    if a.tohum_etiket:
        tk = Path(a.tohum_etiket)
        t_goruntuler = gorunutuleri_bul(tk)
        etiket_dizini = tk / 'labels'
        t_etiketler = [etiket_oku(etiket_dizini / (p.stem + '.txt'))
                       for p in t_goruntuler]
        dolu = sum(1 for e in t_etiketler if e)
        print(f'\n--- TOHUM ---\n  {len(t_goruntuler)} kare, {dolu} tanesi etiketli')
        if not dolu:
            print('  ⛔ Hiçbir tohum karede kutu yok. Prototip kurulamaz.')
            return 1
        t_gomu = gomu_al(t_goruntuler)
        prototipler, tohum_bilgi = prototip_kur(t_gomu, t_etiketler,
                                                len(siniflar))
        for i, ad in enumerate(siniflar):
            isaret = '  ⛔ TOHUM YOK' if tohum_bilgi[i] == 0 else ''
            print(f'  {ad:<10} {tohum_bilgi[i]:>6} patch{isaret}')
        if (tohum_bilgi == 0).any():
            print('  ⚠️ Tohumsuz sınıflar için aday üretilemeyecek.')

    kip = 'prototipli (sınıflı)' if prototipler is not None \
        else 'tohumsuz (sınıfsız)'
    print(f'\n--- ADAY ÜRETİMİ — kip: {kip} ---')

    gomuler = gomu_al(goruntuler)
    cikti_dizini = hedef / 'labels_aday'
    cikti_dizini.mkdir(parents=True, exist_ok=True)
    sayim = Counter()
    kutusuz = 0
    for p, g in zip(goruntuler, gomuler):
        adaylar = adaylari_uret(g, prototipler, a.en_az_benzerlik)
        if not adaylar:
            kutusuz += 1
        satir = []
        for sid, cx, cy, w, h, _ in adaylar:
            sayim[sid] += 1
            satir.append(f'{sid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
        (cikti_dizini / (p.stem + '.txt')).write_text(
            '\n'.join(satir) + ('\n' if satir else ''), encoding='utf-8')

    print(f'  aday kutu: {sum(sayim.values())}')
    for i, ad in enumerate(siniflar):
        if sayim.get(i):
            print(f'    {ad:<10} {sayim[i]:>6}')
    print(f'  hiç kutu çıkmayan görüntü: {kutusuz} / {len(goruntuler)}')

    rapor_yaz(hedef, siniflar, sayim, len(goruntuler), kip, tohum_bilgi,
              kutusuz)
    print(f'\n✅ Yazıldı: {cikti_dizini}  (ADAY — labels/ DEĞİL)')
    print(f'📄 {hedef / "ON_ETIKET_RAPORU.md"}')
    print('\n⚠️ Bu etiketler insan onayından geçmeden eğitime VERİLMEZ.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
