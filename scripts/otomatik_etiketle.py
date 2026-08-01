"""Etiketsiz görüntüden ADAY YOLO etiketi: bul → kırp → sınıflandır → yaz.

    etiketsiz görüntü
        → BULUCU        bölgeleri bul (kutu)
        → kırp
        → SINIFLANDIRICI her kırpıntıya sınıf ver
        → eşik altı = unknown
        → labels_aday/*.txt  +  INCELEME.csv
        → İNSAN KONTROLÜ
        → labels/

╔══════════════════════════════════════════════════════════════════════╗
║ İKİ YAYGIN HATA — İKİSİ DE BURADA ENGELLENİR                         ║
║                                                                       ║
║ 1) "COCO'da eğitilmiş YOLOv8 yaprakları bulur"  → BULMAZ.            ║
║    COCO'nun 80 sınıfı arasında yaprak, meyve, dal YOKTUR; bitkiyle   ║
║    ilgili tek sınıf 'potted plant'tır (saksı bitkisi, bütün olarak). ║
║    Bu betik buluculuğa verilen modelin sınıflarını OKUR ve işe       ║
║    yarar sınıf yoksa DURUR.                                          ║
║                                                                       ║
║ 2) "DINOv2 gömüsünü CLIP metin promptuyla karşılaştır"  → OLMAZ.     ║
║    DINOv2'nin metin kodlayıcısı yoktur ve gömü uzayı CLIP'inkiyle    ║
║    hizalı değildir. Metin promptu istiyorsan görüntü kodlayıcısı da  ║
║    CLIP olmalı (--siniflandirici clip). DINOv2 kullanacaksan metin   ║
║    yerine ÖRNEK verirsin (--siniflandirici dinov2 --tohum ...).      ║
╚══════════════════════════════════════════════════════════════════════╝

BULUCU SEÇENEKLERİ
    --bulucu yolo:<yol.pt>
        Eğitilmiş bir YOLO tespit modeli. Sınıfları okunur ve raporlanır.
        Örn. models/cilek/organ.pt — çilek organ modeli. Fındıkta ne kadar
        işe yaradığı ÖLÇÜLMELİDİR; --olc ile kaç bölge bulduğu basılır.

    --bulucu acik-sozluk:<yol.pt>
        Açık sözlüklü tespit (YOLO-World / YOLOE). Metinle sınıf verilir:
        --bulucu-promptlari "leaf,hazelnut husk,branch"
        COCO'nun kapalı sınıf listesi sorununu çözen doğru araç budur.

ÇALIŞTIRMA YERİ
    ultralytics + torch (+ transformers CLIP için) gerekir; bu depoda
    kurulu DEĞİL. Colab veya Docker imajında çalıştırın.

Kullanım:
    python scripts/otomatik_etiketle.py datasets/findik/organ_etiketleme \
        --urun findik \
        --bulucu acik-sozluk:yoloe-11l-seg.pt \
        --bulucu-promptlari "leaf,hazelnut husk,branch" \
        --siniflandirici clip --bilinmeyen-esigi 0.50
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

KOK = Path(__file__).resolve().parents[1]
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# COCO'nun bitkiyle uzaktan ilgili tek sınıfı. Bir bulucu modelin sınıf
# listesi buna benziyorsa organ tespiti YAPAMAZ.
COCO_ISARETI = {'person', 'car', 'dog', 'potted plant', 'chair', 'bottle'}
# Bölge olarak kabul edilebilecek sınıf adları (küçük harf, kısmi eşleşme).
#
# 'plant' BİLEREK YOK: COCO'nun 'potted plant' sınıfına takılıyordu ve
# COCO modelini organ bulucusu sayıyordu. Test bunu yakaladı
# (test_coco_modeli_bulucu_olarak_reddedilir). Saksı bitkisi bütün olarak
# kutulanır — organ değildir.
ORGAN_IPUCU = ('leaf', 'yaprak', 'fruit', 'nut', 'husk', 'zuruf', 'branch',
               'dal', 'flower', 'cicek', 'çiçek', 'stem', 'govde', 'gövde',
               'cluster', 'cotanak', 'çotanak')

BILINMEYEN = 'unknown'


def urun_siniflari(urun: str, rol: str = 'organ') -> list:
    p = KOK / 'configs' / 'urunler' / urun / 'modeller.yaml'
    if not p.exists():
        return []
    kutuk = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    for t in kutuk.values():
        if (t or {}).get('rol') == rol:
            return list(t.get('siniflar') or [])
    return []


def bulucu_siniflarini_denetle(siniflar) -> tuple:
    """Bu modelin sınıfları organ tespiti için işe yarar mı?

    Dönen: (uygun_mu, gerekce)
    """
    kucuk = {str(s).lower() for s in siniflar}
    if not kucuk:
        return False, 'model sınıf listesi boş'

    # COCO denetimi ÖNCE: güçlü bir olumsuz sinyaldir ve organ ipucu
    # denetimi tek bir yanlış eşleşmeyle atlatılabilir.
    if len(kucuk & COCO_ISARETI) >= 3:
        return False, (
            'bu bir COCO modeli. COCO\'nun 80 sınıfı arasında yaprak/meyve/'
            'dal YOKTUR — bitkiyle ilgili tek sınıf "potted plant"tır ve o da '
            'bitkiyi bütün olarak kutular. Organ tespiti yapamaz.')

    ise_yarar = [s for s in kucuk if any(i in s for i in ORGAN_IPUCU)]
    if ise_yarar:
        return True, 'organ sınıfı bulundu: ' + ', '.join(sorted(ise_yarar))
    return False, ('sınıflar arasında organ yok: ' + ', '.join(sorted(kucuk)[:12]))


def kirp(im, kutu, pay: float = 0.12):
    """Normalize kutu → PIL kırpıntı, kenardan pay bırakarak.

    Pay bırakmak önemlidir: sınıflandırıcı lezyonun BAĞLAMINI da görür
    (yaprak kenarı, damar). Boru hattındaki ROI kırpma da %12 pay kullanır.
    """
    g, y = im.size
    cx, cy, w, h = kutu
    w, h = w * (1 + pay), h * (1 + pay)
    x0 = max(0, int((cx - w / 2) * g))
    y0 = max(0, int((cy - h / 2) * y))
    x1 = min(g, int((cx + w / 2) * g))
    y1 = min(y, int((cy + h / 2) * y))
    if x1 <= x0 or y1 <= y0:
        return None
    return im.crop((x0, y0, x1, y1))


def kutu_normalize(xyxy, genislik, yukseklik):
    x0, y0, x1, y1 = xyxy
    return ((x0 + x1) / 2 / genislik, (y0 + y1) / 2 / yukseklik,
            (x1 - x0) / genislik, (y1 - y0) / yukseklik)


def karar_ver(skorlar, siniflar, esik, ayrim_esigi=0.0):
    """En yüksek skor eşiği geçmiyorsa BİLİNMEYEN.

    Ayrıca ilk iki skor birbirine çok yakınsa da bilinmeyen sayılır:
    kararsız argmax, yanlış etiketten daha sinsidir çünkü güven yüksek
    görünür.
    """
    sirali = np.argsort(skorlar)[::-1]
    en_iyi, ikinci = skorlar[sirali[0]], (skorlar[sirali[1]]
                                          if len(sirali) > 1 else 0.0)
    if en_iyi < esik:
        return BILINMEYEN, float(en_iyi), 'eşik altı'
    if ayrim_esigi and (en_iyi - ikinci) < ayrim_esigi:
        return BILINMEYEN, float(en_iyi), 'ilk iki skor çok yakın'
    return siniflar[sirali[0]], float(en_iyi), ''


def rapor_yaz(hedef, bulucu_bilgi, siniflandirici, sayim, bilinmeyen,
              toplam_kutu, kutusuz, esik):
    s = [
        '# Otomatik etiketleme raporu', '',
        '## ⚠️ Bu etiketler ADAYDIR', '',
        'Dosyalar `labels_aday/` altındadır. İnsan kontrolünden geçmeden',
        '`labels/` altına taşınmaz ve eğitime verilmez.',
        '`INCELEME.csv` en düşük güvenden başlayarak sıralıdır — kontrolü',
        'oradan başlatın, en çok hata orada birikir.', '',
        '## Kurulum', '',
        f'- **Bulucu:** {bulucu_bilgi}',
        f'- **Sınıflandırıcı:** {siniflandirici}',
        f'- **Bilinmeyen eşiği:** {esik}', '',
        '## Üretilen aday kutular', '',
        '| sınıf | kutu |', '|---|---|',
    ]
    for ad, n in sorted(sayim.items(), key=lambda x: -x[1]):
        s.append(f'| `{ad}` | {n} |')
    s += [
        f'| **{BILINMEYEN}** | **{bilinmeyen}** |', '',
        f'Toplam kutu: {toplam_kutu}   ·   hiç kutu çıkmayan görüntü: {kutusuz}',
        '',
    ]
    if toplam_kutu and bilinmeyen / toplam_kutu > 0.4:
        s += ['> ⛔ Kutuların %40\'ından fazlası `unknown`. Sınıflandırıcı bu',
              '> alanda çalışmıyor demektir. Eşiği düşürmek ÇÖZÜM DEĞİLDİR —',
              '> yanlış etiketleri görünür olmaktan çıkarır sadece.',
              '> Etiketli veriyle eğitim gerekir: `scripts/dinov2_egit.py`', '']
    s += [
        '## Sonraki adım', '',
        '```bash',
        '# 1. INCELEME.csv sirasiyla aday kutulari duzelt (CVAT/Roboflow)',
        '# 2. Onaylananlari labels/ altina tasi',
        '# 3. Sizinti ve bolme denetimi',
        'python scripts/harici_paket_duzelt.py <klasor> --urun <urun> \\',
        '    --ad organ_detection --kuru',
        '# 4. Alan tespiti',
        'python scripts/imgsz_oner.py datasets/<urun>/organ_detection',
        '```', '',
    ]
    (hedef / 'OTOMATIK_ETIKET_RAPORU.md').write_text('\n'.join(s),
                                                     encoding='utf-8')


def bulucu_yukle(tanim: str, promptlar, olc: bool):
    """→ (fonksiyon(PIL)->[(xyxy, guven, sinif_adi)], bilgi_metni)"""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            '⛔ ultralytics kurulu değil.\n   pip install ultralytics\n'
            '   Bu depoda kurulu değildir; Colab/Docker\'da çalıştırın.')

    kip, _, yol = tanim.partition(':')
    if not yol:
        raise SystemExit(f'⛔ Bulucu biçimi: yolo:<yol.pt> veya '
                         f'acik-sozluk:<yol.pt>   (verilen: {tanim!r})')
    model = YOLO(yol)

    if kip == 'acik-sozluk':
        if not promptlar:
            raise SystemExit('⛔ acik-sozluk için --bulucu-promptlari zorunlu.')
        adlar = [p.strip() for p in promptlar.split(',') if p.strip()]
        if not hasattr(model, 'set_classes'):
            raise SystemExit(
                f'⛔ {yol} açık sözlüklü değil (set_classes yok).\n'
                '   YOLO-World veya YOLOE ağırlığı kullanın.')
        model.set_classes(adlar)
        bilgi = f'acik-sozluk {Path(yol).name} → {", ".join(adlar)}'
        sinif_adlari = adlar
    else:
        sinif_adlari = list(model.names.values())
        uygun, gerekce = bulucu_siniflarini_denetle(sinif_adlari)
        print(f'  bulucu sınıfları: {", ".join(map(str, sinif_adlari[:12]))}'
              + ('…' if len(sinif_adlari) > 12 else ''))
        print(f'  denetim: {gerekce}')
        if not uygun:
            raise SystemExit(
                '⛔ Bu bulucu organ tespiti için uygun değil.\n'
                f'   {gerekce}\n'
                '   Açık sözlüklü model kullanın:\n'
                '     --bulucu acik-sozluk:yoloe-11l-seg.pt '
                '--bulucu-promptlari "leaf,husk,branch"')
        bilgi = f'yolo {Path(yol).name} → {", ".join(map(str, sinif_adlari))}'

    def bul(im):
        s = model.predict(im, verbose=False)[0]
        cikti = []
        for k in s.boxes:
            sid = int(k.cls[0])
            ad = (sinif_adlari[sid] if sid < len(sinif_adlari) else str(sid))
            cikti.append((k.xyxy[0].tolist(), float(k.conf[0]), str(ad)))
        return cikti

    return bul, bilgi


def main():
    ap = argparse.ArgumentParser(
        description='Etiketsiz görüntüden aday YOLO etiketi üretir')
    ap.add_argument('hedef', help='images/ içeren klasör')
    ap.add_argument('--urun', default='cilek')
    ap.add_argument('--bulucu', required=True,
                    help='yolo:<yol.pt> | acik-sozluk:<yol.pt>')
    ap.add_argument('--bulucu-promptlari', default=None,
                    dest='bulucu_promptlari')
    ap.add_argument('--siniflandirici', default='clip',
                    choices=('clip', 'dinov2', 'yok'),
                    help="'yok' = bulucunun sınıfını kullan")
    ap.add_argument('--tohum', default=None,
                    help='dinov2 az-atış için: sınıf klasörleri olan dizin')
    ap.add_argument('--prompt-dosyasi', default=None, dest='prompt_dosyasi')
    ap.add_argument('--bilinmeyen-esigi', type=float, default=0.50,
                    dest='bilinmeyen_esigi')
    ap.add_argument('--ayrim-esigi', type=float, default=0.02,
                    dest='ayrim_esigi',
                    help='İlk iki skor bu kadar yakınsa unknown')
    ap.add_argument('--pay', type=float, default=0.12)
    ap.add_argument('--kuru', action='store_true')
    a = ap.parse_args()

    hedef = Path(a.hedef)
    goruntu_dizini = hedef / 'images' if (hedef / 'images').is_dir() else hedef
    goruntuler = sorted(p for p in goruntu_dizini.rglob('*')
                        if p.suffix.lower() in GORUNTU_UZANTI)
    if not goruntuler:
        print(f'❌ {goruntu_dizini} içinde görüntü yok.')
        return 1

    # Hedef sınıflar: sınıflandırıcı varsa ürün kütüğünden, yoksa bulucudan
    hedef_siniflar = urun_siniflari(a.urun) if a.siniflandirici != 'yok' else []

    print('=' * 76)
    print(f'HEDEF: {hedef}   ({len(goruntuler)} görüntü)')
    print('=' * 76)
    print(f'  bulucu         : {a.bulucu}')
    print(f'  sınıflandırıcı : {a.siniflandirici}')
    if hedef_siniflar:
        print('  hedef sınıflar : ' + ', '.join(hedef_siniflar))
    print(f'  bilinmeyen eşiği: {a.bilinmeyen_esigi}  '
          f'(ayrım {a.ayrim_esigi})')

    if a.siniflandirici == 'dinov2' and not a.tohum:
        print('\n⛔ DINOv2 metinle sınıflandırma YAPAMAZ — metin kodlayıcısı')
        print('   yoktur ve gömü uzayı CLIP\'inkiyle hizalı değildir.')
        print('   Ya --tohum <sınıf klasörleri olan dizin> verin (az-atış),')
        print('   ya da --siniflandirici clip kullanın (metin promptu).')
        return 1

    if a.kuru:
        print('\n(--kuru: model yüklenmedi, yazım yok)')
        print(f'  yazılacaktı: {hedef / "labels_aday"}')
        return 0

    from PIL import Image

    bul, bulucu_bilgi = bulucu_yukle(a.bulucu, a.bulucu_promptlari, True)

    # --- Sınıflandırıcı --------------------------------------------------
    sinifla = None
    if a.siniflandirici == 'clip':
        import json as _json
        sa = __import__('importlib.util', fromlist=['util'])
        s = sa.spec_from_file_location('saz', KOK / 'scripts' /
                                       'sifir_atis_siniflandir.py')
        m = sa.module_from_spec(s)
        s.loader.exec_module(m)
        promptlar = dict(m.VARSAYILAN_PROMPT)
        if a.prompt_dosyasi:
            promptlar.update(_json.loads(
                Path(a.prompt_dosyasi).read_text(encoding='utf-8')))
        eksik = [c for c in hedef_siniflar if c not in promptlar]
        if eksik:
            print(f'  ⚠️ Prompt tanımsız: {eksik} → sınıf adı kullanılacak')

        def sinifla(kirpintilar):
            _, skor = m.clip_sinifla(kirpintilar, hedef_siniflar, promptlar,
                                     m.CLIP_MODEL, 16)
            return skor
    elif a.siniflandirici == 'dinov2':
        sa = __import__('importlib.util', fromlist=['util'])
        s = sa.spec_from_file_location('saz', KOK / 'scripts' /
                                       'sifir_atis_siniflandir.py')
        m = sa.module_from_spec(s)
        s.loader.exec_module(m)
        t_yollar, t_etiket = m.goruntuleri_topla(Path(a.tohum), '')
        print(f'  tohum: {len(t_yollar)} kare, '
              f'{len(set(t_etiket))} sınıf')
        hedef_siniflar = sorted(set(t_etiket))
        t_gomu = m.dinov2_gomu(t_yollar, m.DINOV2_MODEL, 16)
        proto = m.prototip_kur(t_gomu, t_etiket, hedef_siniflar)

        def sinifla(kirpintilar):
            g = m.dinov2_gomu(kirpintilar, m.DINOV2_MODEL, 16)
            return g @ proto.T

    # --- Çalıştır ---------------------------------------------------------
    aday_dizini = hedef / 'labels_aday'
    aday_dizini.mkdir(parents=True, exist_ok=True)
    sayim = Counter()
    bilinmeyen = kutusuz = toplam = 0
    inceleme = []

    print('\n--- ADAY ÜRETİMİ ---')
    for n, p in enumerate(goruntuler, 1):
        with Image.open(p) as im:
            im = im.convert('RGB')
            g, y = im.size
            bulunan = bul(im)
            if not bulunan:
                kutusuz += 1
                (aday_dizini / (p.stem + '.txt')).write_text('', encoding='utf-8')
                continue
            kutular = [kutu_normalize(k, g, y) for k, _, _ in bulunan]
            if sinifla is None:
                etiketler = [(ad, gv, '') for _, gv, ad in bulunan]
            else:
                kirpintilar, gecerli = [], []
                for i, kutu in enumerate(kutular):
                    kr = kirp(im, kutu, a.pay)
                    if kr is not None:
                        kirpintilar.append(kr)
                        gecerli.append(i)
                skorlar = sinifla(kirpintilar) if kirpintilar else np.zeros((0, 1))
                etiketler = [(BILINMEYEN, 0.0, 'kırpılamadı')] * len(kutular)
                for j, i in enumerate(gecerli):
                    etiketler[i] = karar_ver(skorlar[j], hedef_siniflar,
                                             a.bilinmeyen_esigi, a.ayrim_esigi)
                for kr in kirpintilar:
                    kr.close()

        satir = []
        for kutu, (ad, guven, not_) in zip(kutular, etiketler):
            toplam += 1
            if ad == BILINMEYEN:
                bilinmeyen += 1
            else:
                sayim[ad] += 1
            if ad in hedef_siniflar:
                sid = hedef_siniflar.index(ad)
                satir.append('%d %.6f %.6f %.6f %.6f' % (sid, *kutu))
            inceleme.append([p.name, ad, f'{guven:.4f}', not_,
                             '%.4f %.4f %.4f %.4f' % kutu])
        (aday_dizini / (p.stem + '.txt')).write_text(
            '\n'.join(satir) + ('\n' if satir else ''), encoding='utf-8')
        if n % 50 == 0:
            print(f'  {n}/{len(goruntuler)}  kutu {toplam}  '
                  f'unknown {bilinmeyen}')

    # En düşük güvenden başlayarak sırala: hata orada birikir.
    inceleme.sort(key=lambda r: float(r[2]))
    with (hedef / 'INCELEME.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['goruntu', 'sinif', 'guven', 'not', 'kutu'])
        w.writerows(inceleme)

    print(f'\n  toplam kutu: {toplam}')
    for ad, k in sorted(sayim.items(), key=lambda x: -x[1]):
        print(f'    {ad:<22} {k:>6}')
    oran = bilinmeyen / toplam if toplam else 0
    print(f'    {BILINMEYEN:<22} {bilinmeyen:>6}  (%{oran * 100:.1f})')
    print(f'  hiç kutu çıkmayan görüntü: {kutusuz} / {len(goruntuler)}')
    if oran > 0.4:
        print('\n  ⛔ Kutuların %40\'ından fazlası unknown. Sınıflandırıcı bu')
        print('     alanda çalışmıyor. Eşiği DÜŞÜRMEYİN — yanlış etiketleri')
        print('     yalnızca görünmez yapar. Etiketli veriyle eğitin:')
        print('     python scripts/dinov2_egit.py <veri>')

    rapor_yaz(hedef, bulucu_bilgi, a.siniflandirici, sayim, bilinmeyen,
              toplam, kutusuz, a.bilinmeyen_esigi)
    print(f'\n✅ {aday_dizini}  (ADAY — labels/ DEĞİL)')
    print(f'📄 {hedef / "INCELEME.csv"}  — en düşük güvenden sıralı')
    print(f'📄 {hedef / "OTOMATIK_ETIKET_RAPORU.md"}')
    print('\n⚠️ İnsan kontrolünden geçmeden eğitime VERİLMEZ.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
