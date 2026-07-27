"""Sahadan gelen görüntüleri modelle ön-etiketleyip önceliklendirir (aktif öğrenme).

NEDEN?
    Eğitim bittikten sonra modeli sahada kullanırsınız. Yeni görüntüleri
    eğitime katmanın en verimli yolu HEPSİNİ elle etiketlemek değildir:

    1) Model her görüntüye tahmin üretir → bunlar "ön-etiket" olarak kaydedilir.
       Uzman sıfırdan çizmez, sadece düzeltir (3-5 kat hızlı).
    2) Modelin ZORLANDIĞI görüntüler önceliklidir. Zaten %95 güvenle doğru
       bildiği kareyi etiketlemek modele yeni bir şey öğretmez; düşük güvenli
       veya hiç tespit üretmediği kareler öğrenme değeri en yüksek olanlardır.
       (Aktif öğrenme: aynı etiketleme emeğiyle daha fazla kazanım.)

ÇIKTI YAPISI
    <output>/
    ├── incele/          ← ÖNCE BUNLARI etiketleyin (model zorlanmış)
    │   ├── images/
    │   └── labels/      (ön-etiketler, YOLO formatı — düzeltilecek)
    ├── otomatik/        ← model emin; örnekleme yaparak doğrulayın
    │   ├── images/
    │   └── labels/
    ├── tespit_yok/      ← hiç tespit yok: ya sağlıklı ya da KAÇIRILMIŞ hastalık
    │   └── images/
    └── rapor.csv        ← her görüntü için tespit sayısı, güven değerleri, sınıflar

Usage:
    python scripts/collect_field_data.py \
        --model runs/train/strawberry_exp/weights/best.pt \
        --images saha_fotograflari/ \
        --output saha_2026_07/
"""

import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def triage(n_det: int, min_conf: float, review_threshold: float) -> str:
    """Görüntüyü etiketleme önceliğine göre sınıflandırır.

    - tespit_yok : model hiçbir şey bulamadı. Sağlıklı bitki olabilir (değerli
      background örneği) ya da hastalığı KAÇIRMIŞ olabilir — mutlaka bakılmalı.
    - incele     : en düşük güven eşiğin altında → model kararsız, öğrenme değeri yüksek
    - otomatik   : tüm tespitler yüksek güvenli → örnekleme ile doğrulamak yeterli
    """
    if n_det == 0:
        return 'tespit_yok'
    return 'incele' if min_conf < review_threshold else 'otomatik'


def write_label(path: Path, rows: List[tuple]) -> None:
    """YOLO formatında ön-etiket yazar: 'cls x y w h' (normalize)."""
    with open(path, 'w', encoding='utf-8') as f:
        for cls, x, y, w, h in rows:
            f.write(f'{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')


def collect(model_path: str, images_dir: str, output_dir: str,
            conf: float, review_threshold: float, imgsz: int) -> bool:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics yüklü değil: pip install ultralytics")
        return False

    src = Path(images_dir)
    out = Path(output_dir)
    if not src.exists():
        logger.error(f'Görüntü dizini bulunamadı: {src}')
        return False

    images = sorted(p for p in src.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        logger.error(f'{src} içinde görüntü yok')
        return False

    for grup in ('incele', 'otomatik'):
        (out / grup / 'images').mkdir(parents=True, exist_ok=True)
        (out / grup / 'labels').mkdir(parents=True, exist_ok=True)
    (out / 'tespit_yok' / 'images').mkdir(parents=True, exist_ok=True)

    logger.info(f'Model yükleniyor: {model_path}')
    model = YOLO(model_path)
    names = model.names

    sayac: Dict[str, int] = {'incele': 0, 'otomatik': 0, 'tespit_yok': 0}
    sinif_sayaci: Dict[str, int] = {}
    satirlar = []

    logger.info(f'{len(images)} görüntü işleniyor (conf={conf}, inceleme eşiği={review_threshold})...')
    for i, img in enumerate(images, 1):
        r = model(str(img), conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes = r.boxes

        rows, confs, siniflar = [], [], []
        for b in boxes:
            x, y, w, h = b.xywhn[0].tolist()
            c = int(b.cls[0])
            rows.append((c, x, y, w, h))
            confs.append(float(b.conf[0]))
            siniflar.append(names[c])
            sinif_sayaci[names[c]] = sinif_sayaci.get(names[c], 0) + 1

        min_conf = min(confs) if confs else 0.0
        grup = triage(len(rows), min_conf, review_threshold)
        sayac[grup] += 1

        shutil.copy2(img, out / grup / 'images' / img.name)
        if grup != 'tespit_yok':
            write_label(out / grup / 'labels' / f'{img.stem}.txt', rows)

        satirlar.append({
            'dosya': img.name,
            'grup': grup,
            'tespit_sayisi': len(rows),
            'min_guven': f'{min_conf:.3f}',
            'ort_guven': f'{sum(confs)/len(confs):.3f}' if confs else '0.000',
            'siniflar': ';'.join(sorted(set(siniflar))),
        })

        if i % 100 == 0:
            logger.info(f'  {i}/{len(images)}...')

    with open(out / 'rapor.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        w.writeheader()
        w.writerows(satirlar)

    toplam = len(images)
    logger.info(f'\n✅ {toplam} görüntü işlendi → {out}')
    logger.info(f"   🔍 incele     : {sayac['incele']:>5}  (ÖNCE bunları düzeltin — model kararsız)")
    logger.info(f"   ✅ otomatik   : {sayac['otomatik']:>5}  (örnekleme ile doğrulayın)")
    logger.info(f"   ⬜ tespit_yok : {sayac['tespit_yok']:>5}  (sağlıklı mı, kaçırılmış hastalık mı?)")
    if sinif_sayaci:
        logger.info('\n📊 Tahmin edilen sınıf dağılımı:')
        for k, v in sorted(sinif_sayaci.items(), key=lambda x: -x[1]):
            logger.info(f'   {v:>6}  {k}')
    logger.info(f'\n📄 Ayrıntılı rapor: {out / "rapor.csv"}')
    logger.info('\n📝 Sonraki adım: incele/ klasörünü Roboflow\'a yükleyip (görüntü + label) '
                'etiketleri düzeltin, sonra merge_datasets.py ile ana dataset\'e katın.')
    return True


def main():
    ap = argparse.ArgumentParser(
        description='Saha görüntülerini modelle ön-etiketleyip etiketleme önceliğine ayırır')
    ap.add_argument('--model', required=True, help='Eğitilmiş model (best.pt)')
    ap.add_argument('--images', required=True, help='Saha görüntüleri dizini')
    ap.add_argument('--output', required=True, help='Çıktı dizini')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='Tespit güven eşiği (varsayılan 0.25)')
    ap.add_argument('--review-threshold', type=float, default=0.55,
                    help='Bu güvenin altında tespiti olan görüntü incele/ klasörüne gider '
                         '(varsayılan 0.55)')
    ap.add_argument('--imgsz', type=int, default=1024, help='Inference çözünürlüğü')
    args = ap.parse_args()

    ok = collect(args.model, args.images, args.output,
                 args.conf, args.review_threshold, args.imgsz)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
