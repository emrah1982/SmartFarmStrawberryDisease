"""
Sınıf hedefli (dengesizlik giderici) offline augmentasyon scripti.

NEDEN GEREKLİ?
    Dataset'te sınıflar dengesiz: Anthracnose Fruit Rot 326 kutu iken
    strawberry_ripe 5.162 kutu (~16 kat fark). ~10 katı aşan dengesizlikte
    model az örnekli sınıfı "görmezden gelmeye" başlar. Tüm dataset'i eşit
    çoğaltmak dengesizliği AYNEN KORUR;
    çözüm, SADECE az örnekli sınıfları içeren görüntüleri çoğaltmaktır.

KURALLAR (script bunları uygular):
    - Sadece TRAIN split'i çoğaltılır (val/test'e dokunulmaz — metrik şişmesin)
    - Kopyalar birebir değil, her biri FARKLI dönüşümlerle üretilir
    - Renk/ton (hue) kaydırma YAPILMAZ — hastalık ayrımı renge dayanır
      (kahverengi leke / gri küf / beyaz külleme)
    - Çıktı AYRI klasöre yazılır: orijinal veri bozulmaz, augment klasörünü
      silip yeniden üretebilirsiniz
    - Bir görüntüde birden çok sınıf varsa en yüksek çarpan uygulanır
      (yaygın sınıflar da bir miktar çoğalır — kabul edilebilir yan etki)

Varsayılan çarpanlar (birleşik 10 sınıf düzenine göre, hedef ~1300-2800 kutu):
    1 Anthracnose Fruit Rot  : 4x      0 Angular Leafspot : 2x
    2 Blossom Blight         : 3x      3 Gray Mold        : 2x
    5 Powdery Mildew Fruit   : 3x      8 semi_ripe        : 2x

Usage:
    # Varsayılan çarpanlarla (configs/strawberry_data.yaml'daki train dizinleri)
    python scripts/augment_by_class.py

    # Özel çarpanlar ve data.yaml'a otomatik ekleme
    python scripts/augment_by_class.py --factors "1:5,2:3,5:3" --update-data-yaml
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Sınıf ID → çarpan (nihai adet ≈ çarpan × orijinal; yani çarpan-1 yeni kopya üretilir)
DEFAULT_FACTORS = {0: 2, 1: 4, 2: 3, 3: 2, 5: 3, 8: 2}


def build_transform():
    """Hastalık teşhisine uygun (renk korumalı) augmentasyon pipeline'ı."""
    import albumentations as A
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                rotate=(-15, 15),
                scale=(0.9, 1.1),
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                p=0.8,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            # Bilinçli olarak YOK: HueSaturationValue, ColorJitter(hue), ChannelShuffle
            # → renk hastalık sinyalidir, bozulmamalı
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                 min_visibility=0.3),
    )


def parse_factors(spec: str) -> Dict[int, int]:
    """'1:4,2:3' biçimindeki çarpan tanımını parse eder."""
    factors = {}
    for part in spec.split(','):
        cls, mult = part.split(':')
        factors[int(cls.strip())] = int(mult.strip())
    return factors


def read_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """YOLO label dosyasını okur ve kutuları görüntü sınırlarına oturtur.

    Roboflow etiketlerinde kenara değen kutular x+w/2 > 1 olacak şekilde taşabilir.
    Merkez/genişliği tek tek kırpmak yetmez — albumentations KÖŞE koordinatlarını
    denetler ve tek geçersiz kutu tüm görüntünün augmentasyonunu iptal eder.
    Bu yüzden köşeler kırpılır, sonra merkez/genişlik yeniden hesaplanır.
    """
    boxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:5])

            x_min, x_max = min(max(x - w / 2, 0.0), 1.0), min(max(x + w / 2, 0.0), 1.0)
            y_min, y_max = min(max(y - h / 2, 0.0), 1.0), min(max(y + h / 2, 0.0), 1.0)
            new_w, new_h = x_max - x_min, y_max - y_min
            if new_w <= 1e-6 or new_h <= 1e-6:
                continue  # tamamen kadraj dışı veya sıfır alanlı kutu
            boxes.append((cls, x_min + new_w / 2, y_min + new_h / 2, new_w, new_h))
    return boxes


def resolve_train_dirs(data_yaml: Path, output_dir: Path) -> List[Path]:
    """data.yaml'daki train dizinlerini mutlak yola çevirir (çıktı dizini hariç)."""
    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    train = cfg.get('train', [])
    if isinstance(train, str):
        train = [train]

    dirs = []
    for entry in train:
        p = (data_yaml.parent / entry).resolve()
        if output_dir.resolve() in (p, p.parent):
            continue  # daha önce eklenmiş augment çıktısını tekrar çoğaltma
        if p.exists():
            dirs.append(p)
        else:
            logger.warning(f"⚠️ Train dizini bulunamadı, atlandı: {p}")
    return dirs


def labels_dir_for(images_dir: Path) -> Path:
    """.../images dizininin kardeş labels dizinini döner."""
    return images_dir.parent / 'labels'


def update_data_yaml(data_yaml: Path, output_images: Path) -> None:
    """Çıktı dizinini data.yaml'ın train listesine metin bazlı ekler (yorumları korur)."""
    rel = Path('..') / output_images.resolve().relative_to(data_yaml.parent.parent.resolve())
    entry = f'  - "{rel.as_posix()}"'

    lines = data_yaml.read_text(encoding='utf-8').splitlines()
    if any(rel.as_posix() in ln for ln in lines):
        logger.info("data.yaml zaten güncel (augment dizini train listesinde)")
        return

    out, in_train, inserted = [], False, False
    for ln in lines:
        if ln.startswith('train:'):
            in_train = True
            out.append(ln)
            continue
        if in_train and not inserted and not ln.lstrip().startswith('- ') and ln.strip() != '':
            out.append(entry)
            inserted = True
            in_train = False
        out.append(ln)
    if in_train and not inserted:
        out.append(entry)
        inserted = True

    data_yaml.write_text('\n'.join(out) + '\n', encoding='utf-8')
    logger.info(f"✅ data.yaml güncellendi: train listesine eklendi → {rel.as_posix()}")


def augment(data_yaml_path: str, output_dir: str, factors: Dict[int, int],
            seed: int, do_update_yaml: bool) -> bool:
    try:
        import albumentations  # noqa: F401
    except ImportError:
        logger.error("albumentations kurulu değil: pip install albumentations")
        return False

    data_yaml = Path(data_yaml_path)
    output = Path(output_dir)
    img_out = output / 'images'
    lbl_out = output / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    transform = build_transform()

    train_dirs = resolve_train_dirs(data_yaml, output)
    if not train_dirs:
        logger.error("Hiç train dizini bulunamadı")
        return False

    logger.info(f"Çarpanlar: {factors}")
    box_counts_before: Dict[int, int] = {}
    box_counts_added: Dict[int, int] = {}
    images_augmented = 0
    copies_written = 0

    for src_idx, images_dir in enumerate(train_dirs):
        lbl_dir = labels_dir_for(images_dir)
        tag = f"s{src_idx}"
        logger.info(f"📂 Taranıyor: {images_dir}")

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            boxes = read_label(label_path)
            for cls, *_ in boxes:
                box_counts_before[cls] = box_counts_before.get(cls, 0) + 1
            if not boxes:
                continue

            # Görüntüdeki sınıflardan en yüksek çarpan
            factor = max((factors.get(cls, 1) for cls, *_ in boxes), default=1)
            n_copies = factor - 1
            if n_copies <= 0:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"⚠️ Okunamadı: {img_path}")
                continue

            images_augmented += 1
            bboxes = [(x, y, w, h) for _, x, y, w, h in boxes]
            class_labels = [cls for cls, *_ in boxes]

            for i in range(n_copies):
                try:
                    result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                except Exception as e:
                    logger.warning(f"⚠️ Augmentasyon hatası ({img_path.name}): {e}")
                    continue
                if not result['bboxes']:
                    continue  # tüm kutular kaybolduysa kopyayı atla

                stem = f"aug{i}_{tag}_{img_path.stem}"
                cv2.imwrite(str(img_out / f"{stem}.jpg"), result['image'],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(lbl_out / f"{stem}.txt", 'w', encoding='utf-8') as f:
                    for (x, y, w, h), cls in zip(result['bboxes'], result['class_labels']):
                        # int(): albumentations etiketleri float döndürür, YOLO formatı
                        # tam sayı sınıf ID'si bekler ("0.0" değil "0")
                        cls = int(cls)
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        box_counts_added[cls] = box_counts_added.get(cls, 0) + 1
                copies_written += 1

    logger.info(f"\n✅ {images_augmented} görüntüden {copies_written} yeni kopya üretildi")
    logger.info("📊 Kutu sayıları (train): sınıf → önce + eklenen = sonra")
    for cls in sorted(set(box_counts_before) | set(box_counts_added)):
        before = box_counts_before.get(cls, 0)
        added = box_counts_added.get(cls, 0)
        marker = " ⬅ hedeflendi" if cls in factors else ""
        logger.info(f"  {cls}: {before} + {added} = {before + added}{marker}")

    if do_update_yaml:
        update_data_yaml(data_yaml, img_out)
    else:
        logger.info(f"ℹ️ data.yaml'a eklemek için: --update-data-yaml "
                    f"(veya train listesine elle ekleyin: {img_out})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sınıf hedefli augmentasyon — az örnekli sınıfları çoğaltarak dengesizliği azaltır")
    parser.add_argument("--data", type=str, default="configs/strawberry_data.yaml",
                        help="Dataset config (train dizin listesi buradan okunur)")
    parser.add_argument("--output", type=str, default="dataset/augmented_train",
                        help="Augment çıktı dizini (images/ + labels/)")
    parser.add_argument("--factors", type=str, default=None,
                        help="'sınıf:çarpan' listesi, örn. '1:4,2:3,5:3' (varsayılan: dengesiz sınıflar)")
    parser.add_argument("--seed", type=int, default=0, help="Rastgelelik tohumu")
    parser.add_argument("--update-data-yaml", action="store_true",
                        help="Çıktıyı data.yaml'ın train listesine otomatik ekle")

    args = parser.parse_args()
    factors = parse_factors(args.factors) if args.factors else dict(DEFAULT_FACTORS)

    success = augment(args.data, args.output, factors, args.seed, args.update_data_yaml)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
