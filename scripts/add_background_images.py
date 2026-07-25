"""
Sağlıklı (background) görüntüleri eğitim setine ekleme scripti.

NEDEN GEREKLİ?
    Mevcut 7 sınıfın hepsi hastalıktır; dataset'te "sağlıklı bitki" örneği
    yoksa model, sağlıklı bir yaprakta bile bir hastalık uydurabilir
    (false positive / yanlış alarm). Ticari üründe çiftçinin güvenini en
    çok yanlış alarmlar bozar. Ultralytics'in önerisi, eğitim setinin
    ~%10'unun etiketsiz background (sağlıklı) görüntü olmasıdır: model bu
    görüntülerde "hiçbir şey tespit etme"yi öğrenir ve yanlış alarm oranı
    ciddi şekilde düşer.

    YOLO formatında background görüntü = görüntü + BOŞ label (.txt) dosyası.
    Bu script kopyalama ve boş label oluşturmayı otomatik yapar, %10 hedef
    oranı aşmamaya dikkat eder.

Usage:
    python scripts/add_background_images.py \
        --dataset datasets/split \
        --source datasets/healthy_images \
        --target-ratio 0.10
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def resolve_split_dirs(dataset: Path, split: str) -> tuple:
    """Split'in görüntü/label dizinlerini bulur — iki yaygın düzeni de destekler:
    YOLO düzeni (images/train) ve Roboflow export düzeni (train/images)."""
    candidates = [
        (dataset / 'images' / split, dataset / 'labels' / split),
        (dataset / split / 'images', dataset / split / 'labels'),
    ]
    if split == 'val':  # Roboflow 'val' yerine 'valid' kullanır
        candidates.append((dataset / 'valid' / 'images', dataset / 'valid' / 'labels'))
    for img_dir, lbl_dir in candidates:
        if img_dir.exists():
            return img_dir, lbl_dir
    return None, None


def count_split(img_dir: Path, lbl_dir: Path) -> tuple:
    """Split'teki toplam görüntü ve mevcut background (boş label) sayısını döner."""

    total = 0
    backgrounds = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        total += 1
        label = lbl_dir / f"{img.stem}.txt"
        if not label.exists() or label.stat().st_size == 0:
            backgrounds += 1
    return total, backgrounds


def add_backgrounds(dataset_dir: str, source_dir: str, target_ratio: float,
                    split: str, prefix: str) -> bool:
    """Sağlıklı görüntüleri hedef orana ulaşacak kadar eğitim setine ekler."""
    dataset = Path(dataset_dir)
    source = Path(source_dir)

    img_dst, lbl_dst = resolve_split_dirs(dataset, split)
    if img_dst is None:
        logger.error(f"Split dizini bulunamadı: {dataset} altında ne images/{split} ne de {split}/images var")
        return False
    if not source.exists():
        logger.error(f"Kaynak dizin bulunamadı: {source}")
        return False

    total, existing_bg = count_split(img_dst, lbl_dst)
    labeled = total - existing_bg
    logger.info(f"Mevcut durum ({split}): {total} görüntü, {existing_bg} background "
                f"(%{100 * existing_bg / total:.1f})" if total else "Split boş")

    # Hedef: bg / (labeled + bg) = target_ratio  =>  bg = labeled * r / (1 - r)
    target_bg = int(labeled * target_ratio / (1.0 - target_ratio))
    needed = target_bg - existing_bg

    if needed <= 0:
        logger.info(f"✅ Background oranı zaten hedefte veya üzerinde "
                    f"(mevcut {existing_bg}, hedef {target_bg}). Ekleme yapılmadı.")
        return True

    candidates = sorted(p for p in source.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
    if not candidates:
        logger.error("Kaynak dizinde görüntü bulunamadı")
        return False

    if len(candidates) < needed:
        logger.warning(f"⚠️  {needed} görüntü gerekli ama kaynakta {len(candidates)} var. "
                       f"Hepsi eklenecek; hedef %{100 * target_ratio:.0f} oranına ulaşılamayacak.")

    lbl_dst.mkdir(parents=True, exist_ok=True)
    added = 0
    for img in candidates:
        if added >= needed:
            break
        dst_name = f"{prefix}{img.stem}{img.suffix.lower()}"
        dst_img = img_dst / dst_name
        if dst_img.exists():
            continue
        shutil.copy2(img, dst_img)
        # Boş label dosyası = "bu görüntüde tespit edilecek bir şey yok" sinyali
        (lbl_dst / f"{prefix}{img.stem}.txt").touch()
        added += 1

    new_total = total + added
    new_bg = existing_bg + added
    logger.info(f"✅ {added} sağlıklı görüntü eklendi")
    logger.info(f"📊 Yeni durum: {new_total} görüntü, {new_bg} background "
                f"(%{100 * new_bg / new_total:.1f})")
    logger.info("ℹ️  Bu görüntüler modele 'sağlıklı bitkide tespit üretme' davranışını "
                "öğretir ve sahadaki yanlış alarm oranını düşürür.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sağlıklı (background) görüntüleri eğitim setine ekle — yanlış alarmları azaltır")
    parser.add_argument("--dataset", type=str, required=True,
                        help="YOLO dataset kök dizini (images/train, labels/train içeren)")
    parser.add_argument("--source", type=str, required=True,
                        help="Sağlıklı bitki görüntülerinin bulunduğu dizin")
    parser.add_argument("--target-ratio", type=float, default=0.10,
                        help="Hedef background oranı (varsayılan 0.10 = %%10)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Eklenecek split (varsayılan train)")
    parser.add_argument("--prefix", type=str, default="bg_",
                        help="Eklenen dosyalara ad öneki (varsayılan bg_)")

    args = parser.parse_args()

    if not (0.0 < args.target_ratio < 0.5):
        logger.error("target-ratio 0 ile 0.5 arasında olmalı")
        return 1

    success = add_backgrounds(args.dataset, args.source, args.target_ratio,
                              args.split, args.prefix)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
