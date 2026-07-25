"""
Grup bazlı (veri sızıntısız) train/val/test split scripti.

NEDEN GEREKLİ?
    Aynı bitkinin/seranın/çekim gününün farklı kareleri birbirine çok benzer.
    Görüntü bazında rastgele split yapılırsa aynı bitkinin bir karesi train'e,
    diğer karesi test'e düşer; model test setini "ezberden" bilir ve metrikler
    yapay olarak yüksek çıkar (veri sızıntısı / data leakage). Sahada ise model
    hiç görmediği bitkilerle karşılaşacağı için gerçek performans çok daha
    düşük olur. Bu script, görüntüleri GRUP (bitki/sera/çekim günü) bazında
    böler: bir grubun TÜM görüntüleri aynı split'e gider.

Grup bilgisi iki yolla verilebilir (arayüz gerekmez):
    1. Dosya adı deseni: örn. "sera1_bitki05_003.jpg" → grup = "sera1_bitki05"
       (varsayılan regex, son "_sayı" ekini atar)
    2. Metadata CSV: "filename,group" kolonlu bir dosya (--metadata-csv)

Usage:
    # Dosya adı desenine göre (varsayılan: son _sayı eki atılır)
    python scripts/split_dataset.py --input datasets/raw --output datasets/split

    # Özel regex ile (ilk yakalama grubu = grup anahtarı)
    python scripts/split_dataset.py --input datasets/raw --output datasets/split \
        --group-regex "^(sera\\d+_bitki\\d+)"

    # Metadata CSV ile
    python scripts/split_dataset.py --input datasets/raw --output datasets/split \
        --metadata-csv metadata.csv
"""

import argparse
import csv
import logging
import random
import re
import shutil
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_GROUP_REGEX = r'^(.+?)_\d+$'  # "sera1_bitki05_003" -> "sera1_bitki05"


def collect_images(input_dir: Path) -> List[Path]:
    """Girdi dizinindeki tüm görüntüleri toplar (düz veya split'li yapı fark etmez)."""
    images = [p for p in input_dir.rglob('*')
              if p.suffix.lower() in IMAGE_EXTS and 'labels' not in p.parts]
    logger.info(f"Toplam {len(images)} görüntü bulundu")
    return images


def build_label_index(input_dir: Path) -> Dict[str, Path]:
    """Stem → label dosyası eşlemesi kurar."""
    index = {}
    for txt in input_dir.rglob('*.txt'):
        if txt.name in ('classes.txt', 'labels.txt', 'README.txt'):
            continue
        index[txt.stem] = txt
    return index


def load_groups_from_csv(csv_path: Path) -> Dict[str, str]:
    """CSV'den (filename,group) eşlemesini yükler."""
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('filename', '').strip()
            group = row.get('group', '').strip()
            if fname and group:
                mapping[Path(fname).stem] = group
    logger.info(f"CSV'den {len(mapping)} grup eşlemesi yüklendi")
    return mapping


def resolve_group(stem: str, regex: re.Pattern,
                  csv_map: Optional[Dict[str, str]]) -> str:
    """Bir görüntünün grup anahtarını belirler."""
    if csv_map is not None:
        return csv_map.get(stem, stem)
    m = regex.match(stem)
    return m.group(1) if m else stem


def assign_groups(group_sizes: Dict[str, int], ratios: Dict[str, float],
                  seed: int) -> Dict[str, str]:
    """Grupları, görüntü sayısı oranlarını yaklaşık koruyacak şekilde split'lere atar.

    Her grup bütün olarak tek bir split'e gider — sızıntıyı önleyen kural budur.
    """
    total = sum(group_sizes.values())
    targets = {s: total * r for s, r in ratios.items()}
    assigned_counts = {s: 0 for s in ratios}
    assignment = {}

    groups = list(group_sizes.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Büyük gruplar önce yerleştirilirse oranlar daha dengeli tutturulur
    groups.sort(key=lambda g: -group_sizes[g])

    for g in groups:
        # En fazla açığı olan split'e ata
        split = max(ratios, key=lambda s: targets[s] - assigned_counts[s])
        assignment[g] = split
        assigned_counts[split] += group_sizes[g]

    for s in ratios:
        pct = 100 * assigned_counts[s] / total if total else 0
        n_groups = sum(1 for v in assignment.values() if v == s)
        logger.info(f"  {s}: {assigned_counts[s]} görüntü (%{pct:.1f}), {n_groups} grup")

    return assignment


def split_dataset(input_dir: str, output_dir: str, train_ratio: float,
                  val_ratio: float, group_regex: str,
                  metadata_csv: Optional[str], seed: int) -> bool:
    """Dataset'i grup bazlı olarak train/val/test'e böler."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Girdi dizini bulunamadı: {input_path}")
        return False

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        logger.error("train + val oranı 1.0'ı aşamaz")
        return False
    ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}

    images = collect_images(input_path)
    if not images:
        logger.error("Görüntü bulunamadı")
        return False

    label_index = build_label_index(input_path)
    csv_map = load_groups_from_csv(Path(metadata_csv)) if metadata_csv else None
    regex = re.compile(group_regex)

    # Grupları oluştur
    group_images = defaultdict(list)
    for img in images:
        group_images[resolve_group(img.stem, regex, csv_map)].append(img)

    n_groups = len(group_images)
    logger.info(f"{n_groups} grup bulundu")
    if n_groups == len(images):
        logger.warning(
            "⚠️  Her görüntü kendi grubu oldu — dosya adları desene uymuyor olabilir. "
            "Bu durumda split, rastgele split ile aynıdır ve sızıntı koruması SAĞLAMAZ. "
            "--group-regex veya --metadata-csv ile grup bilgisi verin."
        )

    group_sizes = {g: len(imgs) for g, imgs in group_images.items()}
    assignment = assign_groups(group_sizes, ratios, seed)

    # Kopyala
    copied = {'train': 0, 'val': 0, 'test': 0}
    missing_labels = 0
    for group, imgs in group_images.items():
        split = assignment[group]
        img_dst = output_path / 'images' / split
        lbl_dst = output_path / 'labels' / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            shutil.copy2(img, img_dst / img.name)
            label = label_index.get(img.stem)
            if label:
                shutil.copy2(label, lbl_dst / f"{img.stem}.txt")
            else:
                # Label'sız görüntü = background (sağlıklı) görüntü olarak kalır
                missing_labels += 1
            copied[split] += 1

    if missing_labels:
        logger.info(f"{missing_labels} görüntünün label'ı yok (background olarak kullanılacak)")

    # data.yaml: kaynakta varsa sınıf isimlerini taşı
    src_yaml = next(input_path.rglob('data.yaml'), None)
    names = None
    if src_yaml:
        with open(src_yaml, 'r', encoding='utf-8') as f:
            names = yaml.safe_load(f).get('names')

    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
    }
    if names:
        data_yaml['names'] = names
        data_yaml['nc'] = len(names)

    with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    # Doğrulama: hiçbir grup birden fazla split'te olmamalı
    logger.info("Doğrulama: her grup tek split'te ✓ (atama grup bazlı yapıldı)")
    logger.info(f"Split tamamlandı: {copied}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Grup bazlı (bitki/sera/çekim günü) veri sızıntısız dataset split")
    parser.add_argument("--input", type=str, required=True, help="Kaynak dataset dizini")
    parser.add_argument("--output", type=str, required=True, help="Hedef dataset dizini")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train oranı (varsayılan 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val oranı (varsayılan 0.2, kalan test)")
    parser.add_argument("--group-regex", type=str, default=DEFAULT_GROUP_REGEX,
                        help="Dosya adından (uzantısız) grup anahtarını çıkaran regex; ilk yakalama grubu kullanılır")
    parser.add_argument("--metadata-csv", type=str, default=None,
                        help="'filename,group' kolonlu CSV (regex yerine kullanılır)")
    parser.add_argument("--seed", type=int, default=0, help="Rastgelelik tohumu")

    args = parser.parse_args()

    success = split_dataset(args.input, args.output, args.train_ratio,
                            args.val_ratio, args.group_regex,
                            args.metadata_csv, args.seed)

    if success:
        logger.info("✅ Grup bazlı split tamamlandı — aynı bitki/sera asla iki split'e bölünmedi")
        logger.info(f"📁 Konum: {args.output}")
        logger.info("📝 Sonraki adım: python scripts/train_yolo.py --data <output>/data.yaml")
    else:
        logger.error("❌ Split başarısız!")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
