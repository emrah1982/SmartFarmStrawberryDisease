"""
Farklı kaynaklardan gelen dataset'leri sınıf ÇAKIŞMASI olmadan birleştirme scripti.

PROBLEM:
    Her kaynak dataset kendi sınıf numaralandırmasını kullanır. Örneğin bir
    kaynakta "healthy leaf" = 0 iken, başka bir kaynakta "gray mold" = 0
    olabilir. Label dosyaları (.txt) sadece sayısal ID içerdiği için bu
    dataset'ler doğrudan kopyalanıp birleştirilirse ID'ler çakışır ve model
    tamamen yanlış etiketlerle eğitilir (sağlıklı yaprağı hastalık sanır).

ÇÖZÜM:
    ID'lere değil SINIF İSİMLERİNE göre eşleme yapılır:
    1. Ana (master) sınıf listesi tanımlanır (varsayılan: configs/strawberry_data.yaml)
    2. Her kaynağın data.yaml'ındaki isimler normalize edilip master listeyle eşlenir
       ("Gray-Mold", "gray_mold", "GRAY MOLD" → hepsi aynı sınıfa gider)
    3. Eş anlamlı/farklı yazımlar için configs/class_aliases.yaml kullanılır
       ("botrytis" → "Gray Mold" gibi)
    4. Tüm label dosyalarındaki ID'ler master listeye göre yeniden yazılır
    5. Dosya adları kaynak öneki ile kopyalanır (farklı kaynaklarda aynı isimli
       dosyaların birbirini ezmesini önler)

    Master listede olmayan sınıflar için davranış seçilir (--on-unknown):
    - error: dur ve bildir (varsayılan — sessiz veri kaybını önler)
    - drop:  o sınıfın kutularını at (örn. "healthy" kutularını atarsanız o
             görüntüler background/sağlıklı örneğe dönüşür — yanlış alarm
             azaltmak için genelde İSTENEN davranıştır)
    - add:   master listenin sonuna yeni sınıf olarak ekle

Çıktı DÜZ yapıdadır (images/ + labels/): birleştirme sonrası grup bazlı split
için scripts/split_dataset.py kullanın (sıralama: merge → split → background → train).

Usage:
    python scripts/merge_datasets.py \
        --inputs datasets/kaynak1 datasets/kaynak2 datasets/kaynak3 \
        --output datasets/merged

    # "healthy" etiketli kutuları atıp o görüntüleri background yap:
    python scripts/merge_datasets.py --inputs d1 d2 --output merged \
        --on-unknown drop --drop-classes "healthy" "healthy leaf"
"""

import argparse
import logging
import re
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_MASTER_YAML = Path(__file__).resolve().parent.parent / 'configs' / 'strawberry_data.yaml'
DEFAULT_ALIAS_YAML = Path(__file__).resolve().parent.parent / 'configs' / 'class_aliases.yaml'


def normalize(name: str) -> str:
    """Sınıf adını karşılaştırma için normalize eder: 'Gray-Mold ' → 'gray mold'."""
    return re.sub(r'[\s_\-]+', ' ', name.strip().lower())


def load_names(yaml_path: Path) -> List[str]:
    """data.yaml'dan sınıf isimlerini sıralı liste olarak yükler."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def load_aliases(alias_path: Optional[Path]) -> Dict[str, str]:
    """Alias dosyasını yükler: {normalize(alias): normalize(kanonik ad)}."""
    if not alias_path or not alias_path.exists():
        return {}
    with open(alias_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    aliases = {normalize(k): normalize(v) for k, v in raw.items()}
    logger.info(f"{len(aliases)} alias yüklendi: {alias_path}")
    return aliases


def build_id_map(source_names: List[str], master_names: List[str],
                 aliases: Dict[str, str], drop_classes: Set[str],
                 on_unknown: str) -> Dict[int, Optional[int]]:
    """Kaynak sınıf ID'lerini master ID'lere eşler.

    Returns:
        {kaynak_id: master_id veya None (None = kutu atılacak)}
    Raises:
        ValueError: on_unknown='error' iken eşlenemeyen sınıf varsa
    """
    master_lookup = {normalize(n): i for i, n in enumerate(master_names)}
    id_map: Dict[int, Optional[int]] = {}
    unknown: List[str] = []

    for src_id, src_name in enumerate(source_names):
        key = aliases.get(normalize(src_name), normalize(src_name))

        if key in drop_classes:
            id_map[src_id] = None
            logger.info(f"    '{src_name}' → ATILACAK (drop-classes)")
            continue

        if key in master_lookup:
            id_map[src_id] = master_lookup[key]
            logger.info(f"    '{src_name}' (id {src_id}) → '{master_names[master_lookup[key]]}' (id {master_lookup[key]})")
        elif on_unknown == 'drop':
            id_map[src_id] = None
            logger.warning(f"    ⚠️ '{src_name}' master listede yok → kutuları ATILACAK")
        elif on_unknown == 'add':
            master_names.append(src_name)
            new_id = len(master_names) - 1
            master_lookup[key] = new_id
            id_map[src_id] = new_id
            logger.info(f"    ➕ '{src_name}' master listeye eklendi (id {new_id})")
        else:
            unknown.append(src_name)

    if unknown:
        raise ValueError(
            f"Eşlenemeyen sınıflar: {unknown}\n"
            f"Çözümler:\n"
            f"  - configs/class_aliases.yaml dosyasına alias ekleyin (örn. \"{unknown[0]}\": \"Gray Mold\")\n"
            f"  - --on-unknown drop ile bu sınıfların kutularını atın\n"
            f"  - --on-unknown add ile master listeye yeni sınıf olarak ekleyin"
        )
    return id_map


def remap_label_lines(label_path: Path, id_map: Dict[int, Optional[int]]) -> tuple:
    """Label satırlarını yeni ID'lerle döner. Returns: (yeni_satırlar, atılan_kutu_sayısı)."""
    kept, dropped = [], 0
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            new_id = id_map.get(int(parts[0]))
            if new_id is None:
                dropped += 1
                continue
            parts[0] = str(new_id)
            kept.append(' '.join(parts) + '\n')
    return kept, dropped


def collect_pairs(dataset_dir: Path) -> List[tuple]:
    """Dataset'teki (görüntü, label|None) çiftlerini toplar (düz veya split'li yapı)."""
    label_index = {}
    for txt in dataset_dir.rglob('*.txt'):
        if txt.name not in ('classes.txt', 'labels.txt', 'README.txt', 'README.roboflow.txt'):
            label_index[txt.stem] = txt

    pairs = []
    for img in dataset_dir.rglob('*'):
        if img.suffix.lower() in IMAGE_EXTS and 'labels' not in img.parts:
            pairs.append((img, label_index.get(img.stem)))
    return pairs


def merge_datasets(input_dirs: List[str], output_dir: str, master_yaml: str,
                   alias_yaml: Optional[str], on_unknown: str,
                   drop_classes: List[str]) -> bool:
    """Dataset'leri isim bazlı sınıf eşlemesiyle tek düz dataset'te birleştirir."""
    output_path = Path(output_dir)
    master_path = Path(master_yaml)

    if not master_path.exists():
        logger.error(f"Master sınıf listesi bulunamadı: {master_path}")
        return False

    master_names = load_names(master_path)
    logger.info(f"Master sınıflar ({len(master_names)}): {master_names}")

    aliases = load_aliases(Path(alias_yaml) if alias_yaml else DEFAULT_ALIAS_YAML)
    drop_set = {normalize(c) for c in drop_classes}

    img_out = output_path / 'images'
    lbl_out = output_path / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_dropped_boxes = 0
    total_backgrounds = 0

    for idx, input_dir in enumerate(input_dirs):
        src = Path(input_dir)
        tag = f"src{idx}_{src.name}"
        logger.info(f"\n📦 Kaynak {idx + 1}/{len(input_dirs)}: {src} (önek: {tag}_)")

        src_yaml = next(src.rglob('data.yaml'), None)
        if not src_yaml:
            logger.error(f"  data.yaml bulunamadı: {src} — bu kaynak ATLANDI. "
                         f"Sınıf isimleri bilinmeden ID eşlemesi yapılamaz!")
            continue

        source_names = load_names(src_yaml)
        logger.info(f"  Kaynak sınıflar: {source_names}")

        try:
            id_map = build_id_map(source_names, master_names, aliases, drop_set, on_unknown)
        except ValueError as e:
            logger.error(f"  ❌ {e}")
            return False

        pairs = collect_pairs(src)
        logger.info(f"  {len(pairs)} görüntü bulundu")

        for img, label in pairs:
            # Kaynak öneki: farklı kaynaklardaki aynı isimli dosyalar çakışmasın
            dst_stem = f"{tag}_{img.stem}"
            shutil.copy2(img, img_out / f"{dst_stem}{img.suffix.lower()}")

            if label:
                kept, dropped = remap_label_lines(label, id_map)
                total_dropped_boxes += dropped
                with open(lbl_out / f"{dst_stem}.txt", 'w', encoding='utf-8') as f:
                    f.writelines(kept)
                if not kept:
                    total_backgrounds += 1  # tüm kutuları atılan görüntü → background
            else:
                (lbl_out / f"{dst_stem}.txt").touch()
                total_backgrounds += 1
            total_images += 1

    # Birleşik data.yaml
    merged_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images',
        'val': 'images',
        'nc': len(master_names),
        'names': {i: n for i, n in enumerate(master_names)},
    }
    with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(merged_yaml, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"\n✅ Birleştirme tamamlandı: {total_images} görüntü")
    logger.info(f"📊 Master sınıflar ({len(master_names)}): {master_names}")
    if total_dropped_boxes:
        logger.info(f"🗑️  Atılan kutu: {total_dropped_boxes} (drop edilen sınıflar)")
    if total_backgrounds:
        logger.info(f"🍃 Background görüntü: {total_backgrounds} (label'sız veya tüm kutuları atılmış)")
    logger.info("📝 Sonraki adım (split ÖNCE yapılmamışsa): "
                "python scripts/split_dataset.py --input " + str(output_path) + " --output datasets/split")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Farklı kaynaklardan dataset'leri sınıf ID çakışması olmadan birleştir")
    parser.add_argument("--inputs", type=str, nargs='+', required=True,
                        help="Kaynak dataset dizinleri (her birinde data.yaml olmalı)")
    parser.add_argument("--output", type=str, required=True, help="Birleşik dataset dizini")
    parser.add_argument("--master-yaml", type=str, default=str(DEFAULT_MASTER_YAML),
                        help="Ana sınıf listesini içeren yaml (varsayılan: configs/strawberry_data.yaml)")
    parser.add_argument("--alias-yaml", type=str, default=None,
                        help="Eş anlamlı sınıf adları dosyası (varsayılan: configs/class_aliases.yaml)")
    parser.add_argument("--on-unknown", type=str, default="error",
                        choices=["error", "drop", "add"],
                        help="Master listede olmayan sınıf için davranış (varsayılan: error)")
    parser.add_argument("--drop-classes", type=str, nargs='*', default=[],
                        help="Kutuları atılacak sınıf adları (örn. 'healthy leaf' → görüntü background olur)")

    args = parser.parse_args()

    success = merge_datasets(args.inputs, args.output, args.master_yaml,
                             args.alias_yaml, args.on_unknown, args.drop_classes)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
