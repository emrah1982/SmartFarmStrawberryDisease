"""
SAHI ile dilimleyerek (sliced) inference scripti.

NEDEN GEREKLİ?
    Sahada çekilen görüntüler genelde yüksek çözünürlüklüdür (örn. 4000x3000).
    YOLO bu görüntüyü 640x640'a küçülterek işler; erken evre hastalık lezyonları
    (angular leafspot, leaf spot gibi birkaç mm'lik lekeler) bu küçültmede
    birkaç piksele düşer ve model onları GÖREMEZ. Halbuki erken evre tespiti,
    ticari üründe en değerli özelliktir — hastalık yayılmadan müdahale şansı verir.

    SAHI (Slicing Aided Hyper Inference) görüntüyü örtüşen dilimlere böler,
    her dilimi ayrı ayrı modele verir ve sonuçları birleştirir. Böylece küçük
    lezyonlar tam çözünürlükte incelenir. Bedeli: inference süresi dilim sayısı
    kadar artar — gerçek zamanlı video için değil, fotoğraf bazlı analiz için uygundur.

Kurulum:
    pip install sahi

Usage:
    # Tek görüntü
    python scripts/sahi_predict.py --model runs/train/strawberry_exp/weights/best.pt \
        --source field_photo.jpg

    # Dizindeki tüm görüntüler
    python scripts/sahi_predict.py --model best.pt --source datasets/field_photos \
        --slice-size 640 --overlap 0.2 --conf 0.25
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def run_sahi(model_path: str, source: str, output_dir: str, slice_size: int,
             overlap: float, conf: float, device: str) -> bool:
    """SAHI ile dilimli inference çalıştırır ve görselleştirilmiş sonuçları kaydeder."""
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        logger.error("SAHI kurulu değil. Kurulum: pip install sahi")
        return False

    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        images = sorted(p for p in source_path.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
    elif source_path.exists():
        images = [source_path]
    else:
        logger.error(f"Kaynak bulunamadı: {source}")
        return False

    if not images:
        logger.error("İşlenecek görüntü bulunamadı")
        return False

    logger.info(f"Model yükleniyor: {model_path}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    total_detections = 0
    for img in images:
        logger.info(f"İşleniyor: {img.name}")
        result = get_sliced_prediction(
            str(img),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
        )

        n = len(result.object_prediction_list)
        total_detections += n

        result.export_visuals(export_dir=str(output_path), file_name=img.stem)

        # Sınıf bazlı özet — hangi hastalıktan kaç adet bulundu
        class_counts = {}
        for pred in result.object_prediction_list:
            name = pred.category.name
            class_counts[name] = class_counts.get(name, 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items())) or "tespit yok"
        logger.info(f"  {n} tespit → {summary}")

    logger.info(f"✅ {len(images)} görüntü işlendi, toplam {total_detections} tespit")
    logger.info(f"📁 Görselleştirilmiş sonuçlar: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="SAHI ile dilimli inference — yüksek çözünürlüklü görüntülerde küçük lezyon tespiti")
    parser.add_argument("--model", type=str, required=True, help="Eğitilmiş model yolu (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Görüntü dosyası veya dizini")
    parser.add_argument("--output", type=str, default="runs/sahi", help="Çıktı dizini")
    parser.add_argument("--slice-size", type=int, default=640,
                        help="Dilim boyutu (px). Eğitimdeki imgsz ile aynı olmalı (varsayılan 640)")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="Dilimler arası örtüşme oranı — dilim sınırındaki lezyonların kaçmaması için (varsayılan 0.2)")
    parser.add_argument("--conf", type=float, default=0.25, help="Güven eşiği")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="cihaz: cuda:0 veya cpu")

    args = parser.parse_args()

    success = run_sahi(args.model, args.source, args.output, args.slice_size,
                       args.overlap, args.conf, args.device)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
