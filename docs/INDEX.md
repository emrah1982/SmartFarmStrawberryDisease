# 🍓 Strawberry Vision - Dokümantasyon Ana Sayfa

## Proje Özeti

Strawberry Vision, Google Colab uyumlu, katmanlı mimariye sahip profesyonel bir çilek görüntü analiz sistemidir. Sistem, YOLO tabanlı nesne tespiti ile çileklerde hastalık belirtilerini tespit eder (7 sınıf), takip eder ve sayım/istatistik üretir.

## Hızlı Başlangıç

```bash
# Bağımlılıkları kur
pip install -r requirements.txt

# Tek görsel ile çalıştır
python -m strawberry_vision.main --image sample.jpg --model path/to/best.pt

# Smoke test
python tests/smoke_test.py
```

## Katmanlı Mimari

Bu proje 4 katmana ayrılmıştır:

- **Presentation**: Görselleştirme ve çıktı üretimi (`visualizer.py`)
- **Application**: Pipeline orkestrasyon (`pipeline.py`)
- **Domain**: İş kuralları ve varlıklar (`entities.py`, `services.py`)
- **Infrastructure**: Model, veri kaynakları (`detectors.py`, `sources.py`)

## Temel Dokümantasyon

### Kullanım ve Geliştirme
- **Kullanım Kılavuzu**: `docs/USAGE.md` - Kurulum, çalıştırma, Colab kullanımı
- **Mimari Tasarım**: `docs/architecture.md` - Katmanlar, bağımlılıklar, veri akışı
- **Geliştirme Kuralları**: `docs/development-rules.md` - SOLID, kod stili, test kuralları

### Model Eğitimi ve Dataset
- **Görüntü Analizi**: `docs/1-gorunuAnalizi.md` - Hastalık odaklı dataset stratejisi, etiketleme kuralları
- **YOLO Eğitimi**: `docs/2-YOLOegitimiHiperparametre.md` - Hiperparametre optimizasyonu (7 sınıf)
- **Roboflow Etiketleme**: `docs/2.1-roboflowEtiketlemeTalimati.md` - Etiketleme talimatları
- **Hata Analizi**: `docs/2.2-ModelHataAnaliziIyilestirmePromptu.md` - Model iyileştirme
- **Roboflow Dataset Kullanımı**: `docs/3-RoboflowDatasetKullanimi.md` - Dataset linkleri, augmentation, eğitim

## Proje Yapısı

```
strawberry_vision/
├── presentation/      # Görselleştirme
├── application/       # Pipeline yönetimi
├── domain/           # İş kuralları
├── infrastructure/   # Model ve veri kaynakları
└── main.py          # Giriş noktası

configs/
├── strawberry_data.yaml        # Dataset config (10 sınıf, çoklu kaynak)
├── train_config.yaml           # Eğitim parametreleri
└── class_aliases.yaml          # Sınıf adı eş anlamlıları

scripts/
├── download_dataset.py         # Roboflow'dan dataset indir
├── merge_datasets.py           # Çoklu kaynak birleştirme (ID çakışmasız)
├── split_dataset.py            # Grup bazlı (sızıntısız) split
├── add_background_images.py    # Sağlıklı görüntüleri background olarak ekle
├── augment_by_class.py         # Sınıf hedefli augmentation
├── train_yolo.py               # YOLO26 eğitimi
├── evaluate_model.py           # Model değerlendirme
└── sahi_predict.py             # Dilimli inference (küçük lezyonlar)

tests/
├── test_domain_entities.py     # Domain testleri
├── test_domain_services.py     # Service testleri
├── test_application_pipeline.py # Pipeline testleri
└── smoke_test.py               # Entegrasyon testi
```

## 🎓 Eğitim Pipeline

### 1. Dataset Hazırlama
```bash
# Roboflow'dan indir
python scripts/download_dataset.py --api-key YOUR_KEY --workspace strawberry --project ripeness

# Kaynakları birleştir (sınıf isimlerine göre eşler, ID çakışmasını önler)
python scripts/merge_datasets.py --inputs datasets/kaynak1 datasets/kaynak2 --output datasets/merged

# Sınıf dengesizliğini gider (az örnekli sınıfları hedefli çoğalt)
python scripts/augment_by_class.py --update-data-yaml
```

### 2. Model Eğitimi
```bash
# Config ile eğitim
python scripts/train_yolo.py --data configs/strawberry_data.yaml --config configs/train_config.yaml

# Parametrelerle eğitim
python scripts/train_yolo.py --data datasets/processed/data.yaml --epochs 100 --batch 16 --model yolov8s.pt
```

### 3. Model Değerlendirme
```bash
python scripts/evaluate_model.py --model runs/train/strawberry_exp/weights/best.pt --data configs/strawberry_data.yaml
```

## Katkıda Bulunma

Kod yazarken `docs/development-rules.md` ve `docs/architecture.md` dokümanlarına uyun.
