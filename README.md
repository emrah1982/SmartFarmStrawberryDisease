# 🍓 Strawberry Vision - Çilek Görüntü Analiz Sistemi

Google Colab uyumlu, katmanlı mimariye sahip profesyonel çilek tespit ve olgunluk sınıflandırma sistemi.

## 🎯 Özellikler

- ✅ YOLO26 tabanlı çilek tespiti (Ultralytics — NMS-free, edge-dostu güncel mimari; YOLOv8 ile geriye uyumlu)
- ✅ Çilek hastalık tespiti (7 sınıf)
- ✅ Nesne takibi (tracking)
- ✅ Otomatik sayım ve istatistik
- ✅ Görselleştirme ve sonuç kaydetme
- ✅ Katmanlı mimari (Domain-Driven Design)
- ✅ Google Colab desteği
- ✅ Kapsamlı test coverage

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Repository'yi klonla
git clone <repository-url>
cd SmartFarmBerry

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Kullanım

```bash
# Tek görüntü ile çalıştır
python -m strawberry_vision.main --image sample.jpg --model path/to/best.pt

# Video ile çalıştır
python -m strawberry_vision.main --video video.mp4 --model path/to/best.pt --max-frames 100

# Smoke test
python tests/smoke_test.py
```

### Google Colab – Hızlı Başlangıç

Aşağıdaki adımlarla Google Colab üzerinde hızlıca eğitim ve inference çalıştırabilirsiniz.

#### Yöntem 1: Notebook ile Manuel Çalıştırma

- **1) Colab'i aç ve GPU seç**
  - Runtime > Change runtime type > Hardware accelerator: GPU

- **2) Depoyu Colab'e klonla**
  ```bash
  !git clone https://github.com/emrah1982/SmartFarmStrawberry.git
  %cd SmartFarmStrawberry
  ```

- **3) Bağımlılıkları kur**
  ```bash
  !pip install -q -r requirements.txt
  ```

- **4) Google Drive'ı bağla (checkpoint ve sonuçlar için)**
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

- **5) Roboflow API Key'i güvenli şekilde ayarla (ÖNEMLİ)**
  
  **Önerilen: Colab Secrets kullanın**
  ```python
  from google.colab import userdata
  import os
  
  # Sol panelde 🔑 (Secrets) ikonuna tıklayın
  # Name: ROBOFLOW_API_KEY, Value: rf_... (API key'iniz)
  os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')
  ```
  
  **Alternatif: Manuel giriş (geçici)**
  ```python
  from getpass import getpass
  import os
  
  API_KEY = getpass("Roboflow API Key: ")  # Girdiğiniz görünmez
  os.environ['ROBOFLOW_API_KEY'] = API_KEY
  ```
  
  🔑 API Key alma: https://app.roboflow.com/settings/api

- **6) Production notebook'u aç**
  - Dosya: `StrawberryVision_Colab_Production.ipynb`
  - İçerikte şunlar hazırdır:
    - Roboflow API ile dataset indirme (4 doğrulanmış dataset seçeneği)
    - Sınıf etiketlerini otomatik standardize etme
    - Eğitim konfigürasyonu (`configs/train_config.yaml`) ve augmentasyon ayarları
    - Her 10 epoch'ta checkpoint kaydetme (Google Drive)

- **7) Tüm hücreleri sırayla çalıştır**
  - Eğitim sonunda en iyi model ve tüm checkpoint'ler Drive'a kopyalanır.
  - Sonuç görselleri ve metrikler `runs/train/...` altında da kaydedilir.

#### Yöntem 2: Headless Çalıştırma (nbconvert)

Notebook'u dosya menüsünü açmadan komut satırından çalıştırabilirsiniz:

```python
# 1) Kurulum
!git clone https://github.com/emrah1982/SmartFarmStrawberry.git
%cd SmartFarmStrawberry
!pip install -q -r requirements.txt nbconvert jupyter roboflow

# 2) API Key'i ayarla (Colab Secrets'tan)
from google.colab import userdata, drive
import os

os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')
drive.mount('/content/drive')

# 3) Notebook'u çalıştır
!jupyter nbconvert --to notebook --execute StrawberryVision_Colab_Production.ipynb \
  --output executed.ipynb --ExecutePreprocessor.timeout=-1
```

#### Dataset Versiyonları

Roboflow datasetlerinin çoğu **version 2** veya üstünü kullanır. Eğer version hatası alırsanız:

```python
# Hücre 0'da VERSION parametresini değiştirin
VERSION = 2  # veya 3, 4, vb.
```

Mevcut versiyonları kontrol etmek için: `https://universe.roboflow.com/{workspace}/{project}`

**⚠️ Güvenlik Notu**: API key'inizi asla kod hücresine yazmayın. Colab Secrets veya `getpass()` kullanın.

Not: Colab dışında lokalde çalıştırmak için de aynı dizin yapısı ve `scripts/` altındaki yardımcı komutlar kullanılabilir.

## 📦 Model Eğitimi

### 1. Dataset Hazırlama

```bash
# Roboflow'dan dataset indir
python scripts/download_dataset.py --api-key YOUR_KEY --workspace strawberry --project ripeness

# Farklı kaynaklardan dataset'leri sınıf çakışması OLMADAN birleştir (birden çok kaynak varsa)
# Sınıf isimlerini standardize etme işini de bu script yapar (configs/class_aliases.yaml ile)
python scripts/merge_datasets.py --inputs datasets/kaynak1 datasets/kaynak2 --output datasets/merged

# Grup bazlı (veri sızıntısız) train/val/test split — detaylar için aşağıdaki bölüme bakın
python scripts/split_dataset.py --input datasets/merged --output datasets/split

# Sağlıklı (background) görüntüleri ekle — yanlış alarmları azaltır
python scripts/add_background_images.py --dataset datasets/split --source datasets/healthy_images --target-ratio 0.10

# Sınıf dengesizliğini gider: az örnekli sınıfları hedefli çoğalt
python scripts/augment_by_class.py --update-data-yaml
```

#### 🔀 Farklı Kaynaklardan Dataset Birleştirme (`merge_datasets.py`)

**Neden gerekli?** YOLO label dosyaları (.txt) sınıf adını değil sadece **sayısal ID** içerir ve her kaynak dataset kendi numaralandırmasını kullanır. Bir kaynakta `0 = healthy leaf` iken başka kaynakta `0 = Gray Mold` olabilir. Bu dataset'leri doğrudan kopyalayıp birleştirirseniz ID'ler çakışır ve model **tamamen yanlış etiketlerle** eğitilir — sağlıklı yaprağı hastalık olarak öğrenir. Bu hata eğitim sırasında görünmez (loss düşer, eğitim "başarılı" görünür), sadece saçma tahminlerden fark edilir.

**Çözüm:** ID'lere değil **sınıf isimlerine** göre eşleme:

1. Ana (master) sınıf listesi tanımlanır — varsayılan: [configs/strawberry_data.yaml](configs/strawberry_data.yaml)'daki 7 hastalık sınıfı.
2. Her kaynağın `data.yaml`'ındaki sınıf isimleri normalize edilip master listeyle eşlenir (`gray_mold`, `Gray-Mold`, `GRAY MOLD` → otomatik aynı sınıf).
3. Farklı adlandırmalar (`botrytis` → `Gray Mold`, `külleme` → `Powdery Mildew Leaf`) [configs/class_aliases.yaml](configs/class_aliases.yaml) ile çözülür — kendi kaynaklarınıza göre bu dosyaya ekleme yapın.
4. Tüm label dosyalarındaki ID'ler master listeye göre yeniden yazılır; dosya adlarına kaynak öneki eklenir (aynı isimli dosyaların birbirini ezmesini önler).

```bash
python scripts/merge_datasets.py \
    --inputs datasets/kaynak1 datasets/kaynak2 datasets/kaynak3 \
    --output datasets/merged
```

**"Sağlıklı yaprak" sınıfı için iki seçenek:**

```bash
# Seçenek A (önerilen): healthy kutularını at → o görüntüler background olur
# (model "sağlıklı bitkide tespit üretme"yi öğrenir, yanlış alarm düşer)
python scripts/merge_datasets.py --inputs d1 d2 --output merged --drop-classes "healthy"

# Seçenek B: healthy'yi ayrı bir sınıf olarak master listeye ekle
python scripts/merge_datasets.py --inputs d1 d2 --output merged --on-unknown add
```

⚠️ **Güvenlik varsayılanı:** Eşlenemeyen bir sınıf bulunursa script **durur ve bildirir** (`--on-unknown error`). Sessizce yanlış eşleme veya veri kaybı yaşamazsınız; hata mesajı hangi sınıfın eşlenemediğini ve çözüm seçeneklerini gösterir.

⚠️ **Zorunlu koşul — her kaynak klasöründe `data.yaml` olmalı:** Label dosyaları sadece sayısal ID içerdiği için, sınıf isimleri bilinmeden ID eşlemesi yapmak imkânsızdır. `data.yaml`'ı olmayan kaynak **atlanır** ve log'da açıkça belirtilir. Roboflow export'ları bu dosyayı zaten içerir; elle etiketlediğiniz klasörler için şu kadar basit bir `data.yaml` yazmanız yeterlidir:

```yaml
nc: 2
names: ['healthy leaf', 'Gray Mold']  # ID sırası, etiketlemede kullandığınız sırayla AYNI olmalı
```

**Doğru sıralama:** `merge → split → background → train` (önce birleştir, sonra grup bazlı böl — böylece split oranları birleşik veri üzerinde doğru hesaplanır).

#### 🔒 Veri Sızıntısını Önleme: Grup Bazlı Split (`split_dataset.py`)

**Neden gerekli?** Aynı bitkinin/seranın/çekim gününün farklı kareleri birbirine çok benzer. Görüntü bazında **rastgele** split yaparsanız, aynı bitkinin bir karesi train'e, diğer karesi test'e düşer. Model test setini "ezberden" bildiği için mAP yapay olarak yüksek çıkar; sahada ise model hiç görmediği bitkilerle karşılaşır ve gerçek performans çok daha düşük olur. Buna **veri sızıntısı (data leakage)** denir ve ticari projelerde "laboratuvarda %95, sahada %60" hayal kırıklığının en yaygın sebebidir.

**Çözüm:** Split'i görüntü bazında değil **grup bazında** yapmak — bir bitkinin/seranın/günün TÜM görüntüleri aynı split'e gider. Bunun için **ayrı bir arayüze gerek yoktur**; grup bilgisi iki yoldan verilebilir:

1. **Dosya adı deseni** (önerilen): Çekim yaparken dosyaları `sera1_bitki05_001.jpg` gibi isimlendirin. Script varsayılan olarak son `_sayı` ekini atarak grubu bulur (`sera1_bitki05`). Özel desen için `--group-regex` kullanın:
   ```bash
   python scripts/split_dataset.py --input datasets/processed --output datasets/split \
       --group-regex "^(sera\d+_bitki\d+)"
   ```

2. **Metadata CSV**: Dosya adları düzensizse `filename,group` kolonlu bir CSV hazırlayın:
   ```bash
   python scripts/split_dataset.py --input datasets/processed --output datasets/split \
       --metadata-csv metadata.csv
   ```

> 💡 Arayüz isterseniz Roboflow'un **tag/batch** özelliği ile görüntüleri gruplayıp split'i orada da yönetebilirsiniz — ama script yeterlidir ve süreci otomatikleştirir.

⚠️ Script, her görüntü kendi grubu olarak kalırsa (dosya adları desene uymuyorsa) sizi uyarır — bu durumda sızıntı koruması sağlanmaz.

#### 🍃 Sağlıklı (Background) Görüntü Ekleme (`add_background_images.py`)

**Neden gerekli?** Dataset'teki 7 sınıfın hepsi hastalıktır; "sağlıklı bitki" örneği yoksa model, sağlıklı bir yaprakta bile hastalık uydurabilir (yanlış alarm). Ticari üründe çiftçinin güvenini en çok yanlış alarmlar bozar. Çözüm: eğitim setinin **~%10'u etiketsiz sağlıklı görüntü** olmalıdır (Ultralytics önerisi). Model bu görüntülerde "hiçbir şey tespit etmeme"yi öğrenir. YOLO formatında background görüntü = görüntü + boş `.txt` label dosyası; script bunu otomatik yapar ve %10 hedef oranını korur.

**Sağlıklı görüntü kaynağı (proje dataset'i):**

- Workspace: `test-ydoer`
- Project: `strawberry-healthy-r0op7` (sağlıklı çilek yaprağı/bitkisi görüntüleri)
- URL: https://app.roboflow.com/test-ydoer/strawberry-healthy-r0op7

```bash
# 1) Sağlıklı dataset'i indir (private workspace — kendi API key'iniz gerekir)
python scripts/download_dataset.py --api-key YOUR_KEY \
    --workspace test-ydoer --project strawberry-healthy-r0op7 --version 1 \
    --output datasets/healthy

# 2) Eğitim setine background olarak ekle (varsa etiketleri otomatik yok sayılır,
#    boş label ile kopyalanır — yani "healthy" sınıfı diye bir sınıf OLUŞMAZ)
python scripts/add_background_images.py --dataset datasets/split --source datasets/healthy --target-ratio 0.10
```

> ⚠️ İndirme için Roboflow'da dataset'in bir **version**'ının oluşturulmuş olması gerekir (Generate → Create). Version numarası farklıysa `--version` parametresini güncelleyin. Script hem YOLO düzenini (`images/train`) hem Roboflow export düzenini (`train/images`) otomatik tanır.

#### ⚖️ Sınıf Dengesizliği: Hedefli Augmentasyon (`augment_by_class.py`)

**Neden gerekli?** Birleşik dataset'te sınıflar dengesizdir: `Anthracnose Fruit Rot` 326 kutu iken `strawberry_ripe` 5.162 kutu (~16 kat). **~10 katı aşan dengesizlikte** model az örnekli sınıfı görmezden gelmeye başlar. Tüm dataset'i eşit çoğaltmak dengesizliği aynen korur; çözüm **sadece az örnekli sınıfları içeren görüntüleri** çoğaltmaktır.

**Script'in uyguladığı kurallar ve nedenleri:**

- **Sadece train çoğaltılır** — val/test'e augmentasyon metrikleri yapay şişirir.
- **Her kopya farklı dönüşümle üretilir** — birebir kopya modeli ezberletir, katkısı yoktur.
- **Renk/ton (hue) kaydırma yapılmaz** — hastalık ayrımı renge dayanır (kahverengi leke / gri küf / beyaz külleme); renk bozulursa sınıf sinyali yok olur. Kullanılan dönüşümler: yatay çevirme, ±15° döndürme, parlaklık/kontrast, hafif ölçek/kaydırma, hafif blur ve gürültü (farklı kamera kalitelerini taklit eder).
- **Çıktı ayrı klasöre yazılır** (`dataset/augmented_train/`) — orijinal veri bozulmaz; klasörü silip farklı çarpanlarla yeniden üretebilirsiniz.
- **Çarpan sınırı ~5x** — aynı kaynaktan türetilen kopyalarda getiri hızla düşer, ezber riski artar.

Varsayılan çarpanlar (birleşik 10 sınıf düzeni):

| Sınıf | Mevcut kutu | Çarpan | Hedef |
|---|---|---|---|
| 1 Anthracnose Fruit Rot | 326 | 4x | ~1.300 |
| 2 Blossom Blight | 600 | 3x | ~1.800 |
| 5 Powdery Mildew Fruit | 590 | 3x | ~1.800 |
| 0 Angular Leafspot | 1.074 | 2x | ~2.100 |
| 3 Gray Mold | 1.190 | 2x | ~2.400 |
| 8 strawberry_semi_ripe | 1.402 | 2x | ~2.800 |

```bash
# Varsayılan çarpanlarla üret ve data.yaml'ın train listesine otomatik ekle
python scripts/augment_by_class.py --update-data-yaml

# Özel çarpanlarla
python scripts/augment_by_class.py --factors "1:5,2:3,5:3" --update-data-yaml
```

> 💡 **Augmentasyon yeni bilgi yaratmaz** — 326 vakayı çeşitlendirir ama 1.300 gerçek vaka gibi olmaz. Eğitim sonrası confusion matrix'te hedeflenen sınıfların recall'u hâlâ düşükse çözüm daha fazla çoğaltma değil, o sınıflar için **gerçek veri toplamaktır**.

### 📂 Eğitimde Dosyalar Modele Nasıl Verilir?

Görüntüler **kopyalanmaz veya tek klasörde toplanmaz.** [configs/strawberry_data.yaml](configs/strawberry_data.yaml), Ultralytics'in çoklu dizin desteğiyle 4 kaynağı + augment çıktısını **liste olarak** gösterir; eğitim bunları birlikte okur:

```yaml
train:
  - "../dataset/Strawberry Disease Detection Dataset.v4i.yolo26/train/images"   # hastalık
  - "../dataset/Strawberry -ripe - unripe.yolo26/train/images"                  # olgunluk
  - "../dataset/Strawberry -ripe - unripe-r1.yolo26/train/images"               # olgunluk (3 seviye)
  - "../dataset/Strawberry - Healthy.yolo26/train/images"                       # sağlıklı → background
  - "../dataset/augmented_train/images"                                          # sınıf hedefli augmentasyon
val:  [ ... 3 kaynağın valid/images'ı ... ]     # augment ve background YOK — metrikler dürüst kalsın
test: [ ... 3 kaynağın test/images'ı ... ]
```

**Label'lar otomatik bulunur:** Ultralytics, görüntü yolundaki `/images/` bölümünü `/labels/` ile değiştirip aynı adlı `.txt` dosyasını arar. Bu yüzden her kaynakta `train/images` ile `train/labels` **kardeş dizin** olmak zorundadır — mevcut yapı buna uygundur.

Doğrulanmış mevcut durum:

| Split | Görüntü | Label | Not |
|---|---|---|---|
| train | 9.343 | 9.343 | 200'ü background (sağlıklı, boş label) |
| val | 1.341 | 1.341 | augmentasyon içermez |
| test | 515 | 515 | augmentasyon içermez |

⚠️ **`--data` her zaman MUTLAK yol olmalı.** Ultralytics dataset kökünü `data.yaml`'ın bulunduğu dizinden türetir; göreli yol verilirse kökü kendi `DATASETS_DIR`'i altında arar ve *"images not found"* hatası verir. [train_yolo.py](scripts/train_yolo.py) bunu artık otomatik yapar (`Path(data_yaml).resolve()`), ancak `yolo` CLI'ını doğrudan kullanırsanız mutlak yol verin:

```bash
yolo train data=/tam/yol/configs/strawberry_data.yaml model=yolo26s.pt
```

⚠️ **Windows'ta 260 karakter yol sınırı:** Roboflow dosya adları çok uzun olabilir; mutlak yol 260 karakteri aşarsa Python o dosyayı açamaz ve eğitim hata verir. Bu depodaki 28 dosya bu yüzden kısa adlarla yeniden adlandırılmıştır (`lp_<hash>`). Yeni veri eklerken kontrol edin:

```bash
find "$PWD/dataset" -type f \( -iname "*.jpg" -o -name "*.txt" \) | awk 'length($0)>=245'
```
Sonuç boş değilse dosya adlarını kısaltın (veya depoyu `C:\SFD` gibi kısa bir köke taşıyın). Linux/Colab'de bu sınır yoktur.

### 2. Model Eğitimi

```bash
# Config dosyası ile eğitim
python scripts/train_yolo.py --data configs/strawberry_data.yaml --config configs/train_config.yaml

# Komut satırı parametreleri ile
python scripts/train_yolo.py --data datasets/processed/data.yaml --epochs 100 --batch 16 --model yolov8s.pt
```

### 3. Model Değerlendirme

```bash
python scripts/evaluate_model.py --model runs/train/strawberry_exp/weights/best.pt --data configs/strawberry_data.yaml
```

#### 📏 Ticari Değerlendirme Metrikleri — mAP Tek Başına Yetmez

mAP, akademik karşılaştırma için iyi bir özet metriktir ama **ticari karar** için yetersizdir; tüm sınıfların ortalamasını aldığı için kritik zayıflıkları gizler. Ürün kararlarını şu metriklerle verin:

| Metrik | Neden kullanıyoruz? |
|---|---|
| **Sınıf bazında Precision / Recall** | mAP ortalaması iyi görünürken tek bir hastalıkta (örn. erken evre Gray Mold) recall %40 olabilir. Ticari değer sınıf bazında ölçülür: hangi hastalığı kaçırıyoruz? |
| **Confusion Matrix (karışıklık matrisi)** | Hangi hastalığın hangisiyle karıştığını gösterir (örn. Powdery Mildew Leaf ↔ Leaf Spot). Karışan sınıflar farklı ilaçlama gerektiriyorsa yanlış teşhis çiftçiye maddi zarar verir. |
| **Sağlıklı bitkilerde yanlış alarm oranı (FPR)** | Sadece sağlıklı görüntülerden oluşan ayrı bir test seti tutun. Yanlış alarm, gereksiz ilaçlama maliyeti ve güven kaybı demektir — ürünün terk edilmesinin 1 numaralı sebebi. |
| **Sabit, dokunulmayan saha test seti** | Hedef ortamdan (kendi seranız/tarlanız) toplanmış, hiçbir zaman eğitime karışmayan sabit bir set. Her model versiyonu AYNI setle ölçülür; yoksa "iyileşme" gerçek mi, test seti mi değişti bilemezsiniz. |
| **İş metriği (hedef eşikler)** | Örn: "Erken evre Gray Mold'u ≥%90 recall ile yakala, sağlıklıda yanlış alarm ≤%5." Model versiyonunu yayınlama kararı bu eşiklerle verilir, mAP ile değil. |

> 💡 Ultralytics `val` çıktısındaki `confusion_matrix.png` ve sınıf bazlı P/R tablosu bu analizlerin başlangıç noktasıdır; sağlıklı-set FPR ölçümü için `add_background_images.py` ile hazırladığınız sağlıklı görüntüleri ayrı bir değerlendirme setinde tutun.

#### 🔬 SAHI ile Dilimleyerek Inference (`sahi_predict.py`)

**Neden gerekli?** Sahada çekilen görüntüler yüksek çözünürlüklüdür (örn. 4000×3000). YOLO bu görüntüyü `imgsz` boyutuna (640/1024) **küçülterek** işler; erken evre hastalık lezyonları (birkaç mm'lik leaf spot lekeleri) bu küçültmede birkaç piksele düşer ve model onları göremez. Halbuki **erken evre tespiti ticari üründeki en değerli özelliktir** — hastalık yayılmadan müdahale şansı verir.

**SAHI (Slicing Aided Hyper Inference)** görüntüyü örtüşen dilimlere böler, her dilimi tam çözünürlükte modele verir ve sonuçları birleştirir. Bedeli: inference süresi dilim sayısı kadar artar — gerçek zamanlı video için değil, **fotoğraf bazlı analiz** için uygundur.

```bash
# Tek yüksek çözünürlüklü saha fotoğrafı
python scripts/sahi_predict.py --model runs/train/strawberry_exp/weights/best.pt --source field_photo.jpg

# Dizindeki tüm görüntüler
python scripts/sahi_predict.py --model best.pt --source datasets/field_photos --slice-size 640 --overlap 0.2
```

> 💡 `--slice-size` eğitimdeki `imgsz` ile aynı olmalıdır. `--overlap 0.2` dilim sınırına denk gelen lezyonların kaçmamasını sağlar.

## 🏗️ Proje Yapısı

```
SmartFarmBerry/
├── strawberry_vision/           # Ana uygulama paketi
│   ├── presentation/            # Görselleştirme katmanı
│   │   └── visualizer.py
│   ├── application/             # Uygulama katmanı
│   │   └── pipeline.py
│   ├── domain/                  # Domain katmanı
│   │   ├── entities.py
│   │   └── services.py
│   ├── infrastructure/          # Altyapı katmanı
│   │   ├── detectors.py
│   │   └── sources.py
│   └── main.py                  # Giriş noktası
│
├── configs/                     # Konfigürasyon dosyaları
│   ├── strawberry_data.yaml     # Dataset config (10 sınıf, çoklu kaynak)
│   ├── train_config.yaml        # Eğitim parametreleri
│   └── class_aliases.yaml       # Sınıf adı eş anlamlıları (birleştirme için)
│
├── scripts/                     # Yardımcı scriptler
│   ├── download_dataset.py      # Dataset indirme
│   ├── merge_datasets.py        # Çoklu kaynak birleştirme (sınıf ID çakışmasız)
│   ├── split_dataset.py         # Grup bazlı (veri sızıntısız) train/val/test split
│   ├── add_background_images.py # Sağlıklı (background) görüntü ekleme
│   ├── augment_by_class.py      # Sınıf hedefli augmentation (dengesizlik giderici)
│   ├── train_yolo.py            # Model eğitimi
│   ├── evaluate_model.py        # Model değerlendirme
│   └── sahi_predict.py          # SAHI ile dilimli inference (küçük lezyonlar)
│
├── tests/                       # Test dosyaları
│   ├── test_domain_entities.py
│   ├── test_domain_services.py
│   ├── test_application_pipeline.py
│   └── smoke_test.py
│
├── docs/                        # Dokümantasyon
│   ├── INDEX.md                 # Dokümantasyon ana sayfa
│   ├── USAGE.md                 # Kullanım kılavuzu
│   ├── architecture.md          # Mimari tasarım
│   ├── development-rules.md     # Geliştirme kuralları
│   ├── 1-gorunuAnalizi.md       # Dataset stratejisi
│   ├── 2-YOLOegitimiHiperparametre.md
│   ├── 2.1-roboflowEtiketlemeTalimati.md
│   ├── 2.2-ModelHataAnaliziIyilestirmePromptu.md
│   └── 3-RoboflowDatasetKullanimi.md
│
├── requirements.txt             # Python bağımlılıkları
├── StrawberryVision_Colab_Production.ipynb  # Colab eğitim notebook'u
└── README.md                    # Bu dosya
```

## 🧪 Test

```bash
# Tüm testleri çalıştır
pytest tests/

# Coverage ile
pytest --cov=strawberry_vision tests/

# Belirli bir test dosyası
pytest tests/test_domain_entities.py -v
```

## 📚 Dokümantasyon

Detaylı dokümantasyon için `docs/INDEX.md` dosyasına bakın:

- **Kullanım Kılavuzu**: Kurulum, çalıştırma, örnekler
- **Mimari Tasarım**: Katmanlı mimari, bağımlılıklar, veri akışı
- **Geliştirme Kuralları**: SOLID prensipleri, kod stili, test stratejisi
- **Model Eğitimi**: Dataset hazırlama, eğitim, değerlendirme
- **Roboflow Kullanımı**: Dataset linkleri, augmentation, best practices

## 🎨 Katmanlı Mimari

Proje Domain-Driven Design prensiplerine göre 4 katmana ayrılmıştır:

### 1. Domain Katmanı
- **entities.py**: `Ripeness`, `Detection`, `Strawberry` varlıkları
- **services.py**: `TrackingService`, `CountingService`
- Saf iş kuralları, harici bağımlılık yok

### 2. Infrastructure Katmanı
- **detectors.py**: YOLO detector, ripeness classifier
- **sources.py**: `ImageSource`, `VideoSource`, `CameraSource`
- Model, veri kaynakları, I/O işlemleri

### 3. Application Katmanı
- **pipeline.py**: `InferencePipeline`
- Orkestrasyon, loglama, metrik toplama
- Katmanlar arası koordinasyon

### 4. Presentation Katmanı
- **visualizer.py**: `Visualizer`
- Bounding box çizimi, sonuç kaydetme, overlay

## 🔧 Konfigürasyon

### Dataset Config (strawberry_data.yaml)
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 7
names: ['Angular Leafspot', 'Anthracnose Fruit Rot', 'Blossom Blight', 'Gray Mold', 'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf']
```

### Eğitim Config (train_config.yaml)
```yaml
model: yolo26s.pt  # YOLO26: NMS-free, DFL-free güncel Ultralytics mimarisi
epochs: 200        # Uzun eğitim + early stopping (patience: 50)
batch: 8           # imgsz 1024 için; 640'a dönerseniz 16 yapın
imgsz: 1024        # Erken evre lezyonlar 640'ta kayboluyor
optimizer: auto    # Ultralytics, YOLO26 için uygun optimizer'ı (MuSGD) ve lr'yi seçer
cos_lr: true       # Uzun eğitimde daha iyi yakınsama
hsv_s: 0.5         # Renk hastalık sinyalidir; agresif augmentasyon ipuçlarını bozar
# ... (detaylar ve gerekçeler için config dosyasındaki yorumlara bakın)
```

**Neden bu değerler?**
- `model: yolo26s.pt` — YOLO26 en güncel Ultralytics mimarisidir: NMS-free (edge/mobil export daha basit ve hızlı), küçük nesnelerde iyileştirilmiş performans. `s` boyutu, küçük hastalık lezyonlarında `n`'den belirgin daha doğrudur ve Colab T4'te rahat eğitilir; telefon/edge dağıtımında `n` deneyin. Dataset formatı YOLOv8 ile aynıdır — veri tarafında hiçbir değişiklik gerekmez.
- `optimizer: auto` — YOLO26 ile gelen MuSGD dahil, mimariye uygun optimizer ve öğrenme oranını Ultralytics seçer. Manuel kontrol isterseniz `AdamW` + `lr0: 0.002` kullanın (0.01 SGD içindir, AdamW ile eğitimi bozar).
- `imgsz: 1024` — Hastalıklar küçük lekeler halinde başlar; 640'a küçültmede erken evre lezyonlar birkaç piksele düşüp kaybolur. VRAM yetmezse 640 + SAHI kullanın.
- `cos_lr: true` + `epochs: 200` — Fine-tune senaryosunda kosinüs LR azalması ile uzun eğitim, sabit adımlı kısa eğitimden daha iyi yakınsar; `patience: 50` gereksiz uzamayı keser.
- `hsv_s: 0.5`, `hsv_h: 0.02` — Hastalık teşhisinde renk (kahverengi leke, gri küf, beyaz külleme) ayırt edici özelliktir; agresif renk augmentasyonu sınıflar arası renk ipuçlarını yok eder.
- Not: YOLO26 DFL-free ve NMS-free olduğu için config'deki `dfl` ve inference `iou` eşiği YOLO26'da etkisizdir; YOLOv8'e dönerseniz tekrar geçerli olurlar.

## 🚀 Ürünleşme ve Dağıtım

Prototipten ticari ürüne geçiş için yol haritası:

### 1. Dağıtım Hedefine Karar Verin

| Hedef | Teknoloji | Uygun model |
|---|---|---|
| 📱 Telefon uygulaması (çiftçi fotoğraf çeker) | TFLite / CoreML export + quantization | yolo26n / yolo26s |
| 📷 Sera içi sabit kamera / edge cihaz | NVIDIA Jetson + TensorRT | yolo26n / yolo26s |
| ☁️ Bulut API (toplu analiz, drone görüntüleri) | FastAPI + Docker + ONNX Runtime | yolo26m / yolo26l |

> 💡 YOLO26'nın NMS-free mimarisi tam da bu dağıtım senaryoları için avantajlıdır: export edilen modelde NMS son-işleme adımı olmadığından mobil/edge cihazlarda entegrasyon daha basit, inference daha hızlı ve gecikme daha öngörülebilirdir.

Mevcut katmanlı mimari (infrastructure katmanı ayrık) buna uygundur; eksik olan export/serving katmanıdır:

```bash
# Ultralytics export örnekleri
yolo export model=best.pt format=onnx      # Bulut API için
yolo export model=best.pt format=tflite    # Mobil için
yolo export model=best.pt format=engine    # Jetson/TensorRT için
```

### 2. Sürekli İyileştirme Döngüsü Kurun (Asıl Ticari Değer)

Modelin sahada kalıcı üstünlüğü tek seferlik eğitimden değil, şu döngüden gelir:

```
Sahada tahmin → düşük güvenli/yanlış tahminleri otomatik topla
    → uzman (ziraat mühendisi) etiketlesin → yeniden eğit
    → sabit test setiyle karşılaştır → versiyonla dağıt → (başa dön)
```

- **Model versiyonlama**: MLflow veya Weights & Biases — hangi model hangi veriyle eğitildi, izlenebilir olmalı.
- **Veri versiyonlama**: DVC veya Roboflow — test seti asla değişmemeli, eğitim seti büyümesi kayıtlı olmalı.
- **Neden?** Rakipleriniz de aynı public dataset'e erişebilir; sizin sahanızdan akan gerçek veri ve düzeltme döngüsü kopyalanamaz.

### 3. Teşhisin Yanına Eylem Ekleyin

Çiftçi "Gray Mold %87" değil, **"ne yapmalıyım?"** cevabını ister. Hastalık → kültürel önlem / ilaçlama önerisi eşlemesi ürünün asıl değeridir. ⚠️ İlaç önerisi veriyorsanız zirai danışmanlık mevzuatını ve sorumluluk reddi (disclaimer) metnini bir uzmanla netleştirin.

### 4. Lisans Kontrolü (Ticari Kullanım Öncesi Zorunlu)

- **YOLOv8 (Ultralytics) AGPL-3.0'dır**: kapalı kaynak ticari üründe ya tüm kodu açmanız ya da [Ultralytics Enterprise lisansı](https://www.ultralytics.com/license) almanız gerekir. Alternatif Apache-2.0 modeller: RF-DETR, YOLOX, D-FINE.
- Roboflow dataset'i **CC BY 4.0**: ticari kullanım serbest, atıf zorunlu.

### 5. Pilot Müşteri

1–2 sera/üretici ile ücretsiz pilot: gerçek saha verisi + kullanıcı geri bildirimi toplamanın en hızlı yoludur. Pilot verisi hem modeli (fine-tune) hem ürünü (UX) olgunlaştırır.

## 📊 Sınıflar (Birleşik 10 Sınıf Düzeni)

`dataset/` altındaki 4 kaynak, tek sınıf düzeninde birleştirilmiştir. Etiket ID'leri kaynak dosyalarda **yerinde yeniden yazılmıştır** — ID çakışması yoktur:

| ID | Sınıf | Tür | Kaynak |
|---|---|---|---|
| 0 | Angular Leafspot | Hastalık | Disease Detection v4 (değişmedi) |
| 1 | Anthracnose Fruit Rot | Hastalık | Disease Detection v4 (değişmedi) |
| 2 | Blossom Blight | Hastalık | Disease Detection v4 (değişmedi) |
| 3 | Gray Mold | Hastalık | Disease Detection v4 (değişmedi) |
| 4 | Leaf Spot | Hastalık | Disease Detection v4 (değişmedi) |
| 5 | Powdery Mildew Fruit | Hastalık | Disease Detection v4 (değişmedi) |
| 6 | Powdery Mildew Leaf | Hastalık | Disease Detection v4 (değişmedi) |
| 7 | strawberry_ripe | Olgunluk | ripe→7 (fowax), Ripened→7 (maturity) |
| 8 | strawberry_semi_ripe | Olgunluk | Semi-Ripened→8 (maturity) |
| 9 | strawberry_unripe | Olgunluk | unripe→9 (fowax), Unripened→9 (maturity) |
| — | (Sağlıklı bitki) | **Background** | strawberry-healthy: etiketler boşaltıldı — ayrı sınıf DEĞİL, yanlış alarm azaltır |

Eğitim, [configs/strawberry_data.yaml](configs/strawberry_data.yaml) üzerinden 4 kaynağı **kopyalamadan** birlikte okur (Ultralytics çoklu dizin desteği).

> ⚠️ **Karışık etiketleme uyarısı (gerçek dünya):** Hastalık kaynağındaki görüntülerde meyveler olgunluk etiketi taşımaz; olgunluk kaynağındaki görüntülerde de hastalık etiketi yoktur. Model bu "eksik etiket" gürültüsüne rağmen pratikte iyi çalışır, ancak sınıf bazlı recall'u confusion matrix ile izleyin. Olgunluk ve hastalık performansı ticari hedefe yetmezse en temiz çözüm iki ayrı model eğitmektir (hastalık modeli + olgunluk modeli) — mevcut yapı buna hazırdır.

## 🌐 Roboflow Dataset Bilgisi

### Hastalık Dataset'i (7 sınıf)
- Workspace: `strawberry-disease`
- Project: `strawberry-disease-detection-dataset`
- Version: `4`
- License: `CC BY 4.0`
- URL: https://universe.roboflow.com/strawberry-disease/strawberry-disease-detection-dataset/dataset/4

### Sağlıklı Bitki Dataset'i (background — yanlış alarm azaltma)
- Workspace: `test-ydoer`
- Project: `strawberry-healthy-r0op7`
- URL: https://app.roboflow.com/test-ydoer/strawberry-healthy-r0op7
- Kullanım: `add_background_images.py` ile eğitim setine **etiketsiz background** olarak eklenir (ayrı sınıf oluşturulmaz) — detay için "Sağlıklı (Background) Görüntü Ekleme" bölümüne bakın.

## 🤝 Katkıda Bulunma

1. Kod yazarken `docs/development-rules.md` kurallarına uyun
2. Her değişiklik için test yazın
3. Docstring ve type hint ekleyin
4. SOLID prensiplerine uyun
5. Katman sınırlarını ihlal etmeyin

## 📝 Lisans

[Lisans bilgisi eklenecek]

## 📧 İletişim

[İletişim bilgisi eklenecek]

## 🙏 Teşekkürler

- Ultralytics (YOLOv8)
- Roboflow (Dataset platformu)
- OpenCV
- Albumentations

---

**Not**: Detaylı kullanım ve geliştirme bilgileri için `docs/` klasöründeki dokümantasyonu inceleyin.
