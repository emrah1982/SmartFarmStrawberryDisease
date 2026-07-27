# 🍓 Strawberry Vision - Çilek Görüntü Analiz Sistemi

Google Colab uyumlu, katmanlı mimariye sahip profesyonel çilek hastalık tespit ve olgunluk sınıflandırma sistemi.

## ▶️ Colab'de Tek Tıkla Eğitim

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emrah1982/SmartFarmStrawberryDisease/blob/main/StrawberryVision_Colab_Production.ipynb)

Yukarıdaki butona tıklayın → Runtime ayarlarını yapın → **Run all**. Ayrıntılar için
[Colab Akışı (Adım Adım)](#-colab-akışı-adım-adım) bölümüne bakın.

## 🎯 Özellikler

- ✅ YOLO26 tabanlı çilek tespiti (Ultralytics — NMS-free, edge-dostu güncel mimari; YOLOv8 ile geriye uyumlu)
- ✅ Çilek hastalık tespiti (7 sınıf) + olgunluk (3 sınıf) — birleşik 10 sınıflı model
- ✅ Sağlıklı bitki background örnekleri (yanlış alarm azaltma)
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

### 🚀 Colab Akışı (Adım Adım)

Eğitim tamamen Colab'de yapılır; bilgisayarınızda kurulum gerekmez.

#### 1) Notebook'u aç — üç yol var

**a) GitHub'dan doğrudan (en kolay, önerilen)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emrah1982/SmartFarmStrawberryDisease/blob/main/StrawberryVision_Colab_Production.ipynb)

Link her zaman `main` dalındaki güncel sürümü açar:
```
https://colab.research.google.com/github/emrah1982/SmartFarmStrawberryDisease/blob/main/StrawberryVision_Colab_Production.ipynb
```

**b) Colab menüsünden:** File → Open notebook → **GitHub** sekmesi → `emrah1982/SmartFarmStrawberryDisease` → notebook'u seç

**c) Dosyayı yükleyerek:** File → Upload notebook → `StrawberryVision_Colab_Production.ipynb`

> 💡 GitHub'dan açılan notebook salt-okunurdur; `OVERRIDES` gibi değişiklikleri
> saklamak isterseniz **File → Save a copy in Drive** yapın.

#### 2) Runtime ayarı (Colab Pro+)

Runtime → Change runtime type:
- Hardware accelerator: **A100 GPU** (yoksa L4)
- Runtime shape: **High-RAM** → A100'de `cache='ram'` açılır, epoch süresi düşer
- Runtime menüsü → **Background execution** açık olsun: tarayıcıyı kapatsanız bile eğitim sürer

#### 3) Dataset'i Drive'a yükle (yalnızca bir kez)

Dataset GitHub deposunda **yoktur** (418 MB). Bilgisayarınızda `dataset_colab.zip`
oluşturup Drive'da şu konuma yükleyin:

```
MyDrive/SmartFarmStrawberryDisease/dataset/dataset_colab.zip
```

Zip'i yeniden üretmek gerekirse (repo kökünde):
```bash
python -c "import zipfile;from pathlib import Path;out=Path.home()/'Downloads/dataset_colab.zip';zf=zipfile.ZipFile(out,'w',zipfile.ZIP_STORED,allowZip64=True);[zf.write(p,p.as_posix()) for p in Path('dataset').rglob('*') if p.is_file() and 'zip dosyalar' not in p.parts];zf.close();print(out)"
```

#### 4) Runtime → Run all

Notebook sırayla: paketleri kurar ve uyumluluğu doğrular → Drive'ı bağlar →
GitHub'dan projeyi çeker → zip'i yerel diske açar → dizin ve label eşleşmesini
kontrol eder → GPU'ya göre `batch`/`workers`/`cache` ayarlar → eğitir →
sınıf bazlı metrikleri tablolar.

Tek elle müdahale: Drive bağlama hücresi bir kez yetki onayı ister.

Sonuçlar doğrudan Drive'a yazılır:
```
MyDrive/SmartFarmStrawberryDisease/
├── results/strawberry_exp/     # grafikler, confusion matrix, weights/
└── best_models/best_strawberry_exp.pt
```

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
│   ├── collect_field_data.py    # Saha görüntülerini ön-etiketle + önceliklendir
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

- **[Yol Haritası](docs/YOL-HARITASI.md)**: bugünkü durum, sonraki adımlar, ticari/hukuki notlar

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

## 🖥️ Web Arayüzü (Sahada Kullanım)

Eğitilmiş modeli günlük kullanıma açan yerel web uygulaması. Telefondan fotoğraf/video
yükleyin veya IP kameradan anlık görüntü alın; sonuçlar veritabanına kaydedilsin.

### Kurulum ve çalıştırma

```bash
pip install -r requirements-app.txt

# Eğitilmiş modeli yerleştirin
#   Drive: MyDrive/SmartFarmStrawberryDisease/best_models/best_*.pt
#   →      models/best.pt

python -m app.main
```

Açılan adresler:
- Bu bilgisayarda: `http://localhost:8000`
- **Telefondan** (aynı Wi-Fi): `http://<bilgisayarın-IP-adresi>:8000`
  (IP'yi öğrenmek için Windows'ta `ipconfig`)

### 🧭 Menü düzeni (neden böyle?)

Menü, sayfaları **kullanım sıklığına** göre üç katmana ayırır. Düz bir liste
hâlinde 9 bağlantı olduğunda hangisinin günlük iş, hangisinin bir kerelik
kurulum olduğu anlaşılmıyordu:

| Katman | Ne zaman kullanılır | Menüdeki yeri |
|--------|--------------------|---------------|
| **📷 Analiz · 🕘 Kayıtlar · 📊 Durum · 🗺️ Harita** | Her gün — görüntü ver, sonuca bak, yaygınlığı izle | Doğrudan görünür |
| **🎓 Model** (Onay bekleyenler, Etiketlerim) | Haftada birkaç kez — modeli iyileştirme döngüsü | Açılır grup, **bekleyen iş sayısı rozetle** gösterilir |
| **⚙️ Ayarlar** (Üretici & Sera, Kameralar, Veritabanı) | Kurulumda bir kez | Açılır grup |

Tasarım kararları:
- **Rozet = yapılacak iş.** İnceleme bekleyen kayıt varsa sayı menüde görünür;
  kullanıcı "bugün onaylanacak bir şey var mı?" diye sayfa gezmez.
- **Bulunduğun sayfa vurgulanır** (`etkin` sınıfı), ilgili grup otomatik açılır.
- **Adlar işe göre**: teknik "İnceleme/Etiketlenenler" yerine
  "Onay bekleyenler/Etiketlerim"; "Panel" yerine "Durum".
- **Modüller kendi katmanını seçer.** Bir modül `Modul(grup='ana'|'model'|'ayarlar')`
  diyerek menüde nereye düşeceğini bildirir; `base.html` değiştirilmez
  (bkz. [Konum modülü](#️-konum-ve-yaygınlık-modülü)).

#### ✏️ Tarayıcıda etiketleme (Roboflow'a gerek yok)

Her analiz kaydında **"Etiketleme ekranını aç"** düğmesi vardır; inceleme kuyruğundaki
kartlarda da doğrudan **✏️ Etiketle** bağlantısı bulunur.

- Model tahminleri **ön-etiket** olarak yüklenir — sıfırdan çizmezsiniz, düzeltirsiniz
- Boş alanda sürükleyerek yeni kutu çizilir; kutuya tıklayıp sınıfı değiştirilir veya silinir
- Dokunmatik destekli (`pointer` olayları): telefon ve tablette de çalışır
- Sınıf listesi **`configs/strawberry_data.yaml`'dan** okunur → ID'ler eğitimdekiyle birebir aynı
- Hastalık yoksa hiç kutu bırakmayın: kayıt **background örneği** olur, yanlış alarmı azaltır

**Kaydedince ne olur:** kayıt `elle_etiketlendi` olarak işaretlenir, inceleme kuyruğundan
çıkar ve eğitim verisi havuzuna girer.

**Kalıcı silme:** Etiketlenenler ve kayıt detay sayfalarındaki **🗑️ Kalıcı sil**
düğmesi, onay penceresinden sonra kaydı **tamamen** kaldırır: veritabanı satırı ve
etiket kutuları, yüklenen orijinal görüntü, sonuç görseli ve eğitim havuzundaki
kopyası. Geri alınamaz.

> Aynı görüntünün etiketlenmiş **başka bir kaydı** varsa havuz dosyası silinmez —
> o kayda ait olduğu için kalması gerekir. Son kayıt da silindiğinde havuzdan kalkar.

**Ne etiketlediğinizi görmek:** menüdeki **Etiketlenenler** sayfası tüm elle
düzeltilmiş kayıtları kutularıyla birlikte gösterir; sınıf dağılımı, toplam kutu sayısı
ve her kaydın havuza yazılıp yazılmadığı görünür. Kutular **her açılışta veritabanından
yeniden çizilir** (`/kayit/<id>/etiket-onizleme.jpg`) — dosya olarak saklanmadığı için
etiketi düzelttiğinizde görsel anında güncellenir, eskimiş önizleme kalmaz.

#### 🎓 Eğitim formatında dışa aktarma

İnceleme sayfasındaki **"Etiketlenmiş kayıtları eğitim formatında dışa aktar"** düğmesi
kayıtları **tek bir birikimli klasöre** yazar — her aktarımda yeni klasör açılmaz:

```
storage/egitim_verisi/
├── images/   Sera_1_42.jpg  Sera_1_57.jpg …
├── labels/   Sera_1_42.txt  Sera_1_57.txt …   (YOLO: cls x y w h)
└── data.yaml                                   (10 sınıf — merge_datasets.py şart koşar)
```

Eğitim öncesi **tek yol** verilir; klasörleri elle toplamak gerekmez:

```bash
python scripts/merge_datasets.py --inputs dataset storage/egitim_verisi --output dataset_v2
python scripts/split_dataset.py --input dataset_v2 --output dataset_v2_split
```

Dosya adının sera ile başlaması bilinçlidir: `split_dataset.py` grubu buradan çıkarır ve
aynı seranın görüntülerini train/test'e bölmez — [veri sızıntısı](#-veri-sızıntısını-önleme-grup-bazlı-split-split_datasetpy) önlenir.

**Kopya önleme:** Dosya adı görüntünün **içerik hash'ine** göre verilir
(`Sera_1_a43c080a21a9.jpg`). Aynı fotoğrafı iki kez yükleyip iki kez etiketlerseniz
havuzda **tek dosya** olur ve **en son etiketlenen** sürüm geçerli sayılır.

> Neden önemli: aynı görüntünün iki kopyası, üstelik çelişen etiketlerle eğitime
> girerse model aynı örneği iki kez öğrenir ve split sırasında görüntü hem train
> hem val'e düşerek **veri sızıntısı** yaratabilir.

Aynı görüntü daha önce analiz edilmişse kayıt sayfasında bilgi notu gösterilir.
`disa_aktarildi` işareti sayesinde aktarım yalnızca yeni/değişmiş kayıtları işler;
`yeniden=1` ile havuz baştan yazılabilir (eskimiş dosyalar temizlenir).

**Dış araca gönderim** (Roboflow vb.) ayrı bir klasör kullanır:
`storage/inceleme_paketi/`. Bu klasör de **her aktarımda silinip yeniden yazılır** —
tarihli anlık görüntü biriktirilmez.

> Neden: Anlık görüntü klasörleri **eskir**. Bir kayıt aktarıldıktan sonra
> etiketlenirse, klasördeki etiket dosyası veritabanıyla çelişir (eski tahmin vs
> düzeltilmiş etiket) ve hangi sürümün doğru olduğu belirsizleşir. Her iki klasör de
> (`egitim_verisi/`, `inceleme_paketi/`) **veritabanının o anki hâlinden** üretilir;
> adlandırma ikisinde de aynıdır: `<sera>_<içerik-hash>`.

#### 🔍 Ayrıntılı analiz — ölçek kaynaklı hatalı sınıflandırma

Gerçek bir saha fotoğrafında (2712×2496, yaprak güneşe karşı) yapılan ölçüm,
tek ölçekli tahminin **kararsız** olduğunu gösterdi:

| Yöntem | Sonuç |
|---|---|
| Tam görüntü, imgsz=640 | ✅ Angular Leafspot %68 |
| Tam görüntü, imgsz=1024 (varsayılan) | ❌ strawberry_unripe %42 (yanlış sınıf) |
| Tam görüntü, imgsz=1536 | ❌ yalnızca olgunluk sınıfları |
| Tam görüntü, imgsz=2048 | ❌ hiç tespit yok |
| **Ayrıntılı (çok ölçek + dilimli)** | ✅ **Angular Leafspot %68** + diğerleri |

**Sebep — domain (alan) farkı:** Hastalık dataset'indeki görüntüler **280×280**
yakın çekim kırpmalarıdır. Saha fotoğrafı 6,8 MP'lik geniş bir sahnedir; tek
ölçeğe indirildiğinde lezyonların göründüğü ölçek eğitimdekinden çok farklı olur.
Sonuç, seçilen `imgsz` değerine göre savrulur — bu savrulmanın kendisi modelin
bu tür fotoğraflarda henüz güvenilir olmadığının göstergesidir.

**Ayrıntılı analiz** (arayüzdeki onay kutusu) görüntüyü birden çok ölçekte ve
büyükse örtüşen dilimler halinde tarar, sonuçları NMS ile birleştirir. Yaklaşık
**5-8 kat yavaştır** ama ölçek kaynaklı kaybı belirgin azaltır.

> ⚠️ Bu bir **yama**dır, kalıcı çözüm değildir. Kalıcı çözüm: bu tür gerçek saha
> fotoğraflarını [inceleme kuyruğundan](#-sahadan-gelen-veriyle-sürekli-iyileştirme)
> etiketleyip eğitime katmaktır. Model ancak gördüğü türden görüntülerde güvenilirdir.

#### Görüntü kaynakları ve kalite denetimi

| Cihaz | "Fotoğraf Çek" / "Video Çek" | Not |
|---|---|---|
| 📱 Telefon | Cihazın **kamera uygulamasını** açar | HTTP üzerinden de çalışır |
| 💻 Bilgisayar | **Dosya seçici** açar | Bilgisayar kamerası için sayfadaki "kamerayı açın" bağlantısı kullanılır |

`capture="environment"` özniteliği yalnızca mobil tarayıcılarda kamerayı açar;
masaüstünde yok sayılır. Masaüstünde gerçek kamera için `getUserMedia` gerekir ve
bu API **yalnızca güvenli bağlamda** (`https` veya `localhost`) çalışır — telefondan
`http://192.168.x.x` ile bağlanıldığında tarayıcı engeller. Bu yüzden telefonda
dosya girdisi (cihazın kamera uygulaması), masaüstünde tarayıcı içi kamera kullanılır.

**Bulanıklık denetimi.** Yürürken video çekimi pratiktir ama hareket bulanıklığı
üretir; bulanık kareyi modele vermek yanlış veya eksik tespit doğurur. Her örneklenen
karenin keskinliği **Laplacian varyansı** ile ölçülür (`BULANIKLIK_ESIGI`, varsayılan 60):

- Eşiğin altındaki kareler modele **verilmez**, atlanır.
- Kaç karenin atlandığı sonuç sayfasında kullanıcıya bildirilir.
- Tüm kareler bulanıksa yine de en keskin kare işlenir — kullanıcı boş sonuç almaz.
- Tek fotoğraf bulanıksa "sabit tutarak tekrar çekin" uyarısı gösterilir.

Arayüzde çekim rehberi yer alır: mesafe (30-60 cm), ışık (sert gölge renk ipuçlarını
bozar), sabitlik ve odak. Yürüyerek çekim için "2-3 adımda bir yarım saniye durun"
önerisi verilir; duraklama anındaki kareler keskin olduğu için doğruluk belirgin artar.

#### İşletme yapısı: Üretici → Sera → Kamera

Birden çok sera ve müşteriyle çalışırken "hangi hastalık, kimin serasında, hangi
kamerada" sorusunun cevabı kayıtlarda tutulur:

```
Üretici (Ahmet Yılmaz)
└── Sera 1 (Kuzey blok, Çilek)
    ├── Kamera: Giriş
    └── Kamera: 3. sıra
└── Sera 2 (Güney blok)
    └── Kamera: Orta koridor
```

- **Kamera analizleri** sera bilgisini kameradan otomatik alır.
- **Telefon yüklemelerinde** hangi seraya ait olduğu açılır listeden seçilir.
- `sera_id` analiz kaydında **ayrıca saklanır**: kamera silinse veya başka seraya
  taşınsa bile geçmiş kaydın hangi seraya ait olduğu kaybolmaz.
- Üretici/sera/kamera silinmez, **pasife alınır** — geçmiş kayıtlar sahipsiz kalmasın.

Yönetim: **İşletmeler** sayfası (üretici + sera), **Kameralar** sayfası (kamera → sera).
Geçmiş sayfasında üretici ve sera filtreleri, Panel'de sera bazlı özet tablosu
(analiz, tespit, en sık hastalık, bekleyen inceleme) bulunur.

### 🔴 Canlı Tespit (kamerayı tut, anında gör)

`/canli` sayfası kamerayı ekranda tutar ve tespitleri **görüntünün üzerine anlık çizer** —
fotoğraf çekip yüklemeye gerek kalmaz. Sıra aralarında yürürken hastalıklı bitkiyi ekranda
görürsünüz.

#### Nasıl çalışır?

```
tarayıcı ──kare(640px JPEG)──▶ sunucu ──model──▶ kutular ──JSON──▶ tarayıcı ──çizim
    ▲                                                                    │
    └──────────────── sıradaki kare ancak sonuç gelince ◀────────────────┘
```

Tasarım kararları ve sebepleri:

- **Görüntü değil koordinat taşınır.** Sunucu kutulanmış resim değil, yalnızca kutu
  koordinatlarını (JSON, birkaç yüz bayt) döner; çizimi tarayıcı yapar. Ağdan resim
  taşımak akışı kilitlerdi.
- **Geri basınç (backpressure).** Sabit FPS ile gönderilmez; bir sonraki kare, öncekinin
  sonucu gelmeden yollanmaz. Sunucu yavaşsa kare/sn kendiliğinden düşer, **kuyruk
  birikmez**. Sabit FPS'te yavaş sunucuda görüntü saniyelerce geriden gelirdi.
- **Kare tarayıcıda küçültülür** (640 px, JPEG %60) ve model canlıda `CANLI_IMGSZ=640`
  ile çalışır — tek kare analizindeki 1024'e göre birkaç kat hızlıdır.
- **Bulanık kare modele verilmez.** Laplacian keskinliği eşiğin altındaysa kare atlanır ve
  ekranda "sabit tutun" uyarısı çıkar; hareket bulanıklığı en sık doğruluk kaybı sebebidir.
- **WebSocket engelliyse REST'e düşer** (`POST /canli/kare`). Davranış aynıdır — iki uç da
  aynı `_isle()` fonksiyonunu kullanır.
- **Sekme arkaya alınınca kamera kapanır** — pil ve mobil veri boşa gitmesin.

#### Kayıt nasıl açılır? (her kare kaydedilmez)

Saniyede birkaç kare gelirken her tespiti kaydetmek veritabanını doldurur ve tek karelik
yanlış tespitler de kayda geçerdi. Bir bulgu ancak **kararlı** hale gelince kaydedilir:

| Kural | Varsayılan | Ortam değişkeni |
|-------|-----------|-----------------|
| Aynı sınıf üst üste N karede görülmeli | 3 | `CANLI_KARARLILIK_KARE` |
| En az güven | %60 | `CANLI_KAYIT_GUVEN` |
| Aynı sınıf için bekleme | 20 sn | `CANLI_BEKLEME_SN` |
| Otomatik kaydı kapat | açık | `CANLI_OTOMATIK_KAYIT=0` |

"Üst üste" gerçekten ardışık kareleri ifade eder: bulgu bir karede kaybolursa sayaç
sıfırlanır. **💾 Bu Kareyi Kaydet** düğmesiyle istediğiniz anı elle de saklarsınız.

Kayıtlar `kaynak_tip='canli'` ile **çekirdekle aynı biçimde** açılır: geçmişte, inceleme
kuyruğunda, etiketleme ekranında ve haritada diğerleriyle birlikte görünür — ayrı bir
"canlı kayıt" kavramı yoktur.

#### ⚠️ Telefonda kamera açılmıyorsa: HTTPS gerekir

Tarayıcılar kamerayı (`getUserMedia`) **yalnızca güvenli bağlamda** verir: `https://` veya
`localhost`. Telefondan `http://192.168.x.x:8000` ile bağlandığınızda canlı kamera açılmaz —
bu tarayıcı kuralıdır, uygulamanın yapabileceği bir şey yoktur.

```bash
python scripts/https_sertifika.py     # certs/ altına kendinden imzalı sertifika üretir
docker compose up -d --build          # certs/ varsa https AYRICA açılır
# Telefondan: https://192.168.x.x:8443/canli
```

**http kapanmaz.** Sunucu iki dinleyici birden açar — mevcut adresleriniz ve yer
imleriniz kırılmaz:

| Adres | Ne için |
|-------|---------|
| `http://<ip>:8000` | Her şey: yükleme, geçmiş, etiketleme, harita (eskisi gibi) |
| `https://<ip>:8443` | Yukarıdakilerin hepsi **+ canlı kamera** |

`/canli` sayfasını http üzerinden açarsanız **🔒 Güvenli adrese geç** düğmesi çıkar;
tek tıkla https adresine geçersiniz.

#### "Bağlantınız gizli değil" uyarısı — hata değil

Sertifika bir şirketten satın alınmadığı, sizin bilgisayarınızda üretildiği için tarayıcı
onu tanımaz (`ERR_CERT_AUTHORITY_INVALID`). Trafik yine de şifrelidir.

- **Bilgisayarda sertifikaya gerek yok.** `http://localhost:8000/canli` uyarısız çalışır —
  tarayıcılar `localhost`'u zaten güvenli bağlam sayar. Uyarıyı görüyorsanız muhtemelen
  gereksiz yere https adresini açtınız.
- **Telefonda hızlı çözüm:** Uyarı ekranında **Gelişmiş → Yine de devam et**
  (her tarayıcıda bir kez).
- **Telefonda kalıcı çözüm:** `/canli/sertifika` sayfasını açıp sertifikayı cihaza kurun;
  sonrasında uyarı hiç çıkmaz. Sertifika `CA:true` olarak üretilir, bu yüzden Android'de
  *Ayarlar → Güvenlik → Sertifika yükle → CA sertifikası*, iOS'ta *Profil İndirildi* →
  ardından *Sertifika Güveni Ayarları*'ndan tam güven verilerek kurulabilir.

İnternete açık kurulumda kendinden imzalı sertifika kullanmayın; gerçek sertifika alın
(Let's Encrypt / ters vekil).

Sertifika istemiyorsanız telefonda [tek kare analizi](#-web-arayüzü-sahada-kullanım)
(📷 Fotoğraf Çek / 🎥 Video Çek) sertifikasız çalışmaya devam eder.

#### Beklenen hız

Docker'da **CPU** ile 640 px karede ölçülen: kare başına ~0.4–1.0 sn → **1–2.5 kare/sn**.
Yavaş yürüyüş hızında yeterlidir. GPU'lu makinede (`docker-compose.yml` içindeki GPU
bölümü açılarak) 10+ kare/sn'ye çıkar. Şüpheli bir bölge görürseniz durup **tek kare
analizi + Ayrıntılı analiz** yapın: canlı akış hız için düşük çözünürlük kullanır,
uzaktan/küçük lezyonlarda tek kare analizi çok daha hassastır.

#### Bileşen düzeni

Canlı akış çekirdekten farklı çalıştığı (WebSocket, geri basınç, otomatik kayıt) için ayrı
modüldür; klasör silinse uygulama çalışmaya devam eder ve menüden kendiliğinden kalkar.

```
app/moduller/canli/
├── __init__.py              # modül tanımı (menüde 'ana' grubu)
├── ayarlar.py               # eşikler/parametreler — tek yerden
├── servis.py                # SAF mantık: kare çözme, tespit, KayitKarari
├── depo.py                  # tek DB temas noktası: kaydet, sera listesi
├── rotalar.py               # sayfa + WebSocket + REST yedeği
├── templates/canli/izle.html
└── static/                  # tarayıcı bileşenleri (birbirini bilmez)
    ├── kamera.js            # kamera aç/kapa/çevir, kare üret
    ├── akis.js              # sunucuya gönder (WS → REST yedeği, geri basınç)
    ├── cizim.js             # kutuları tuvale çiz
    └── izle.js              # yapıştırıcı: üçünü bağlar
```

`servis.KayitKarari` zamanı dışarıdan alır (`simdi` parametresi), böylece **kamerasız ve
sunucusuz** test edilir — kayıt kuralının doğruluğu `tests/test_canli.py` içinde 5 testle
sabitlenmiştir.

### 🗺️ Konum ve Yaygınlık Modülü

"Hastalık **nerede** yoğunlaşmış?" sorusunu yanıtlar. Menüdeki **Yaygınlık** sayfası.

#### Konum üç yoldan gelir

| Kaynak | Nasıl | Ne zaman kullanışlı |
|---|---|---|
| **EXIF GPS** | Telefon/drone fotoğrafındaki koordinat otomatik okunur | Açık alan, tarla, drone uçuşu |
| **Kamera** | Kameraya bir kez konum tanımlanır; o kameradan gelen analizler devralır | Sabit sera kameraları |
| **Elle** | Kayıt sayfasından blok/sıra girilir | **Sera içi** — GPS hassasiyeti yetersiz kalır, "A blok / 3. sıra" daha kullanışlıdır |

#### Yaygınlık nasıl ölçülür

**Enfekte görüntü oranı** kullanılır, kutu sayısı değil:

> Tek bir yaprakta 16 leke olması o bölgeyi 16 kat sorunlu yapmaz. Asıl soru
> "bu bölgede çekilen görüntülerin yüzde kaçında hastalık var". Kutu sayısı
> ayrıca **şiddet** göstergesi olarak raporlanır.

Olgunluk sınıfları (`strawberry_*`) yaygınlık hesabına girmez — onlar hastalık değildir.

Sayfa üç görünüm sunar: bölge tablosu (yaygınlık %'sine göre sıralı), **ısı haritası**
(yeşil→kırmızı) ve GPS noktaları varsa **dağılım grafiği**. Harita karosu kullanılmaz,
yani internet olmadan da çalışır.

#### Modüler yapı — neden

Konum yeteneği `app/moduller/konum/` altında **kendi tablosu, rotaları ve şablonlarıyla**
durur; çekirdek `Analiz` tablosuna sütun eklemez (ayrı `analiz_konumlari` tablosu kullanır).

```
app/moduller/
├── __init__.py            # modül kaydı (Modul dataclass + kaydet())
└── konum/
    ├── __init__.py        # modül tanımı: ad, başlık, menü yolu, router
    ├── modeller.py        # AnalizKonum tablosu
    ├── servis.py          # EXIF GPS okuma, yaygınlık hesabı
    ├── rotalar.py         # /konum/... sayfaları
    └── templates/konum/   # kendi şablonları
```

Bunun getirisi:

- **Başka projeye taşıma:** tek klasör kopyalanır, `yuklu_moduller()` listesine eklenir
- **Kapatma:** listeden çıkarılır; çekirdek şema ve sayfalar etkilenmez
- **Menü:** modüller kendilerini bildirir, `base.html` otomatik listeler
- **Çekirdek sadeliği:** `app/main.py` modülün ayrıntısını bilmez, yalnızca
  `moduller.kaydet(app, engine)` çağırır

#### Modül eklerken

`app/moduller/<ad>/` klasörü açılır, `Modul(...)` döndüren bir `modul()` yazılır ve
`yuklu_moduller()` listesine eklenir. Çekirdekte hiçbir dosya değişmez:

| Alan | Ne işe yarar |
|------|--------------|
| `grup='ana'\|'model'\|'ayarlar'` | menüde hangi katmana düşeceği |
| `ikon`, `baslik`, `yol` | menüdeki görünümü |
| `tablolar_olustur(engine)` | kendi tablolarını kurar (varsa) |
| `statik` | kendi js/css klasörü → `/statik/<ad>/...` olarak sunulur |

> `/statik/<ad>` kullanılır, `/static/<ad>` değil: çekirdek `/static`'i zaten bağlamıştır
> ve Starlette önce onu eşleştirir; alt yol oraya düşüp 404 verirdi.

#### Drone kullanımı

Drone görüntüleri de EXIF GPS taşır; ek geliştirme gerekmez — fotoğrafları yüklemeniz
yeterlidir. Yüksekten çekimde lezyonlar küçük göründüğü için
[Ayrıntılı analiz](#-ayrıntılı-analiz--ölçek-kaynaklı-hatalı-sınıflandırma) seçeneğini
işaretlemeniz önerilir.

### 🗄️ Veritabanını görüntüleme

SQLite dosyası tek yerdedir ve üç şekilde incelenebilir:

| Yol | Nasıl |
|---|---|
| **Tarayıcıdan** | Menüdeki **Veritabanı** sayfası — tablolar, satır sayıları, sayfalanmış içerik (salt okunur) |
| **Terminalden** | `python scripts/db_incele.py` · `--tablo analizler --limit 20`<br>Docker'da: `docker exec cilek-tespit python scripts/db_incele.py --tablo tespitler` |
| **Masaüstü araçla** | `storage/kayitlar.db` dosyasını *DB Browser for SQLite* gibi bir programla açın |

> **Docker'da dosya nerede?** Konteynerde `/app/storage/kayitlar.db`,
> bilgisayarınızda `storage/kayitlar.db` — **aynı dosyadır** (volume ile bağlı).
> Konteyneri silseniz bile veriler kalır.

Görüntüleyici **salt okunurdur**: serbest SQL çalıştırılmaz, yalnızca uygulamanın
tanımladığı tablolar listelenir. Bu da `app/moduller/veritabani/` altında ayrı bir
modüldür; gerek yoksa `yuklu_moduller()` listesinden çıkarılabilir.

### 📍 Telefon fotoğraflarında konum

Telefonla çekilen fotoğrafın GPS taşıması için:

- Kamera uygulamasında **konum etiketleme açık** olmalı
  (Android: Kamera → Ayarlar → Konum etiketleri · iPhone: Ayarlar → Gizlilik → Konum Servisleri → Kamera)
- Yükleme sırasında telefonun **kendi kamera uygulaması** kullanılmalı ("📷 Fotoğraf Çek")
  veya galeriden seçilmeli. **Tarayıcı içi kamerayla** alınan kareler yeniden
  kodlandığı için EXIF taşımaz.
- iPhone'da paylaşırken **"Tüm Fotoğraf Verileri"** açık olmalı
- **WhatsApp/Telegram ile aktarılan görseller EXIF'i tamamen kaybeder**

Konum gelmezse kayıt sayfasındaki açılır bölümde bu nedenler listelenir ve blok/sıra
elle girilebilir. Sera içinde GPS zaten ±5-10 m sapar; **blok/sıra yazmak daha kullanışlıdır**.

### 🔒 Güvenlik ve çok müşterili kullanım

Uygulamada **kullanıcı girişi yoktur**; yerel ağda tek işletme için tasarlanmıştır.
Ağdaki herkes tüm üreticilerin verisini görebilir.

Altyapı çok müşterili kullanıma **hazırlanmıştır**:

- `Kullanici` tablosu tanımlı (`rol`: admin / uretici, `uretici_id` bağı)
- [app/yetki.py](app/yetki.py) tek geçiş noktasıdır: rotalar veriye doğrudan değil
  `analiz_sorgusu()`, `gorunur_seralar()`, `erisebilir_mi()` gibi yardımcılar
  üzerinden erişir. Giriş sistemi eklendiğinde **yalnızca bu dosya değişir**,
  izolasyon tüm sayfalara otomatik uygulanır.

> **Neden şimdiden:** İzolasyonu sonradan eklemek her sorguyu tek tek bulup filtre
> eklemek demektir; bir tanesini atlamak bir müşterinin verisini başkasına gösterir.

Ayrıntılı plan: [docs/YOL-HARITASI.md](docs/YOL-HARITASI.md)

### 🐳 Docker ile çalıştırma (önerilen)

Docker, Python/CUDA sürüm karmaşasını ortadan kaldırır: aynı imaj sizin
bilgisayarınızda, sunucuda veya seradaki mini PC'de birebir aynı çalışır.

```bash
# 1) Modeli yerleştirin (imaja gömülmez, dışarıdan bağlanır)
#    models/best.pt

# 2) Başlatın
docker compose up -d

# 3) Açın:  http://localhost:8000
#    Telefondan: http://<bilgisayarın-IP-adresi>:8000
```

Durdurmak: `docker compose down` · Log: `docker compose logs -f`

**Neler dışarıda tutuldu ve neden:**

| Öğe | Nerede | Neden |
|---|---|---|
| `models/best.pt` | Volume (salt-okunur) | Model sık değişir; her seferinde imaj derlemek gerekmesin |
| `storage/` | Volume | Kayıtlar, görseller ve SQLite konteyner silinse de kalır |
| `tedavi_onerileri.yaml` | Volume (salt-okunur) | Önerileri düzenleyip konteyneri yeniden başlatmak yeterli |
| `dataset/` (640 MB) | `.dockerignore` | Eğitim verisi çalışma zamanında gereksiz |

**Model güncelleme:** Yeni `best.pt` dosyasını `models/` içine kopyalayın ve
`docker compose restart` deyin — yeniden derleme gerekmez.

#### CPU mu GPU mu?

Varsayılan imaj **CPU** kullanır: torch'un CPU tekerleği ~200 MB, CUDA sürümü ~2,5 GB'dır.
Fotoğraf analizinde CPU yeterlidir (görüntü başına birkaç saniye); video ve yoğun
kullanımda GPU belirgin fark yaratır.

GPU için `docker-compose.yml` içinde iki yeri açın (dosyada yorum satırı olarak hazır):
`TORCH_INDEX` build argümanını CUDA sürümüyle değiştirin ve `deploy.resources` bloğunu etkinleştirin.
Windows'ta ek olarak **WSL2 + NVIDIA Container Toolkit** kurulu olmalıdır.

#### IP kamera erişimi

Konteyner varsayılan ağ ayarıyla yerel ağdaki kameralara erişebilir; RTSP adresinde
kameranın **IP'sini** kullanın (`localhost` konteynerin kendisini gösterir, kamerayı değil).

### Neler yapabilir

| Sayfa | İşlev |
|---|---|
| **Analiz** | Telefon kamerasıyla çekim, galeriden fotoğraf/video yükleme, IP kameradan anlık görüntü |
| **Kayıt** | Kutulanmış sonuç + sınıf bazlı özet + **hastalığa özel yönetim önerisi** |
| **Geçmiş** | Tüm analizler; sınıfa, kaynağa ve tarihe göre filtreleme |
| **İnceleme** | Modelin zorlandığı kayıtlar — ön-etiketleriyle dışa aktarılır |
| **Panel** | Toplam analiz, sınıf dağılımı, 30 günlük trend |
| **Kameralar** | RTSP/HTTP kamera tanımlama |

### Sürekli iyileştirme ile bağlantısı

Güveni `REVIEW_THRESHOLD` (varsayılan %55) altında kalan veya hiç tespit üretmeyen her
analiz **otomatik olarak inceleme kuyruğuna** düşer. Kuyruktaki kayıtları tek tuşla
ön-etiketleriyle dışa aktarır (`storage/exports/`), Roboflow'da düzeltir ve
`merge_datasets.py` ile ana dataset'e katarsınız. Böylece sahadaki her kullanım
bir sonraki modeli besler — bkz. [Sürekli İyileştirme](#-sahadan-gelen-veriyle-sürekli-iyileştirme).

### Teşhisin yanında eylem önerisi

Çiftçi "Gray Mold %87" değil **"ne yapmalıyım?"** cevabını ister. Her tespitin yanında
[configs/tedavi_onerileri.yaml](configs/tedavi_onerileri.yaml)'dan gelen etken bilgisi,
belirti ve kültürel önlem listesi gösterilir.

> ⚠️ Bu dosyada **ilaç adı ve dozu bilinçli olarak yoktur.** Ruhsatlı ilaçlar ülkeye,
> ürüne ve döneme göre değişir; yanlış tavsiye hem yasal sorumluluk hem ürün kaybı
> doğurur. Metinler kültürel önlem ve izleme tavsiyesidir; kimyasal müdahale kararı
> için ziraat mühendisine danışılmalıdır. Kendi bölgenize göre düzenleyebilirsiniz.

### Yapılandırma

Ortam değişkenleriyle ayarlanır (bkz. [app/config.py](app/config.py)):

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Model dosyası |
| `CONF_THRESHOLD` | `0.25` | Tespit güven eşiği |
| `REVIEW_THRESHOLD` | `0.55` | Altındaki tespitler inceleme kuyruğuna düşer |
| `IMGSZ` | `1024` | Inference çözünürlüğü |
| `PORT` | `8000` | Sunucu portu |
| `DATABASE_URL` | `sqlite:///storage/kayitlar.db` | PostgreSQL'e geçmek için değiştirin |

### Mimari

```
app/
├── config.py      # ayarlar (ortam değişkenleriyle geçersiz kılınır)
├── database.py    # SQLAlchemy modelleri: Analiz, Tespit, Kamera
├── detector.py    # model yükleme + görüntü/video/kamera tahmini
├── main.py        # FastAPI rotaları
├── templates/     # Jinja2 sayfaları (mobil uyumlu)
└── static/        # CSS
storage/           # yüklenenler, sonuçlar, dışa aktarımlar, SQLite (git'e girmez)
```

**Neden bu yapı:** Tahmin katmanı (`detector.py`) arayüzden bağımsızdır; ileride mobil
uygulama veya başka bir istemci eklenirse aynı API kullanılır. SQLite tek dosyadır,
kurulum gerektirmez; SQLAlchemy sayesinde PostgreSQL'e geçiş yalnızca `DATABASE_URL`
değişikliğidir.

**Testler:** `pytest tests/test_app.py` — sahte bir detector kullanır, yani ultralytics
ve eğitilmiş model olmadan da uygulama mantığı sınanabilir.

> 🔒 **Güvenlik notu:** Uygulamada kullanıcı girişi yoktur; yerel ağda kullanım için
> tasarlanmıştır. İnternete açacaksanız önce kimlik doğrulama ve HTTPS ekleyin.

## 🔄 Sahadan Gelen Veriyle Sürekli İyileştirme

Model eğitildikten sonra asıl değer burada başlar: sahada çekilen gerçek görüntüleri
toplayıp eğitime katmak. Rakipleriniz de aynı public dataset'e erişebilir; **sizin
seranızdan akan veri kopyalanamaz.**

### Döngü

```
Sahada tahmin  →  zorlanılan kareleri topla  →  uzman düzeltsin
      ↑                                              ↓
  yeni sürümü dağıt  ←  sabit test setiyle karşılaştır  ←  yeniden eğit
```

### 1) Saha görüntülerini topla ve önceliklendir (`collect_field_data.py`)

**Neden hepsini etiketlemiyoruz?** Modelin zaten %95 güvenle doğru bildiği kareyi
etiketlemek ona yeni bir şey öğretmez. Öğrenme değeri en yüksek olanlar modelin
**kararsız kaldığı** karelerdir. Bu yaklaşıma *aktif öğrenme* denir: aynı etiketleme
emeğiyle çok daha fazla kazanım sağlar.

```bash
python scripts/collect_field_data.py \
    --model runs/train/strawberry_exp/weights/best.pt \
    --images saha_fotograflari/ \
    --output saha_2026_07/
```

Script her görüntüye tahmin üretir, bunları **ön-etiket** olarak YOLO formatında
kaydeder ve üç gruba ayırır:

| Klasör | Anlamı | Ne yapmalı |
|---|---|---|
| `incele/` | Model kararsız (düşük güvenli tespit var) | **Önce bunları düzeltin** — en değerli veri |
| `otomatik/` | Tüm tespitler yüksek güvenli | Örnekleme yapıp doğrulayın |
| `tespit_yok/` | Hiç tespit yok | Sağlıklı mı, yoksa **kaçırılmış hastalık** mı? Mutlaka bakın |

`rapor.csv` her görüntü için tespit sayısı, güven değerleri ve sınıfları listeler.

> 💡 Ön-etiket sayesinde uzman sıfırdan kutu çizmez, sadece düzeltir — etiketleme
> 3-5 kat hızlanır.

### 2) Etiketleri düzelt

`incele/` klasörünü Roboflow'a yükleyin (görüntüler + `labels/` ön-etiketleri birlikte).
Roboflow bunları hazır kutular olarak gösterir; uzman yanlışları düzeltip eksikleri ekler.

⚠️ **Etiket kalitesi veri miktarından önemlidir.** Hastalık teşhisi uzmanlık ister;
etiketleri bir ziraat mühendisi/fitopatoloji uzmanına doğrulatın. Yanlış etiketli veri,
az veriden daha zararlıdır.

### 3) Ana dataset'e kat

```bash
# Düzeltilmiş veriyi indirip mevcut dataset ile birleştir
python scripts/merge_datasets.py \
    --inputs dataset/ saha_2026_07_duzeltilmis/ \
    --output dataset_v2/

# Grup bazlı böl (aynı bitki/sera iki split'e düşmesin)
python scripts/split_dataset.py --input dataset_v2 --output dataset_v2_split

# Sınıf dengesizliğini yeniden değerlendir
python scripts/augment_by_class.py --update-data-yaml
```

🔒 **Test setini DONDURUN.** Yeni veri yalnızca train (ve gerekirse val) setine girmeli.
Test seti hiç değişmezse model sürümlerini adil karşılaştırabilirsiniz; her seferinde
değişirse "iyileşme gerçek mi, test mi kolaylaştı" ayırt edemezsiniz.

📸 **Dosya adlandırma:** Saha çekimlerinde `sera1_bitki05_001.jpg` gibi bir düzen
kullanın. `split_dataset.py` grubu buradan çıkarır ve aynı bitkinin kareleri
train/test'e bölünmez (veri sızıntısı önlenir).

### 4) Yeniden eğit

İki seçenek:

```bash
# A) Sıfırdan (temiz, tercih edilen): tüm veriyle yeni model
#    Colab notebook'ta MOD = 'sifirdan'

# B) İnce ayar (hızlı): mevcut modelden devam
#    train_config.yaml içinde model: runs/train/strawberry_exp/weights/best.pt
```

Veri seti belirgin büyüdüyse (A) daha iyi sonuç verir. Küçük eklemelerde (B) hızlıdır
ama modelin eski veriye aşırı uyumu kalıcı olabilir.

### 5) Karşılaştır ve karar ver

```bash
python scripts/evaluate_model.py --model <yeni_best.pt> --data configs/strawberry_data.yaml
```

Sadece genel mAP'ye bakmayın. **Sınıf bazlı recall** ve **sağlıklı bitkide yanlış alarm
oranı** yeni sürümün gerçekten daha iyi olup olmadığını gösterir. Yeni model eskisinden
kötüyse dağıtmayın — sebebini araştırın (etiket hatası? dengesiz ekleme?).

### 6) Sürüm takibi

Her eğitim için şunları kaydedin: hangi veri anlık görüntüsüyle eğitildi, hangi config,
sabit test setindeki sonuçlar. Basit bir tablo bile yeterlidir:

| Sürüm | Tarih | Train görüntü | Test mAP50 | Anthracnose recall | Yanlış alarm |
|---|---|---|---|---|---|
| v1 | 2026-07 | 9.343 | — | — | — |
| v2 | | | | | |

Daha kurumsal bir yapı isterseniz model için **MLflow / Weights & Biases**, veri için
**DVC / Roboflow versiyonlama** kullanılabilir.

### Ne kadar veri, ne zaman?

- **Hemen:** `tespit_yok` çıkan sağlıklı bitki görüntüleri — background örneği olarak
  bedava değer üretir, yanlış alarmı düşürür.
- **Öncelikli:** Az örnekli sınıflar (Anthracnose Fruit Rot, Powdery Mildew Fruit).
  Augmentasyon bunları çeşitlendirir ama **yeni bilgi yaratmaz**; recall hâlâ düşükse
  çözüm gerçek veridir.
- **Pratik hedef:** Sınıf başına 500-1.000 gerçek saha örneği. Her toplama turunda
  200-500 görüntü etiketlemek sürdürülebilir bir tempodur.

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
