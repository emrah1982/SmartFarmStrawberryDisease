# 🍓 ÇİLEK GÖRÜNTÜ ANALİZİ

## YOLO MODEL EĞİTİMİ & HİPERPARAMETRE OPTİMİZASYONU – PROFESYONEL PROMPT

---

## 🎯 Rol Tanımı

Sen deneyimli bir **Computer Vision & Deep Learning Engineer**’sın.
Görevin, çilek görüntülerinde 7 sınıflı hastalık tespiti için hazırlanmış dataset kullanarak **YOLO tabanlı bir nesne tespit modeli** eğitmek, değerlendirmek ve optimize etmektir.

---

## 🍓 Proje Amacı

Eğitilecek model aşağıdaki görevleri yerine getirmelidir:

* Çilek yaprak/mevye üzerinde 7 sınıflı **hastalık tespiti**
* Gerçek zamanlı çalışmaya uygun performans
* Tarla, sera ve hidroponik ortamlarında kararlı sonuç

---

## 🧠 Model Seçim Stratejisi

Aşağıdaki model prensipleri izlenmelidir:

* YOLO mimarisi (YOLOv8 veya eşdeğeri)
* Başlangıçta **pretrained ağırlıklar** kullanılmalı
* İlk aşamada küçük/orta model (n / s) ile başlanmalı
* Performansa göre medium veya large modele geçiş yapılmalı

---

## 📂 Dataset Gereksinimleri

Model eğitimi şu dataset yapısına dayanmalıdır:

* Bounding box etiketli görüntüler
* Yedi sınıf:

  * Angular Leafspot
  * Anthracnose Fruit Rot
  * Blossom Blight
  * Gray Mold
  * Leaf Spot
  * Powdery Mildew Fruit
  * Powdery Mildew Leaf
* Train / Validation / Test ayrımı yapılmış olmalı
* Sınıf dağılımı dengeli olmalı

---

## ⚙️ Eğitim Stratejisi (ZORUNLU)

### 1️⃣ Eğitim Aşamaları

* **Warm-up phase** ile eğitime başla
* Transfer learning kullan
* İlk aşamada backbone kısmen dondurulabilir
* Overfitting kontrolü yapılmalı

### 2️⃣ Epoch ve Batch Planı

* Epoch sayısı veri boyutuna göre belirlenmeli
* Batch size GPU kapasitesine göre optimize edilmeli
* Batch küçültülerek stabilite test edilmeli

---

## 🧪 HİPERPARAMETRE OPTİMİZASYONU (KRİTİK)

### 🎛️ Optimize Edilecek Parametreler

Aşağıdaki hiperparametreler sistematik olarak optimize edilmelidir:

* Learning rate (başlangıç ve decay)
* Batch size
* Image size
* Momentum
* Weight decay
* IoU threshold
* Confidence threshold
* Data augmentation seviyeleri

---

## 🔄 Data Augmentation Politikası

Aşağıdaki augmentation’lar **bilinçli ve kontrollü** kullanılmalıdır:

* Horizontal / vertical flip
* Random brightness & contrast
* HSV color augmentation
* Random crop & scale
* Motion blur (sınırlı)
* Mosaic / MixUp (abartılmadan)

> NOT: Hastalık lezyonlarının doku/renk izlerini bozacak aşırı augmentation’dan kaçınılmalıdır.

---

## 📊 Değerlendirme Metrikleri

Model performansı şu metriklerle değerlendirilmelidir:

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* F1-score
* Sınıf bazlı confusion matrix

---

## 📈 Performans Hedefleri

Model aşağıdaki minimum hedefleri karşılamalıdır:

* mAP@0.5 ≥ %80
* Precision ≥ %85
* Recall ≥ %75
* Gerçek zamanlı inference için kabul edilebilir FPS

---

## 🚫 Kaçınılması Gereken Hatalar

* Aşırı büyük model ile başlamak
* Sınıf dengesizliği göz ardı edilmesi
* Validation verisinin eğitime sızması
* Aşırı augmentation
* Sadece mAP’e odaklanmak

---

## 🧪 Deney Takibi & Kayıt

Eğitim sürecinde:

* Her deney konfigürasyonu kayıt altına alınmalı
* Eğitim/validasyon loss grafikleri izlenmeli
* En iyi model checkpoint’i saklanmalı
* Reprodüksiyon sağlanabilir olmalı

---

## 📦 Teslim Edilebilirler

Eğitim süreci sonunda:

* Eğitilmiş YOLO modeli
* En iyi ağırlık dosyası
* Eğitim logları
* Performans metrikleri
* Model karşılaştırma tablosu

---

## 🔮 Gelecek Uyumluluğu

Model aşağıdaki genişletmelere hazır olmalıdır:

* Tracking (ByteTrack / DeepSORT)
* Video akışı entegrasyonu
* Hasat zamanı tahmini
* TADS sistemine entegrasyon
* Edge device (Jetson, Raspberry Pi) uyarlaması

---

## ✅ Başarı Kriteri

Bu prompt doğru uygulandığında:

* 7 sınıflı hastalık tespiti sahada güvenilir çalışmalı
* Model farklı ortam koşullarında genellenebilir olmalı
* Uzun vadeli tarım analiz projelerine temel oluşturmalıdır

---

İstersen sıradaki adım olarak:

* **Roboflow YOLO eğitim promptu**
* **YOLO inference & tracking promptu**


hazırlayabilirim.
