# 🍓 ÇİLEK GÖRÜNTÜ ANALİZİ

## DATASET & ETİKETLEME STRATEJİSİ – PROFESYONEL PROMPT

---

## 🎯 Rol Tanımı

Sen deneyimli bir **Computer Vision Dataset Architect & AI Trainer**’sın.
Görevin, çilek görüntülerinde HASTALIK belirtilerini tespit eden (7 sınıf) **yüksek kaliteli**, **genellenebilir**, **gerçek tarım koşullarına uygun** bir **dataset ve etiketleme stratejisi** oluşturmaktır.

---

## 🍓 Proje Kapsamı

Bu dataset aşağıdaki yapay zeka görevlerini destekleyecektir:

* Çilek yaprak/fruit üzerinde **hastalık tespiti (object detection)**
* Video bazlı **takip (tracking)** ve **sayım**
* Gerçek saha koşullarında kararlı çalışma

---

## 🧠 Temel Hedefler

Dataset şu hedefleri karşılamalıdır:

* Farklı ışık koşullarına dayanıklılık
* Farklı kamera açıları ve mesafeler
* Farklı çilek çeşitleri
* Yaprak, gölge, toprak gibi gürültülere karşı tolerans
* Modelin **overfitting yapmasını engelleyecek çeşitlilik**

---

## 🦠 Hastalık Sınıfları (7)

- Angular Leafspot
- Anthracnose Fruit Rot
- Blossom Blight
- Gray Mold
- Leaf Spot
- Powdery Mildew Fruit
- Powdery Mildew Leaf

Etiketleme ilkeleri: Lezyon merkezli kutular; aynı yapraktaki farklı lezyonlar ayrı ayrı kutulanır; belirsiz vakalar QA listesine alınır.

---

## 📊 Veri Türleri

Dataset aşağıdaki veri türlerini içermelidir:

### 1️⃣ Görüntü Türleri

* RGB görüntüler
* Yüksek çözünürlük (tercihen ≥1280x720)
* Video karelerinden elde edilen frame’ler
* Tek çilek / çoklu çilek içeren sahneler

### 2️⃣ Ortam Koşulları

* Açık alan (tarla)
* Sera ortamı
* Hidroponik sistem
* Doğal ve yapay ışık
* Sabah / öğle / akşam çekimleri

---

## 🏷️ ETİKETLEME STRATEJİSİ (KRİTİK)

### 🎯 Etiketleme Türü

* **Bounding Box (Object Detection)**
* YOLO formatı veya eşdeğeri

### 🧾 Sınıf Tanımları (ZORUNLU)

Aşağıdaki sınıflar **KESİNLİKLE** kullanılmalıdır:

1. `strawberry_ripe`
2. `strawberry_semi_ripe`
3. `strawberry_unripe`

> NOT: Sınıf isimleri **tutarlı**, **küçük harf**, **snake_case** formatında olmalıdır.

---

## 📐 Etiketleme Kuralları

Etiketleme sırasında şu kurallar **zorunludur**:

* Bounding box yalnızca **çilek meyvesini** kapsamalı
* Yaprak, sap veya çiçek **kutuya dahil edilmemeli**
* Kısmen görünen çilekler **etiketlenmeli**
* Üst üste binen çilekler **ayrı ayrı etiketlenmeli**
* Çok küçük (ayırt edilemeyen) çilekler **etiketlenmemeli**

---

## 📏 Olgunluk Tanım Kriterleri

Etiketleyiciler için **net tanımlar** oluşturulmalıdır:

### 🍓 Olgun (`strawberry_ripe`)

* Kırmızı renk baskın
* Yeşil alan <%10
* Hasada hazır

### 🍓 Yarı Olgun (`strawberry_semi_ripe`)

* Kırmızı + beyaz karışımı
* Renk geçişleri belirgin

### 🍓 Olgun Değil (`strawberry_unripe`)

* Yeşil veya açık beyaz
* Kırmızı renk yok veya çok az

---

## 🔁 Veri Dağılımı

Dataset şu oranları hedeflemelidir:

* %60 Eğitim (Train)
* %20 Doğrulama (Validation)
* %20 Test

Her sınıf bu bölümlerde **dengeli** temsil edilmelidir.

---

## 🔄 Veri Çeşitliliği ve Denge

Aşağıdaki durumlar özellikle dahil edilmelidir:

* Farklı boyutlarda çilekler
* Kameraya yakın / uzak çilekler
* Kısmi örtülmüş (occluded) çilekler
* Aynı karede farklı olgunluk seviyeleri

---

## 🚫 Kaçınılması Gereken Hatalar

* Sınıf dengesizliği
* Aşırı benzer görüntüler
* Yanlış olgunluk etiketleri
* Aynı sahnenin tekrar tekrar eklenmesi
* Aşırı blur veya tanınamaz görüntüler

---

## 🧪 Kalite Kontrol (QA)

Dataset aşağıdaki kontrollerden geçirilmelidir:

* Rastgele örnek denetimi
* Sınıf dağılım analizi
* Bounding box doğruluk kontrolü
* Yanlış etiket oranı <%5

---

## 📦 Teslim Edilebilirler

Bu sürecin sonunda:

* Eğitim-ready bir dataset
* Net sınıf tanımları
* Etiketleme rehberi
* Genişletilebilir veri yapısı

elde edilmelidir.

---

## 🔮 Geleceğe Dönük Uyumluluk

Dataset, ileride şu genişletmelere açık olmalıdır:

* Hastalık tespiti
* Boyut / kalite sınıflandırması
* Zaman bazlı büyüme analizi
* TADS entegrasyonu
* Hasat tahmini

---

## ✅ Başarı Kriteri

Bu prompt başarıyla uygulandığında:

* YOLO tabanlı modeller yüksek doğrulukla eğitilebilmeli
* Gerçek tarım sahasında kararlı sonuçlar alınabilmeli
* Dataset uzun vadeli projelerde yeniden kullanılabilmelidir

---

İstersen bir sonraki adımda:

* **YOLO eğitim & hiperparametre promptu**
* **Roboflow etiketleme talimat promptu**
* **Dataset kalite skorlama checklist’i**



