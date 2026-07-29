# Hiyerarşik Model Mimarisine Geçiş — İş Planı

Tek 10 sınıflı modelden, her problem alanı için bağımsız modele geçiş.
Her madde **tek başına verilebilecek bir iş tanımıdır**; sırayla veya paralel
yürütülebilir. Bağımlılıklar açıkça yazılmıştır.

## Neden

Somut sorun: olgunluk sınıfları (yeşil meyve) **yaprakları** "olgunlaşmamış
çilek" sanıyordu. Ölçüm: gerçek meyve güveni medyan 0.58, yaprak yanlış
pozitifleri 0.69-0.79 — **üst üste biniyorlardı**, ayıran eşik yoktu.

Hiyerarşik yapıda olgunluk modeli yalnızca **meyve kırpıntısı** görür. Yaprağı
olgunlaşmamış meyve diye işaretlemesi yapısal olarak imkânsız hale gelir:
sorun eşikle bastırılmaz, ortadan kalkar.

---

## Durum tablosu

| # | İş | Bağımlılık | Durum |
|---|-----|------------|-------|
| 1 | Model kütüğü (`configs/modeller.yaml`) | — | ✅ yapıldı |
| 2 | Model deposu (`app/modeller.py`) | 1 | ✅ yapıldı |
| 3 | Boru hattı (`app/pipeline.py`) | 2 | ✅ yapıldı |
| 4 | Çekirdeğe bağlama (foto/video/kamera/canlı) | 3 | ✅ yapıldı |
| 5 | Mevcut veriden 3 dataset türetme | — | ✅ betik hazır |
| 6 | Organ dataset'i | — | ✅ **veri bulundu** (16.358 görüntü) |
| 7 | Zararlı dataset'i (yeni etiketleme) | — | ⬜ veri gerekli |
| 8 | Model eğitimleri | 5,6,7 | ⬜ |
| 9 | Arayüz: model durumu ve boru hattı izi | 3 | ⬜ |
| 10 | Testler + belgeler | 3,4 | ✅ 19 test |

---

## 1. Model kütüğü ✅

`configs/modeller.yaml`: her modelin dosyası, rolü, hangi organda tetikleneceği,
sınıfları ve eşiği. Model eksikse boru hattı çökmez, mirasa düşer.

## 2. Model deposu ✅

`app/modeller.py`: kütüğü okur, modelleri **gerektiğinde** yükler (görüntüde
meyve yoksa meyve modeli hiç yüklenmez), durum raporu verir.

## 3. Boru hattı ✅

`app/pipeline.py`:

```
görüntü → organ modeli → ROI kırpma (pay ile)
        → tetiklenen uzman modeller → koordinat geri dönüşümü
        → eşik süzme → birleştirme → tek sonuç
```

**Kritik ayrıntılar:**
- ROI kırpılırken **pay (padding)** bırakılmalı: lezyon organ kutusunun kenarına
  taşabilir.
- Uzman modelin kutuları ROI koordinatındadır; **orijinal görüntüye geri
  dönüştürülmelidir**. Bu dönüşüm testle sabitlenmelidir — sessiz hata kaynağı.
- Aynı lezyon birden çok ROI'de görülebilir → birleştirmede NMS.
- Organ modeli yoksa: uzman modeller tüm görüntüde çalışır (ROI'siz).
- Hiçbiri yoksa: miras model (mevcut davranış).

## 4. Çekirdeğe bağlama

`detector.goruntu()` çağrılan 5 nokta (foto, ayrıntılı, video, IP kamera, canlı)
boru hattına yönlendirilir. `Sonuc`/`Kutu` yapısı **korunur** — veritabanı,
etiketleme, dışa aktarım ve harita dokunulmadan çalışmaya devam eder.

Kutuya `kaynak_model` alanı eklenir: tespiti hangi modelin ürettiği kayıtlarda
görünsün (hata ayıklama ve güven için).

## 5. Mevcut veriden 3 dataset türetme ✅

`scripts/dataset_ayir.py` — kuru çalıştırma sonucu (mevcut veriyle):

| Dataset | İçerikli görüntü | train / valid / test |
|---------|-----------------|----------------------|
| `leaf_disease` | 6.749 | 5.641 / 1.169 / 426 |
| `fruit_disease` | 4.825 | 4.662 / 658 / 228 |
| `fruit_ripeness` | 3.732 | 3.723 / 468 / 100 |

```bash
python scripts/dataset_ayir.py --kuru                # rapor, hiçbir şey yazmaz
python scripts/dataset_ayir.py --paketle             # ayır + her birini zip'le
```

Üretilen paketler (Colab'e **yalnızca eğitilecek olan** yüklenir):

| Paket | Boyut | Neden ayrı |
|-------|-------|-----------|
| `leaf_disease.zip` | 211 MB | Yaprak modelini eğitirken meyve verisi hiç taşınmaz |
| `fruit_disease.zip` | 182 MB | |
| `fruit_ripeness.zip` | 243 MB | |
| (birleşik) `dataset_colab.zip` | 772 MB | Eski tek model için |

**Colab'de kullanım:** dataset hücresindeki tek değişken belirler:

```python
EGITILECEK = 'leaf_disease'    # birlesik | leaf_disease | fruit_disease |
                               # fruit_ripeness | organ_detection | pest_detection
```

Bu değişken hem hangi arşivin açılacağını hem hangi `data.yaml`'ın kullanılacağını
hem de **koşu adını** belirler — farklı modeller birbirinin klasörünü ezmez ve
"yarım kalan eğitim" mantığı doğru koşuyu bulur.

Mevcut 10 sınıflı birleşik veriden **sınıf bazında ayırarak** üç dataset üretir:

| Yeni dataset | Alınacak sınıflar |
|--------------|-------------------|
| `leaf_disease` | Angular Leafspot, Leaf Spot, Powdery Mildew Leaf, Gray Mold* |
| `fruit_disease` | Anthracnose Fruit Rot, Powdery Mildew Fruit, Blossom Blight, Gray Mold* |
| `fruit_ripeness` | strawberry_unripe, strawberry_semi_ripe, strawberry_ripe |

\* Gray Mold hem yaprakta hem meyvede görülür; hangi dataset'e gideceği
görüntüdeki diğer sınıflara bakılarak veya elle ayrılmalıdır.

**Dikkat:** İlgisiz sınıfların kutuları atılır, görüntü **background** olarak
kalır (silinmez) — "burada bu sınıf yok" bilgisi değerlidir.

**Sınırı bilin:** Bu türetme, uzman modelleri **tam görüntüyle** eğitir; oysa
çalışma anında **ROI kırpıntısı** görecekler. Aradaki fark doğruluğu düşürür.
Kalıcı çözüm 6. maddedeki organ etiketlemesinden sonra ROI kırpıntılarıyla
yeniden üretmektir (`--roi` kipi).

## 6. Organ dataset'i ✅

**Veri hazır** — `dataset/zip dosyalar/strawberry.v2i.yolo26.zip` içinde çıktı.
Etiketleme gerekmedi.

```
datasets/organ_detection/
  train  14.313 görüntü    Flower 780 · Fruit 10.662 · Leaf 11.833
  valid   1.363 görüntü    Flower  71 · Fruit    991 · Leaf  1.072
  test      682 görüntü    Flower  39 · Fruit    487 · Leaf    561
  ────────────────────────────────────────────────────────────────
  16.358 görüntü · 26.496 kutu · eşleşmeyen dosya: 0
```

| Konu | Durum |
|------|-------|
| Sınıflar | `Flower`, `Fruit`, `Leaf` |
| Eksik organlar | `stem`, `runner` yok — bu veride etiketlenmemiş |
| Sınıf dengesi | ⚠️ Flower 890 kutu vs Leaf 13.466 → **15 kat dengesizlik** |

**Flower dengesizliği önemli mi?** Şu an hayır: hiçbir uzman model çiçekte
tetiklenmiyor (`tetik` listelerinde `flower` yok). Blossom Blight (çiçek
yanıklığı) için ileride çiçek ROI'si gerekirse önce bu sınıf güçlendirilmelidir.

**Sınıf adları büyük harfle** (`Leaf`, `Fruit`) — model bu adlarla çıktı verir.
Tetik eşleşmesi büyük/küçük harf duyarsızdır (`app/modeller.py: tetiklenen`),
doğrulandı.

> Roboflow'un `data.yaml` dosyası `../train/images` yazar; bu yol
> `datasets/train/images`'a çözülür ve "images not found" verir. Diğer türetilen
> dataset'lerle aynı biçime çevrildi: mutlak `path` + göreli alt yollar.

## 7. Zararlı dataset'i — YENİ ETİKETLEME GEREKLİ

Sıfırdan. Sınıflar kütükte planlanan olarak hazır (Thrips, Spider Mites,
Aphids, Whiteflies, Slug, Weevil).

- Zararlılar küçüktür → yakın çekim ve yüksek çözünürlük şart
- Hedef: sınıf başına **100-200 örnek**
- Sahadan toplama: canlı tespitteki 🗃️ "her kare" kipi hızlandırır

## 8. Model eğitimleri ve kurulumu

Her dataset bağımsız eğitilir; **sıra bağımlılığı yoktur**, paralel yürütülebilir.

**Colab'de:** dataset hücresinde tek değişken:

```python
EGITILECEK = 'organ_detection'   # sonra leaf_disease, fruit_disease, fruit_ripeness
```

**Yerelde:**

```bash
python scripts/train_yolo.py --data datasets/cilek/organ_detection/data.yaml        --model yolo26s.pt --name organ
```

### Çıktılar nereye yazılır? (Colab oturumu kapansa da kalır)

Eğitim boyunca **her şey Drive'a** yazılır — Colab'in geçici diskine değil:

| Ne | Nerede |
|----|--------|
| Koşu dizini (ağırlıklar, `results.csv`, grafikler) | `MyDrive/SmartFarmStrawberryDisease/results/<koşu>/` |
| Ara checkpoint (her 10 epoch) | `.../results/<koşu>/weights/epoch*.pt` |
| Eğitim sonunda kopya | `.../best_models/best_<koşu>.pt` |
| **Boru hattı adıyla kopya** | `.../best_models/<urun>/organ.pt` gibi |

Son satır işi kolaylaştırır: eğitim çıktısı hep `best.pt` adındadır, oysa boru
hattı kütükteki adı arar (`organ.pt`). Notebook ikisini de yazar; Drive'dan
indirip doğrudan `models/<urun>/` altına koyabilirsiniz.

`TRAIN_CONFIG['project']` **tüm modlar için** Drive'daki `results/` klasörüdür
(sıfırdan, devam, ince ayar ve `EGITILECEK` ile uzman model eğitimleri dahil).
`tests/test_notebook_kosu.py` bunu sabitler — bir kez ince ayar bu ayarı
devralmadığı için sonuçlar geçici diske yazılmış ve oturum koptuğunda
kaybolmuştu.

### Eğitim bitince: kurulum

Eğitim `runs/.../weights/best.pt` üretir; boru hattı ise modeli kütükteki adla
arar. Elle kopyalamayın — üç sessiz hata olur:

| Hata | Sonuç |
|------|-------|
| Yanlış ada kopyalama | Model **hiç kullanılmaz**, kimse fark etmez |
| Yanlış modeli kopyalama | Yaprak modeli olgunluk yerine geçer |
| Sınıfları uymayan model | Boru hattı çalışır ama **sonuçlar saçmadır** |

```bash
python scripts/model_kur.py --listele                      # hangi model nereye
python scripts/model_kur.py organ runs/train/organ/weights/best.pt
```

Betik kopyalamadan **önce** modelin sınıflarını kütükle karşılaştırır; uyuşmazsa
kurulumu yapmaz ve farkı yazar. Önceki model `.pt.onceki` olarak yedeklenir.
Kurulum sonrası hangi modellerin eksik kaldığını ve boru hattının aktif olup
olmadığını bildirir.

> **Sınıf sırası kütükle dataset arasında birebir aynı olmalıdır.** Eğitim
> ID'lerini dataset belirler; kütük ona uyar. Sıra kayarsa `Gray Mold` tespiti
> `Powdery Mildew Leaf` olarak görünür — `tests/test_model_kur.py` bunu sabitler.

### Model hazır olunca ne değişir?

Kod değişikliği **gerekmez**. `models/<urun>/` altına doğru adla konan model,
uygulama yeniden başlatıldığında boru hattına kendiliğinden girer. Analiz
yollarının **hepsi** (fotoğraf, ayrıntılı analiz, video, IP kamera, canlı akış)
aynı boru hattını kullanır — biri hiyerarşik, diğeri tek model diye ayrışmaz.

## 9. Arayüz

- **Model durumu sayfası**: hangi model hazır, hangisi eksik, hangisi bellekte
- Kayıt sayfasında tespitin **hangi modelden** geldiği
- Boru hattı izi: "1 yaprak → yaprak hastalığı modeli → 2 lezyon"

## 10. Testler

- ROI koordinat dönüşümü (kırp → tespit → geri dönüştür = orijinal konum)
- Organ modeli yokken mirasa düşme
- Uzman model eksikken diğerlerinin çalışmaya devam etmesi
- Olgunluk modelinin yaprak ROI'sinde **hiç çalıştırılmaması** (asıl kazanım)

---

## Geçiş stratejisi

Tek seferde değil, **kademeli**. Her model eğitildikçe devreye girer:

| Aşama | Eldeki modeller | Davranış |
|-------|-----------------|----------|
| Bugün | miras | Mevcut davranış, değişiklik yok |
| +organ | organ + miras | Organ kutuları görünür, tespit hâlâ mirastan |
| +olgunluk | organ + olgunluk + miras | **Yaprak/meyve karışması biter** |
| +hastalık | organ + hepsi | Uzman hastalık modelleri devrede |
| Tam | hepsi + zararlı | `miras: aktif: false` yapılabilir |

En büyük kazanç **üçüncü aşamada** gelir: organ + olgunluk modelleri hazır
olduğunda bu belgenin başındaki sorun tamamen çözülür.
