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
| 6 | Organ dataset'i (yeni etiketleme) | — | ⬜ veri gerekli |
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

## 6. Organ dataset'i — YENİ ETİKETLEME GEREKLİ

Mevcut veride organ etiketi **yok**. 5 sınıf: `leaf, fruit, flower, stem, runner`.

- Hedef: **300-500 görüntü**, her sınıftan en az 100 örnek
- Kaynak: `dataset/` içindeki mevcut görüntüler + sahadan gelenler
- Araç: uygulamadaki etiketleme ekranı (sınıflar `sinif_ekle.py` ile eklenir)
- Bu model **en kritik olanıdır**: yanlış organ, yanlış uzman modele yönlendirir

## 7. Zararlı dataset'i — YENİ ETİKETLEME GEREKLİ

Sıfırdan. Sınıflar kütükte planlanan olarak hazır (Thrips, Spider Mites,
Aphids, Whiteflies, Slug, Weevil).

- Zararlılar küçüktür → yakın çekim ve yüksek çözünürlük şart
- Hedef: sınıf başına **100-200 örnek**
- Sahadan toplama: canlı tespitteki 🗃️ "her kare" kipi hızlandırır

## 8. Model eğitimleri

Her dataset bağımsız eğitilir; **sıra bağımlılığı yoktur**, paralel yürütülebilir:

```bash
python scripts/train_yolo.py --data datasets/organ_detection/data.yaml   --model yolo26s.pt --name organ
python scripts/train_yolo.py --data datasets/leaf_disease/data.yaml      --model yolo26s.pt --name leaf_disease
python scripts/train_yolo.py --data datasets/fruit_disease/data.yaml     --model yolo26s.pt --name fruit_disease
python scripts/train_yolo.py --data datasets/fruit_ripeness/data.yaml    --model yolo26s.pt --name fruit_ripeness
python scripts/train_yolo.py --data datasets/pest_detection/data.yaml    --model yolo26s.pt --name pest
```

Çıkan `best.pt` dosyaları `models/` altına kütükteki adlarla konur
(`organ.pt`, `leaf_disease.pt`, ...). **Her model hazır olduğunda kendiliğinden
devreye girer** — kod değişikliği gerekmez.

Küçük dataset'lerde `yolo26n` yeterli olabilir; organ modeli için `yolo26s`.

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
