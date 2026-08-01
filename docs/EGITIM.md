# Eğitim ve Model Kurulumu

> Eğitim Google Colab'de yapılır (A100 + High-RAM), sonuçlar Drive'a yazılır.
> Bu belge **ölçerek karar verme** ilkesine dayanır: epoch, imgsz ve
> önbellek tahminle değil ölçümle seçilir.

---

## 1. Hızlı akış

```
1. Yerelde paketi hazırla      → scripts/dataset_ayir.py --paketle
2. Zip'i Drive'a yükle         → MyDrive/SmartFarmStrawberryDisease/datasets/cilek/
3. Colab notebook'u aç         → 4️⃣ hücresi: EGITILECEK = 'organ_detection'
4. Runtime → Run all
5. best.pt indir → model_kur.py ile kur
6. docker compose restart
```

Notebook: `StrawberryVision_Colab_Production.ipynb`

> **Colab sekmesi bayat kalabilir.** GitHub'dan açılan notebook Google
> tarafında önbelleğe alınır; `Ctrl+Shift+R` yetmez. Commit SHA'lı link
> kullanın. Notebook 3️⃣ hücresi sürüm farkını fark edip uyarır.

---

## 2. Hangi modeli eğiteceksiniz?

4️⃣ hücresindeki tek satır belirler:

```python
EGITILECEK = 'organ_detection'
URUN       = 'cilek'
```

| Değer | train / valid / test | Sınıf |
|---|---|---|
| `organ_detection` ⭐ | 14.313 / 1.363 / 682 | 3 |
| `leaf_disease` | 5.641 / 1.169 / 426 | 4 |
| `fruit_disease` | 4.662 / 658 / 228 | 4 |
| `fruit_ripeness` | 3.723 / 468 / 100 | 3 |
| `bocek_teshis` | 3.109 / 565 / 225 | 6 |
| `pest_detection` | — | *veri toplanacak* |
| `birlesik` | 10.780 / 1.793 / 570 | 10 (eski tek model) |

⭐ **Önce `organ_detection`** — o olmadan hiyerarşi çalışmaz, sistem tek
modele düşer. Diğerlerinin sırası önemsizdir.

### Modeller arası izolasyon

Bütün koşular Drive'da tek `results/` klasörünü paylaşır. Koşu adı
`EGITILECEK`'ten türetilir; farklı modeller birbirinin klasörünü göremez.
Bu bir kez kırıldığında `organ_detection` eğitimi atlandı ve birleşik
modelin ağırlığı `organ.pt` adıyla kopyalandı.

---

## 3. Ölçerek belirlenen ayarlar

### 3.1 `imgsz` — çözünürlük

```bash
python scripts/imgsz_oner.py --hepsi
```

Ölçüt **"görüntü kaç piksel" değil**, "nesneler bu imgsz'de kaç piksel
kalıyor". YOLO'nun en ince tespit katmanı 8 piksel adımlıdır; nesne o
ızgarada en az ~2 hücre (≈16 piksel) kaplamalı.

Ölçüm sonucu (bu projedeki dataset'ler):

| Dataset | Kaynak | Öneri |
|---|---|---|
| `bocek_teshis` | 416 px | 320 |
| `fruit_disease` | 280 px | 320 |
| `leaf_disease` | 280 px | 320 |
| `fruit_ripeness` | 640 px | 416 |
| `organ_detection` | 640 px | 320 (güvenli: 640) |

**Hiçbiri 1024 değil** — hepsi büyütülerek eğitiliyordu.

> **Büyütme "işe yaramaz" demek değildir.** Yeni bilgi eklemez, ama YOLO'nun
> ızgarası eğitim pikselinde sabittir: kaynakta 3 px olan lezyon 320'de
> yarım hücre, 1024'te ~1,5 hücre kaplar. Küçük nesneli setlerde yüksek
> imgsz bu yüzden yaygındır. Betik iki öneri verir:
> **hızlı** (yeterli olan en küçük) ve **güvenli** (kaynak çözünürlük).
> Kararsızsanız ikisini eğitip `model_karsilastir.py` ile ölçün.

### 3.2 Epoch

```bash
python scripts/epoch_oner.py    # notebook otomatik çalıştırır
```

Geçmiş koşuların eğrisinden doyma noktasını bulur. Öneri **aynı modelin**
koşularından hesaplanır; yoksa diğerlerini kullanır ama açıkça uyarır
(doyma noktası modele ve dataset'e özgüdür).

> ⚠️ Epoch'u "üst sınır" diye şişirmeyin. Öğrenme oranı takvimi ve
> `close_mosaic` **toplam** epoch'a göre hesaplanır; 200 verip 70'te
> durursanız model takvimin ortasında, yüksek öğrenme oranında kalır.

### 3.3 RAM önbelleği

Görüntüler **açılmış** halde tutulur: `görüntü × imgsz² × 3 bayt`.
Notebook seçilen modelin dataset'ini sayar ve RAM'in %40'ını aşacaksa
önbelleği kapatır.

| Dataset | @1024 | @320 |
|---|---|---|
| organ_detection (16.358) | 51,5 GB | 5,0 GB |
| leaf_disease (7.236) | 22,8 GB | 2,2 GB |

Önbellek sınırı aşılırsa eğitim **hata vermeden** epoch 1-2'de ölür.

---

## 4. Eğitim modları

```python
MOD = 'otomatik'
```

| Mod | Ne yapar |
|---|---|
| `otomatik` | Yarım varsa devam, bitmişse atlar, yoksa baştan (**önerilen**) |
| `devam` | Yarım kalandan devam etmeye zorlar; yoksa hata verir |
| `sifirdan` | Checkpoint'leri yok sayar, yeni koşu açar |
| `ince_ayar` | Mevcut `best.pt`'den devam eder — yeni veri eklendiğinde |

`devam` ve `ince_ayar` modlarında **ayarlar checkpoint'ten okunur**;
`OVERRIDES` etkisizdir. `imgsz` değiştirmek istiyorsanız `sifirdan` şart.

### İnce ayar

`configs/finetune_config.yaml` kullanılır — `optimizer: auto` `lr0`'ı
yok sayar ve sıfırdan eğitime uygun yüksek bir oran seçer; ince ayarda bu
öğrenilmiş ağırlıkları bozar.

Eğitim **başlamadan önce** sınıf uyumu kontrol edilir: ağırlığın sınıfları
dataset ile uyuşmuyorsa durur. Uyuşmazlıkta Ultralytics hata vermez, tespit
başını sessizce yeniden kurar — sonuç saatler sonra fark edilir.

---

## 5. Çıktılar Drive'a yazılır

```
MyDrive/SmartFarmStrawberryDisease/
├── results/<model>/weights/best.pt      koşu dizini, checkpoint'ler
└── best_models/cilek/<boru_hatti>.pt    boru hattı adıyla kopya
```

Colab oturumu kapandığında yerel disk silinir; `project` Drive'a bakmazsa
eğitim tamamen kaybolur.

### Eğitimi izleme

```bash
python scripts/egitim_izle.py            # özet
python scripts/egitim_izle.py --bekle    # ölünce/bitince haber ver
```

Drive'a bakarak çalışır, Colab'e girmek gerekmez. Kesintileri
`results.csv`'deki `time` kolonunun geri gitmesinden bulur.

> Colab oturumu düzensiz aralıklarla ölür. Pahalı olan ölmesi değil
> **fark edilmemesi**: bir koşu 8,5 saat boşta bekledi. `--bekle` bunu
> dakikalara indirir.

---

## 6. Model kurulumu

```bash
python scripts/model_kur.py organ models/cilek/best_organ_detection.pt
docker compose restart web
```

| Eğitilen | Kütük adı |
|---|---|
| `organ_detection` | `organ` |
| `leaf_disease` | `yaprak_hastalik` |
| `fruit_disease` | `meyve_hastalik` |
| `fruit_ripeness` | `olgunluk` |
| `bocek_teshis` | `bocek_teshis` |

Betik kopyalamadan önce sınıfları kütükle karşılaştırır. **Elle
kopyalamayın** — üç sessiz hata kaynağıdır: yanlış ada kopyalama (model hiç
kullanılmaz), yanlış modeli kopyalama, sınıfları uymayan model.

Durum: `python scripts/model_kur.py --listele`

> Host'ta `torch` yoksa doğrulama atlanır. O durumda Docker imajında
> çalıştırın:
> ```bash
> docker run --rm -v "${PWD}/models:/app/models" -v "${PWD}/configs:/app/configs" \
>   -v "${PWD}/scripts:/app/scripts" -v "${PWD}/app:/app/app" -w /app \
>   cilek-tespit:latest python scripts/model_kur.py organ models/cilek/best_organ_detection.pt
> ```

---

## 7. Model karşılaştırma

```bash
python scripts/model_karsilastir.py eski.pt yeni.pt --veri datasets/cilek/leaf_disease/data.yaml
```

**Aynı test setinde** sınıf bazında karşılaştırır. Yeni model her sınıfta
iyi olmayabilir; toplam mAP artarken bir sınıf gerileyebilir.

### Bugünkü sonuçlar

| Model | epoch | mAP50-95 |
|---|---|---|
| `organ_detection` | 200/200 | **0,8326** |
| `fruit_ripeness` | 200/200 | 0,6157 |
| `bocek_teshis` | 200/200 | 0,5712 |
| `fruit_disease` | 200/200 | 0,5559 |
| `leaf_disease` | 200/200 | 0,4000 |

`leaf_disease` düşük — sebebi ölçüldü: en küçük %10 kutu **kaynak
görüntüde zaten 3 piksel**. Bu imgsz sorunu değil veri sorunudur.

---

## 8. Sınıf dengesizliği — çoğu zaman sorun değil

Nesne tespitinde 10:1, 50:1 gibi oranlar sorunludur. Bu projede ölçülen:

| Model | Oran | Sonuç |
|---|---|---|
| `fruit_disease` | 1,92 | sorun yok |
| `leaf_disease` | 1,88 | sorun yok |
| `fruit_ripeness` | 2,67 | sorun yok |
| `organ_detection` | **15,17** | **en az veriye sahip sınıf (Flower) en iyi sonucu verdi: 0,99** |

> **Görüntü çoğaltmayın.** Tespitte etiket kutu bazındadır; görüntüyü
> kopyalayınca içindeki *bütün* kutular kopyalanır. `fruit_ripeness`'te
> 1.037 görüntüde hem `unripe` hem `ripe` var — oranı düzeltmez, başka
> yerden bozar. Ayrıca kopya çeşitlilik katmaz, ezberleme yapar; doğrulama
> setine sızarsa metrikler yalan söyler.

Bir sınıf gerçekten geri kalıyorsa sırayla: **eşik ayarı** → **hedefli veri
toplama** → (gerekirse) yalnızca o sınıfı içeren görüntülere artırım.

---

## İlgili belgeler

- [VERI-ALMA.md](VERI-ALMA.md) — dataset hazırlama
- [HATA-YONETIMI.md](HATA-YONETIMI.md) § 3 — eğitim hataları
- [MIMARI.md](MIMARI.md) — model kütüğü
