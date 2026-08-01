# Veri Alma — Roboflow, Kaggle ve Diğer Kaynaklar

> Dışarıdan indirilen hiçbir paket **doğrudan** kullanılmaz. Hepsi bir
> dönüşümden geçer. Sebebi: indirilen paketlerde tekrar tekrar aynı üç
> sorun çıktı ve ikisi **sessizdi** — eğitim sorunsuz görünüp sahada
> çöküyordu.

---

## 1. Projenin dataset standardı

Bir dataset bu yapıda olmak zorundadır:

```
datasets/<urun>/<model>/
├── data.yaml
├── train/images/   train/labels/
├── valid/images/   valid/labels/
└── test/images/    test/labels/
```

`data.yaml`:

```yaml
# 'path' anahtarı BİLEREK YAZILMAZ
train: train/images
val: valid/images
test: test/images
nc: 3
names:
  0: Flower
  1: Fruit
  2: Leaf
```

### Neden `path` yazılmaz?

Ultralytics kökü şöyle çözer (`data/utils.py`):

```python
path = data.get('path') or Path(data['yaml_file']).parent
```

`path` yoksa **yaml'ın kendi klasörü** kök olur. Mutlak yol yazılsaydı
klasör taşındığında veya Colab'de açıldığında `images not found` verirdi.
Bu bir kez yaşandı: `datasets/` → `datasets/cilek/` taşınması bütün
paketleri bozdu.

---

## 2. Dışarıdan gelen pakette çıkan üç sorun

### 2.1 Veri sızıntısı — EN TEHLİKELİSİ, sessizdir

Roboflow artırımı tek kaynak fotoğraftan 4-5 kopya üretir. Bölme
artırımdan **sonra** yapıldıysa aynı fotoğrafın kopyaları hem `train` hem
`valid` içinde olur. Model doğrulama setini zaten görmüştür: **mAP yüksek
çıkar, sahada çöker.**

Gerçek ölçüm (`pest_detection.zip`):

```
train benzersiz kaynak : 778
valid benzersiz kaynak : 534
ORTAK (sızıntı)        : 532   →  valid'in %99,6'sı
```

İki görüntü açılıp karşılaştırıldı: **aynı fotoğraf, biri 90° döndürülmüş**.

Dosya adı deseni sızıntıyı ele verir:

```
605_0_Black_Cutworm--1-_jpg.rf.7281585f….jpg   → train
605_1_Black_Cutworm--1-_jpg.rf.f5472521….jpg   → valid   ⛔ aynı kaynak
605_2_Black_Cutworm--1-_jpg.rf.2d51220a….jpg   → valid
```

`.rf.<karma>` artırım kimliğidir; `<n>_<m>_` öneki de artırım sırasıdır.
**Gerçek kaynak** ortadaki isimdir (`Black_Cutworm--1-`).

### 2.2 Yol biçimi

Roboflow `train: ../train/images` yazar — bir üst dizine çıkar. Bizim
standardımız kendi klasörünü gösterir.

### 2.3 Olmayan bölüm bildirimi

`test: ../test/images` yazılıp test klasörü konmamış olabilir; Ultralytics
test istediğinde hata verir.

---

## 3. Doğru akış: `harici_paket_duzelt.py`

```bash
# 1) Önce SADECE RAPOR — hiçbir şey yazmaz
python scripts/harici_paket_duzelt.py datasets/cilek/pest_detection.zip \
    --ad bocek_teshis \
    --siniflar "Army Worm,Black Cutworm,Grub,Mole Cricket,Peach Borer,Spider Mites" \
    --kuru

# 2) Rapor doğruysa uygula ve paketle
python scripts/harici_paket_duzelt.py datasets/cilek/pest_detection.zip \
    --ad bocek_teshis --siniflar "…" --paketle
```

Betiğin yaptıkları:

| İş | Nasıl |
|---|---|
| Sızıntı ölçümü | Kopyaları **kaynak fotoğrafa** göre gruplar, bölmeler arası ortak kaynağı sayar |
| Yeniden bölme | Grup düzeyinde böler — bir kaynağın bütün kopyaları aynı bölmede kalır |
| Sınıf dengesi | Grupları baskın sınıfa göre dağıtır (rastgele bölme küçük sınıfları yığar) |
| Yol düzeltme | `../train/images` → `train/images`, `path` silinir |
| Bölüm düzeltme | Olmayan bölüm `data.yaml`'a yazılmaz |
| Sınıf adı | `--siniflar` ile **sıra korunarak** değiştirilebilir; ID'ler değişmez |

Örnek çıktı:

```
--- SIZINTI KONTROLÜ (kaynak fotoğraf düzeyinde) ---
  ⛔ train ↔ valid: 532 ortak kaynak (küçük bölmenin %99.6'i)

--- YENİDEN BÖLME (780 kaynak grubu) ---
  sınıf                    train  valid   test
  Army Worm                  818    160    110
  …
  sızıntı sonrası: ✅ temiz
```

### Sınıf adı değiştirirken

`--siniflar` **SIRAYA göre** eşleşir. Sıra yanlış verilirse etiketler
sessizce yanlış sınıfa kayar — betik eski ve yeni adı yan yana yazar,
kontrol edin:

```
0: army worm              → Army Worm  ←
5: red spider             → Spider Mites  ←
```

> **Aynı canlı, aynı sınıf adı.** `Red Spider Mite` yerine `Spider Mites`
> kullanıldı çünkü saha zararlı modeli de o adı kullanıyor. Farklı ad
> verilseydi arayüzde iki ayrı "Kırmızı Örümcek" görünür, tedavi metni
> ikiye bölünürdü. Test bunu koruyor.

> **Ama bu kural ÜRÜN İÇİNDEDİR.** Farklı bitkilerde aynı adı taşıyan
> hastalıklar birleştirilmez — `Leaf Spot` çilekte ve fındıkta ayrı
> etkenlerdir, ayrı kütüklerde kalır.
> Bkz. [COK_BITKILI_YAPI.md](COK_BITKILI_YAPI.md).

### "Healthy" sınıfları — ALINMAZ

Roboflow paketlerinin çoğunda `Healthy`, `Healthy Leaf`, `Fresh` gibi
sınıflar bulunur. **Bu projede bunlar uzman dataset'e girmez.**

Sebep mimaridir: *sağlıklı* durumu organ modelinden türetilir —
organ modeli yaprağı bulur, uzman model orada bulgu çıkarmazsa yaprak
sağlıklıdır. Arayüz bunu *"🌿 Yapraklarda (5 adet) — Bakıldı, bulgu yok"*
diye yazar. Ayrıntı: [MIMARI.md](MIMARI.md) § "Üçüncü fayda".

Sınıf olarak alınırsa iki somut arıza olur:

1. Hastalıklı bir yaprak hem `Leaf Spot` hem `Healthy` kutusuna girer;
   iki kutu NMS'te birbirini bastırır, sonuç kararsız olur.
2. Arayüzde "Healthy Leaf" bir **bulgu** gibi listelenir, tedavi kütüğünde
   karşılığı olmadığı için boş kart çıkar.

Ne yapılır:

```bash
# Healthy görüntüleri ATMA — etiketini sil, background olarak bırak
python scripts/harici_paket_duzelt.py datasets/findik/ham/leaf \
    --arka-plana-al "Healthy,Healthy Leaf,Fresh Leaf" \
    --cikti datasets/findik/leaf_disease
```

Etiketsiz kalan görüntü YOLO için negatif örnektir: modele *"burada hastalık
yok"* diye öğretir. Görüntü silinirse bu bilgi de kaybolur — hatalı pozitif
oranı artar.

**İstisna:** Organ dataset'inde `leaf` / `nut` / `flower` sınıfları elbette
kalır; organ modelinin işi zaten onları bulmaktır.

---

## 4. Alınan paket projeye nasıl bağlanır?

### Adım 1 — Kütüğe kaydet

`configs/urunler/<urun>/modeller.yaml`:

```yaml
bocek_teshis:
  dosya: bocek_teshis.pt
  rol: tekil            # organ | yaprak_hastalik | … | tekil | miras
  tetik: []             # hangi organ bulununca çalışsın; boş = ROI'ye girmez
  imgsz: 416            # bu modelin ÇIKARIM çözünürlüğü (ölçün!)
  siniflar: [Army Worm, Black Cutworm, Grub, Mole Cricket, Peach Borer, Spider Mites]
  esik: 0.35
  aktif: true
```

**`siniflar` sırası dataset ile birebir aynı olmalı.** Kaynak dataset'tir;
`model_kur.py` uymazsa kurulumu reddeder.

### Adım 2 — Türkçe adlar

`configs/urunler/<urun>/siniflar.yaml`:

```yaml
Army Worm:
  tr: Bozkurt Tırtılı
  en: Army Worm
  grup: zararli       # hastalik | zararli | olgunluk | organ | diger
```

> **`id:` vermeyin** — ID alanı birleşik modelin etiket dosyalarındaki
> sayıdır. Uzman model kendi dataset'inde 0..n-1 kullanır; ID verilirse
> etiketleme ekranındaki numaralarla çakışır ve geçmiş etiketler kayar.

### Adım 3 — Tedavi önerisi

`configs/urunler/<urun>/tedavi_onerileri.yaml`. Sınıf hem yaprakta hem
meyvede çıkabiliyorsa `organ:` bloğu ekleyin:

```yaml
Gray Mold:
  ad: Kurşuni Küf (Botrytis)
  aciliyet: yuksek
  belirti: Gri, tozlu küf tabakası.
  onlem: [...]
  organ:
    fruit:
      belirti: Meyvede gri küf; hızlı çürüme.
      onlem: [Enfekte meyveleri HEMEN toplayın, ...]
    leaf:
      aciliyet: orta
      belirti: Yaşlı yapraklarda kenardan kahverengi kuruma.
      onlem: [Yaşlanmış yaprakları temizleyin, ...]
```

Önlem listesi **birleştirilmez, değiştirilir**. Kısmi birleştirme yaprak
bulgusuna "Meyveyi hemen toplayın" maddesini eklerdi.

### Adım 4 — imgsz'i ölç

```bash
python scripts/imgsz_oner.py datasets/cilek/bocek_teshis
```

Ölçüt "görüntü kaç piksel" **değil**, "nesneler bu imgsz'de kaç piksel
kalıyor". YOLO'nun en ince katmanı 8 piksel adımlıdır; nesne en az ~16
piksel kaplamalı.

### Adım 5 — Eğit ve kur

Bkz. [EGITIM.md](EGITIM.md).

---

## 5. Var olan dataset'e veri eklemek

İki durum var:

### 5.1 Aynı sınıflar, yeni görüntüler

```bash
python scripts/merge_datasets.py --in-place --kuru   # önce rapor
```

Sınıf **çakışması olmadan** birleştirir; `class_aliases.yaml` farklı
kaynakların aynı sınıfa verdiği farklı adları eşler
(örn. `angular_leafspot` ↔ `Angular Leafspot`).

### 5.2 Yeni sınıf

```bash
python scripts/sinif_ekle.py "Spider Mites" --tr "Kırmızı Örümcek" --grup zararli
```

> ⚠️ Kütüğe eklemek modele **öğretmez**. Sıra:
> 1. Betik → sınıf etiketleme ekranında görünür
> 2. 100-200 örnek toplanıp etiketlenir
> 3. Dışa aktarım → `merge_datasets.py` → yeniden eğitim
> 4. Yeni model sınıfı tanır → `egitimde: true`

**ID kuralı:** verilen ID bir daha değiştirilmez. Etiket dosyalarında sayı
olarak saklandığı için ID kayarsa geçmişte etiketlenen her şey yanlış
sınıfa döner.

---

## 6. Birleşik dataset'i uzman modellere ayırmak

```bash
python scripts/dataset_ayir.py --paketle
```

`dataset/` içindeki birleşik veriyi `datasets/<urun>/<model>/` altına ayırır
ve Colab'e yüklenecek zip'leri üretir. Eşleme betiğin içindeki `AYRIM`
sözlüğündedir.

---

## 7. Kontrol listesi — yeni paket alırken

- [ ] `harici_paket_duzelt.py --kuru` çalıştırıldı, **sızıntı raporu temiz**
- [ ] Sınıf adları mevcut kütükle çakışmıyor (aynı canlı → aynı ad)
- [ ] `etiket_temizle.py` ile taşan/bozuk kutu kontrol edildi
- [ ] `imgsz_oner.py` ile çözünürlük ölçüldü; "sorun veride" uyarısı yoksa
- [ ] `modeller.yaml` sınıf sırası dataset ile birebir
- [ ] `siniflar.yaml`'a Türkçe ad eklendi, **`id` verilmedi**
- [ ] Tedavi önerisi yazıldı (iki organda çıkıyorsa `organ:` bloğuyla)
- [ ] Zip Drive'a `datasets/<urun>/` altına yüklendi

---

## İlgili belgeler

- [MIMARI.md](MIMARI.md) — model kütüğü ve tetik mantığı
- [HATA-YONETIMI.md](HATA-YONETIMI.md) § 2 — veri seti hataları
- [EGITIM.md](EGITIM.md) — eğitim ve kurulum
