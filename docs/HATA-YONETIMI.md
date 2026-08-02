# Hata Yönetimi ve Tekrarlayan Tuzaklar

> Bu belgenin amacı **geriye dönük iş yapmamak**. Aşağıdaki hataların her
> biri gerçekten yaşandı, çoğu birden fazla kez. Her birinin yanında
> **belirtisi**, **sebebi** ve **koruması** yazılıdır.
>
> Yeni bir hata bulunca buraya ekleyin. Koruması testse test adını yazın.

---

## Genel ilke: sessiz hata en pahalısıdır

Bu projede yaşanan hataların çoğu **çökmedi**. Sistem çalışıyor göründü,
yanlış sonuç üretti ve sebebi görünmedi. Bu yüzden üç kural:

1. **Bir şey yapılamıyorsa söyle.** Boş dönme, sessizce atlamayı bırak.
2. **Sayı veriyorsan ne anlama geldiğini yaz.** Yanlış okunan sayı,
   olmayan sayıdan kötüdür.
3. **Varsayım yerine ölç.** "Muhtemelen küçük lezyonlar var" diye imgsz
   1024 yapılmıştı; ölçünce hiçbir dataset'in 640'tan büyük olmadığı çıktı.

---

## 1. Model bulunamıyor / kurulmuyor

### 1.1 Dosya adı yanlış — model sessizce kullanılmıyor

**Belirti:** Model dosyası `models/cilek/` içinde duruyor ama sistem
"tespit yok" diyor ya da eski modeli kullanıyor.

**Sebep:** Drive'dan inen dosya `best_organ_detection.pt` adıyla gelir;
boru hattı kütükteki adı (`organ.pt`) arar.

**Koruma:** Elle kopyalama **yok**. Her zaman:

```bash
python scripts/model_kur.py organ models/cilek/best_organ_detection.pt
```

Betik kopyalamadan **önce** ağırlığın sınıflarını kütükle karşılaştırır.
Uymuyorsa kurmaz ve sebebini yazar.

| Eğitilen | İnen dosya | Kurulum komutu |
|---|---|---|
| `organ_detection` | `organ.pt` | `model_kur.py organ <yol>` |
| `leaf_disease` | `leaf_disease.pt` | `model_kur.py yaprak_hastalik <yol>` |
| `fruit_disease` | `fruit_disease.pt` | `model_kur.py meyve_hastalik <yol>` |
| `fruit_ripeness` | `fruit_ripeness.pt` | `model_kur.py olgunluk <yol>` |
| `bocek_teshis` | `bocek_teshis.pt` | `model_kur.py bocek_teshis <yol>` |

Durum kontrolü: `python scripts/model_kur.py --listele`

### 1.2 `Model bulunamadı: best.pt` — hiyerarşi kuruluyken

**Belirti:** Fotoğraf yükleme 400 döner:
`{"detail":"Model bulunamadı: /app/models/cilek/best.pt"}`

**Sebep:** Kontrol yalnızca `detector.hazir`e bakıyordu; o da **sadece eski
tek modeli** tanır. Hiyerarşiye geçip miras modeli kaldırınca dört uzman
model kurulu olmasına rağmen yükleme reddediliyordu.

**Koruma:** `main.analiz_yapilabilir()` — `hiyerarsik_hazir() or detector.hazir`.
Üç yerde kullanılır (dosya yükleme, IP kamera, ana sayfa göstergesi).
Test: `test_app.py::TestMirasModelsizCalisir`

### 1.3 Sınıf sırası kütükle uyuşmuyor

**Belirti:** Model kuruldu ama sonuçlar saçma; yanlış sınıf adları çıkıyor.

**Sebep:** Eğitim ID'lerini **dataset** belirler
(`datasets/<urun>/<model>/data.yaml`). Kütükteki `siniflar:` listesi
sıradan saparsa etiketler kayar.

**Koruma:** `model_kur.py` kurulumu reddeder. Kütüğü düzeltmek gerekiyorsa
`configs/urunler/<urun>/modeller.yaml` — ama önce **dataset'in doğru
olduğundan** emin olun; kütüğü uydurmak hatayı gizler.

---

## 2. Veri seti hataları

### 2.1 Veri sızıntısı — doğrulama skoru yalan söyler

**Belirti:** mAP çok yüksek, sahada model çöküyor.

**Sebep:** Roboflow artırımı tek kaynak fotoğraftan 4-5 kopya üretir. Bölme
artırımdan **sonra** yapıldıysa aynı fotoğrafın kopyaları hem train hem
valid'e düşer. Ölçülen gerçek olay:

> `pest_detection.zip`: valid'in **%99,6'sı** train görüntülerinin
> artırılmış kopyasıydı. İki görüntü açılıp karşılaştırıldı — aynı
> fotoğraf, biri 90° döndürülmüş.

**Koruma:** Dışarıdan gelen her paket için:

```bash
python scripts/harici_paket_duzelt.py <zip> --ad <hedef> --kuru
```

`--kuru` yalnızca rapor verir. Bölmeyi **kaynak fotoğrafa göre** yeniden
yapar; bir kaynağın bütün kopyaları aynı bölmede kalır.
Test: `test_tekil_model.py::TestPaketTemizlendi`

### 2.2 Kutular görüntü dışına taşıyor

**Belirti:** Eğitim uyarı veriyor, bazı sınıflar öğrenilmiyor.

**Sebep:** Dışa aktarım/artırım sırasında koordinatlar 0-1 aralığını aşıyor.
Bu projede **6.658 taşan kutu (%20)** ve 468 bozuk kutu bulundu.

**Koruma:** `python scripts/etiket_temizle.py` — taşanları kırpar,
geçersizleri atar.

### 2.3 Nesne kaynakta zaten çok küçük

**Belirti:** Bir model diğerlerinden belirgin düşük mAP veriyor
(`leaf_disease` 0,40 · diğerleri 0,55-0,83).

**Sebep:** `imgsz_oner.py` ölçtü: `leaf_disease`'te en küçük %10 kutu
**kaynak görüntüde zaten 3 piksel**. YOLO'nun en ince tespit katmanı 8
piksel adımlıdır; 16 pikselin altındaki nesne öğrenilemez.

**Koruma:** `python scripts/imgsz_oner.py --hepsi` — sorunun imgsz'de mi
veride mi olduğunu ayırt eder. Veride ise üç olasılık: hatalı etiket, çok
uzaktan çekim, ya da zaten küçültülmüş dışa aktarım.

### 2.4 `data.yaml` yolları taşınınca bozuluyor

**Belirti:** Colab'de `images not found`.

**Sebep:** Roboflow `train: ../train/images` yazar (bir üst dizine çıkar).
Mutlak `path:` yazılırsa klasör taşınınca bozulur.

**Koruma:** Bizim standardımız — `path` anahtarı **yazılmaz**, yollar
`train/images` biçimindedir. Ultralytics kökü yaml'ın kendi klasöründen
türetir. `harici_paket_duzelt.py` bunu otomatik düzeltir.

### 2.5 Aynı canlı iki farklı sınıf adıyla

**Belirti:** Arayüzde iki ayrı "Kırmızı Örümcek" satırı, tedavi metni ikiye
bölünmüş, geçmiş sorgusu ayrılmış.

**Sebep:** `bocek_teshis` dataset'i `Red Spider Mite`, saha zararlı modeli
`Spider Mites` diyordu; ikisinin Türkçe adı aynıydı.

**Koruma:** Sınıf adı **paylaşılır**, model ayrı kalır.
Test: `test_tekil_model.py::TestSinifAdlariTekil` — iki sınıfa aynı Türkçe
ad verilirse suite kırılır.

> **Ama bu yalnızca ÜRÜN İÇİNDE geçerlidir.** Farklı bitkilerde aynı adı
> taşıyan sınıflar birleştirilmez — `Leaf Spot` çilekte *Mycosphaerella*,
> fındıkta *Piggotia*'dır. Birleştirilseydi fındık yaprağına çilek
> tedavisi önerilirdi ve sistem hiç hata vermezdi. Bkz. § 2.7.

### 2.6 Paketin adı doğru, ALANI yanlış

**Belirti:** Model eğitimde iyi, sahada hiçbir şey bulamıyor — ya da
tamamen alakasız şeyler buluyor.

**Sebep:** Paketin adı ürüne uyuyor diye alan uyumu varsayıldı. Dataset'in
görüntü alanı (studio/makro/uydu) hedef akışın girdisinden farklıysa model
öğrendiğini sahada göremez. Ölçülen iki olay:

> `bocek_teshis`: 416×416 makro böcek fotoğrafı, kutu alanı medyanı
> karenin %15'i. ROI boru hattına bağlansaydı 60-250 px'lik yaprak
> kırpıntısında çalışacaktı — hiç benzemiyor. **Ayrı akış yapıldı.**

> `hazelnut detection v9`: adı fındık, verisi bahçe değil. Ölçüm:
> görüntü başına kutu medyanı **1**, maksimumu **1**; kutu merkezleri
> hepsi (0.50, 0.48); beyaz zeminde tek fındık. Bu bir **hasat sonrası
> ayıklama** verisidir. `tetik: [nut, husk]` ile boru hattına bağlanmak
> üzereydi; ölçüm bunu durdurdu. **`rol: tekil, tetik: []` yapıldı.**

**Koruma — paketi kütüğe bağlamadan ÖNCE ölçün:**

```bash
python scripts/imgsz_oner.py datasets/<urun>/<paket>
```

Betik artık `--- ALAN TESPİTİ ---` bölümü basar ve boru hattına uygun
olmayan paket için ⛔ verir. Ayıran iki sinyal, mevcut beş dataset
ölçülerek seçildi:

| dataset | kutuMax | çok kutulu | merkez sapması | alan |
|---|---|---|---|---|
| `findik/findik_kalite` | 1 | %0 | 0.036 | **stüdyo** |
| `cilek/bocek_teshis` | 14 | %12 | 0.089 | **makro** |
| `cilek/organ_detection` | 8 | %24 | 0.119 | saha |
| `cilek/fruit_disease` | 7 | %29 | 0.171 | saha |
| `cilek/leaf_disease` | 11 | %37 | 0.229 | saha |
| `cilek/fruit_ripeness` | 30 | %60 | 0.227 | saha |

Kural: `kutuMax == 1` **veya** (merkez < 0.15 **ve** çok kutulu < %20)
→ ayrı akış.

> **`organ_detection` satırı düzeltildi.** Önce 0.199 yazıyordu; o sayı,
> poligon etiketleri kutu sanan bir ölçümden geliyordu (§ 2.6b).
> Düzeltilince 0.119 oldu — **merkez sapması artık eşiğin altında** ve
> "saha" sınıflandırması yalnızca *çok kutulu %24 > %20* koşuluyla
> ayakta duruyor. bocek_teshis (0.089) ile arasındaki boşluk ilk
> ölçümde göründüğünden dardır; sınırdaki bir pakette **görüntülere
> bakarak** karar verin, tek başına bu tabloya güvenmeyin.

> **Kutu ALANI ayırt etmez.** İlk denemede alan eşiği (>%10 → makro)
> kullanıldı ve `organ_detection`'ı (%32.7) yanlışlıkla makro saydı.
> Böcek %18.7 iken saha organ %32.7 — sinyal yok. Eşik atıldı.
> Alan yüzdesi raporda hâlâ basılır ama "ayırt edici değil" etiketiyle.

`tetik: []` yazmak bir tercih değil, **yapısal kilittir**:
`app/modeller.py: tetiklenen()` yalnızca tetik listesinde organ adı geçen
modelleri döndürür, o yüzden bu model ROI akışında asla çalıştırılamaz.

### 2.6b Etiket satırı POLİGON, ölçüm onu kutu sanıyor

**Belirti:** Yok. Eğitim doğru çalışır, yalnızca **ölçümler** yanlış çıkar
ve o yanlış sayılara bakarak karar verilir.

**Sebep:** YOLO etiketinin iki biçimi vardır:

```
kutu    : sinif cx cy w h                 (5 alan)
poligon : sinif x1 y1 x2 y2 x3 y3 ...     (7+ alan, TEK sayıda)
```

Poligon, Ultralytics'in **segmentasyon** etiketidir. Tespit eğitiminde
Ultralytics onu kutuya çevirir (`segments2boxes`) — bu yüzden eğitim
sorunsuz çalışır. Ama satırı 5 alan varsayan bir ölçüm betiği,
**2. noktanın koordinatlarını genişlik/yükseklik sanır.**

> **Ölçülen olay:** `datasets/cilek/organ_detection` satırlarının
> **%68'i poligon**. `imgsz_oner.py` bunu bilmiyordu; o dataset için
> ürettiği bütün kutu ölçümleri çöptü ve belgelere öyle yazılmıştı.
> Düzeltmeden sonra merkez sapması 0.199 → **0.119**, kutu alanı
> %32.2 → **%26.7** oldu.

**Koruma:** `scripts/imgsz_oner.py: etiket_satiri()` her satırı çözer;
poligon gelirse köşelerin min/max'ından kutu üretir. Etiket okuyan yeni
bir betik yazarken **bu fonksiyonu kullanın**, satırı elle parçalamayın.

Paketin biçimini önceden görmek için:

```bash
python - <<'PY'
from pathlib import Path; import collections
c = collections.Counter()
for e in Path('datasets/<urun>/<ad>').rglob('labels/*.txt'):
    for s in e.read_text(errors='ignore').splitlines():
        t = s.split()
        if t: c['kutu' if len(t) == 5 else 'poligon'] += 1
print(c)
PY
```

### 2.6c Tam-kadraj kutu — "bütün fotoğraf bir organdır"

**Belirti:** Organ modeli ara sıra tüm kareyi kaplayan bir kutu üretir.
ROI kırpma o kutuyu alınca uzman modele neredeyse fotoğrafın tamamı
gider ve hiyerarşi işlevsizleşir.

**Sebep:** Kaynak dataset'te `sinif 0.5 0.5 1 1` biçiminde etiketler
vardır. Bunlar tespit değil, **görüntü düzeyi sınıflandırma etiketinin
kutuya çevrilmiş hali**dir. Model onları taklit etmeyi öğrenir.

> **Ölçülen olay:** `cilek/organ_detection` görüntülerinin **%32.7'sinde
> (5341 / 16358) tek etiket tam-kadraj kutu** — 5231'i `Fruit`, 159'u
> `Leaf`. Aynı kusur `findik/organs/hazelnut_organs_split` ön-etiketlerinde
> de bulundu: 4310 `hazelnut_cluster` kutusunun **hepsi birebir aynı**
> (`0.499 0.499 0.8 0.8`).

**⚠️ ALAN EŞİĞİ BU HATAYI YAKALAYAMAZ.** Denendi ve ölçümle çürütüldü:
21.106 gerçek organ kutusunda `Fruit` p90 = **%90.9**, `Leaf` p90 =
**%66**, ikisinin de maksimumu **%98.9**. Yakın çekimde tek çilek kareyi
gerçekten doldurur. %64'ü eleyen bir eşik gerçek kutuların **%17.3'ünü**
de atardı.

**Koruma:** Eşik değil, **denetim**. Yeni pakette şunu ölçün:

```bash
# Tam-kadraj kutuların oranı — %1'i geçiyorsa kaynağı inceleyin
python - <<'PY'
from pathlib import Path
import importlib.util
s = importlib.util.spec_from_file_location('i', 'scripts/imgsz_oner.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
tam = toplam = 0
for e in Path('datasets/<urun>/<ad>').rglob('labels/*.txt'):
    for satir in e.read_text(errors='ignore').splitlines():
        c = m.etiket_satiri(satir.split())
        if not c: continue
        toplam += 1
        if c[3] > 0.99 and c[4] > 0.99: tam += 1
print('tam-kadraj: %d / %d  (%%%.1f)' % (tam, toplam, 100*tam/max(toplam,1)))
PY
```

Ayrıca **benzersizlik** denetimi: bir sınıfın bütün kutuları aynı değere
sahipse o sınıf hiç bilgi taşımıyordur.

### 2.7 Bitkiler arası sınıf karışması

**Belirti:** Yok — hata vermez. Yanlış bitkinin tedavisi önerilir.

**Sebep:** Hastalık adları bitkiler arasında ortaktır, etkenleri değil.
Ölçülen çakışma (çilek kütüğü ↔ fındık): `Leaf Spot` birebir aynı;
`Anthracnose`, `Powdery Mildew`, `Spider Mites`, `Mold`, `Weevil`
kısmen çakışıyor.

**Koruma:** Ürüne göre klasörleme — hem veri hem yapılandırma:

```
datasets/cilek/…        configs/urunler/cilek/…
datasets/findik/…       configs/urunler/findik/…
```

Sınıf ID'leri ürün **içinde** `0..n-1`'dir. Dataset'ler ortak havuzda
**birleştirilmez**. Doğrulandı: `Leaf Spot` çilekte "Fungal
(Mycosphaerella fragariae)", fındıkta "Fungal (Piggotia coryli)" döner —
aynı ad, ayrı kayıt. Bkz. [COK_BITKILI_YAPI.md](COK_BITKILI_YAPI.md).

### 2.8 `Healthy` sınıfı uzman modele sızıyor

**Belirti:** Hastalıklı yaprakta kararsız sonuç; arayüzde "Healthy Leaf"
bir bulgu gibi listeleniyor ve tedavi kartı boş çıkıyor.

**Sebep:** Roboflow paketlerinin çoğunda `Healthy` sınıfı vardır ve
düşünülmeden alınır. Bu mimaride sağlıklı durumu **organ modelinden
türetilir**: organ yaprağı bulur, uzman model bulgu çıkarmazsa yaprak
sağlıklıdır. Sınıf olarak da eklenirse aynı yaprak iki kutuya girer.

**Koruma:** Etiketi sil, **görüntüyü tut** — etiketsiz görüntü negatif
örnektir, silinirse hatalı pozitif artar:

```bash
python scripts/harici_paket_duzelt.py <paket> --urun <urun> \
    --arka-plana-al "Healthy,Healthy Leaf,Fresh"
```

Betik yazım hatasında sessizce geçmez: olmayan sınıf adı verilirse
durur ve dataset'teki adları listeler.
Test: `test_harici_paket.py`

**İSTİSNA — `rol: tekil` akışları.** Orada organ modeli çalışmaz, bu
yüzden "nesne var ve kusursuz" ile "fotoğrafta nesne yok" başka türlü
ayrılamaz. `findik_kalite` modelindeki `Sound Nut` bu yüzden sınıftır:
sağlık sınıfı değil, **varlık** sınıfıdır — o akıştaki organ karşılığı.

---

## 3. Eğitim hataları

### 3.1 Oturum epoch 1-2'de sessizce ölüyor

**Belirti:** Eğitim başlıyor, birkaç epoch ilerliyor, **hata vermeden** duruyor.

**Sebep:** RAM önbelleği. Görüntüler açılmış halde tutulur:
`görüntü × imgsz² × 3 bayt`. 13.000 görüntü @1024 ≈ 41 GB.

**Koruma:** Notebook 5️⃣ hücresi dataset büyüklüğünü sayar ve RAM'in
%40'ını aşacaksa önbelleği kapatır. Sayım **seçilen modelin** klasöründe
yapılır — sabit `dataset/` yazıldığında uzman modellerde 0 sayılıp önbellek
sessizce kapanıyordu.

### 3.2 Yanlış koşudan devam ediliyor

**Belirti:** `organ_detection` seçildi ama "Eğitim zaten tamamlanmış" deyip
atladı; üstelik başka modelin ağırlığı `organ.pt` adıyla kopyalandı.

**Sebep:** Bütün koşular Drive'da tek `results/` klasörünü paylaşır. Koşu
arayıcı ince ayar koşularını **ad ekinden** buluyordu ama model adına
bakmıyordu.

**Koruma:** `KOSU_ONEKI` — uzman modelde `EGITILECEK`, birleşikte
`strawberry`. Ayrıca boru hattı kopyası **sınıf doğrulaması** yapar.
Test: `test_notebook_kosu.py`

### 3.3 Colab sekmesi bayat kalıyor

**Belirti:** `git pull` yapıldı ama hücre kodu eski; aynı hata tekrarlıyor.

**Sebep:** Colab, GitHub'dan açılan notebook'u **kendi sunucusunda**
önbelleğe alır. `Ctrl+Shift+R` tarayıcı önbelleğini temizler, Google
tarafındakine dokunmaz.

**Koruma:**
- Notebook 3️⃣ hücresi `NOTEBOOK_SURUM` ile depodaki sürümü karşılaştırır ve uyarır
- Commit SHA'lı link kullanın (önbelleğe takılmaz):
  `colab.research.google.com/github/<user>/<repo>/blob/<SHA>/<notebook>.ipynb`

### 3.4 Sonuçlar Colab'in geçici diskine yazılıyor

**Belirti:** Oturum kapandı, eğitim tamamen kayboldu.

**Sebep:** `project` yapılandırmada `runs/train` (göreli).

**Koruma:** `TRAIN_CONFIG['project'] = str(RESULTS_DIR)` — Drive altı.
İnce ayar dalı da bu anahtarı **devralmalı** (bir kez unutulmuştu).
Test: `test_notebook_kosu.py::test_ince_ayar_project_devralir`

### 3.5 Drive eşitleme kopyası koşuyu ikiye bölüyor

**Belirti:** Gözcü "eğitim ölmüş, 92/200" diyor ama eğitim aslında bitmiş.

**Sebep:** Drive masaüstü uygulaması çakışmada yerel aynada `<ad> (1)`
klasörü üretir. Colab tarafında tek klasör vardır.

> Gerçek olay: `bocek_teshis (1)` epoch 1-108, `bocek_teshis` epoch 109-200.
> Ayrı okununca ikisi de "yarım" göründü; koşu **200/200 tamamlanmıştı**.

**Koruma:** `egitim_izle.py` `(N)` sonekli klasörleri birleştirir, ilerlemeyi
satır sayısından değil `epoch` sütununun en büyüğünden okur.
Test: `test_egitim_izle.py`

### 3.6 Kesintiyi kimse fark etmiyor

**Belirti:** Saatlerce GPU boşa gidiyor.

> `organ_detection-2` epoch 190'a 23:55'te geldi, 195'te öldü, **08:35'e
> kadar** öyle kaldı. Kalan 5 epoch 15 dakika sürdü. Kayıp: **8,5 saat**.

**Koruma:** `python scripts/egitim_izle.py --bekle` — koşu durunca haber verir.
Colab oturum ölümlerini kodla engelleyemeyiz; kaybı azaltabiliriz.

---

## 4. Çıkarım (inference) hataları

### 4.1 Tek `imgsz` bütün modellere dayatılıyor

Bkz. [MIMARI.md § 2](MIMARI.md). Kısaca: uzman modeller ROI kırpıntısı
görür, 1024 onları çökertir. Model başına `imgsz` kütükte yazılıdır.

### 4.2 Videoda sayı şişiyor

**Belirti:** 4 meyveli sabit sahne "11 tespit" diyor.

**Sebep:** Her örneklenen karenin kutuları biriktiriliyordu, kareler arası
eşleştirme yoktu.

**Koruma:** `app/takip.py` — kareler arası takip, benzersiz sayım.
Arayüzde `benzersiz_sayi` gösterilir, `tespit_sayisi` değil.
Ayrıntı: [GORUNTU-KAYNAKLARI.md](GORUNTU-KAYNAKLARI.md)

### 4.3 Sabit kare adımı farklı fps'te farklı davranıyor

**Sebep:** "Her 15. kare" 30 fps'te 0,5 sn, 60 fps'te 0,25 sn demektir.
Takibin arama penceresi süreye bağlı olduğu için de kayar.

**Koruma:** `VIDEO_ORNEK_ARALIK_SN = 0.5`; adım `fps × aralık`.
Test: `test_takip.py::TestOrneklemeAdimi`

---

## 5. Arayüz ve ağ hataları

### 5.1 Koyu temada metin okunmuyor

**Belirti:** Tedavi önerileri beyaz kutuda açık yazıyla, neredeyse görünmez.

**Sebep:** `background: var(--zemin2, #fafafa)` yazılmıştı ama `--zemin2`
**hiç tanımlı değildi**. Tanımsız değişken sessizce yedeğe düşer.

**Koruma:** Denetim testi — CSS'teki tüm kurallar taranır; açık sabit zemin
veren bir kural yazı rengi de vermiyorsa test kırılır.
Test: `test_tema.py::TestOkunabilirlik`

### 5.2 Telefon bağlanamıyor — IP değişti

**Belirti:** `ERR_CONNECTION_TIMED_OUT` / `REFUSED`.

**Sebep:** Router DHCP ile yeni IP veriyor (.103 → .101 → .104).

**Koruma:** **Makine adı kullanın** — mDNS ile yayınlanır, IP değişse de
değişmez:

```
https://<MAKINE-ADI>.local:8443/
```

`/baglan` sayfası güncel adresleri **QR koduyla** gösterir ve kalıcı olanı
öne çıkarır.

### 5.3 Sertifika uyarısı geçmiyor

**Sebep:** Sertifika adresleri `sys.argv[1:]` ile alınıyordu; biri
`--help` yazınca `--help` bir SAN girdisi oldu ve **IP tespiti hiç
çalışmadı**. Sertifika yalnızca `localhost` kapsıyordu — betik hata vermeden
"✅ Hazır" diyordu.

**Koruma:** `https_sertifika.py` bayrakları eler, makine adını ve mevcut
IP'leri **her zaman** ekler. `/baglan` sayfası kapsamı kontrol eder.
Test: `test_https_sertifika.py`

### 5.4 Modül statik dosyaları 404

**Sebep:** Çekirdek `/static`'i zaten bağlamış; Starlette önce onu eşleştirir.

**Koruma:** Modül statikleri `/statik/<ad>` altına bağlanır.

---

## 6. Kod tarafında hata yönetimi deseni

### Kural 1 — Model çağrısı akışı kesmemeli

```python
try:
    r = model(goruntu, conf=esik, imgsz=boy, verbose=False)[0]
except Exception as e:
    logger.warning(f'Tahmin başarısız: {e}')
    return []          # boş sonuç, çökme yok
```

Bir uzman model patlarsa diğerleri çalışmaya devam eder.

### Kural 2 — Eksik model `None` döner, istisna atmaz

`modeller.yukle()` dosya yoksa `None` döner. Çağıran taraf kontrol eder.
Bu, kademeli geçişi mümkün kılar.

### Kural 3 — Kullanıcıya dönen hata NE YAPILACAĞINI söyler

```
Analiz için model yok. En az organ modeli (models/<urun>/organ.pt) gerekir.
Eksik: bocek_teshis. Kurulum: python scripts/model_kur.py --listele
```

### Kural 4 — Bozuk yapılandırma akışı kesmez

`siniflar.py`, `modeller.py`, `sonuc_ozeti.izi_coz()` bozuk YAML/JSON'da
boş sözlük döner ve hatayı **loglar**. Uygulama açılır, sorun görünür kalır.

### Kural 5 — Yan iş, ana işi bozmaz

Konum atama, böcek yedeği, takip — hepsi `try/except` içinde. Biri
başarısız olursa analiz yine kaydedilir.

---

## 7. Yeni hata eklerken

Bu belgeye ekleyin ve şu üçünü yazın:

1. **Belirti** — kullanıcı ne görüyor?
2. **Sebep** — neden oluyor? (Varsayım değil, ölçüm.)
3. **Koruma** — tekrar etmesini ne engelliyor? Testse test adını yazın.

Testsiz koruma, koruma değildir. Bu belgedeki hataların çoğu ikinci kez
yaşandığı için test edildi.
