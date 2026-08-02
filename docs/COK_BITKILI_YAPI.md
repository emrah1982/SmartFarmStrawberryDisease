# Çok Bitkili Kurulum — Ürün Kapsamı

Bu belge, ikinci bitki eklendiğinde **hiçbir şeyin karışmaması** için kurulan
yapıyı ve yeni bitki ekleme adımlarını anlatır.

**Durum (2026-08-01):**

| Ürün | Yapılandırma | Dataset | Model |
|---|---|---|---|
| `cilek` | tam | 5 dataset | 5 model eğitildi, 4'ü boru hattında |
| `findik` | tam (iskelet) | `findik_kalite` (7.156 görüntü) | yok — eğitim bekliyor |

## Çözülen asıl risk: hastalık adları bitkiler arasında AYNI

Bitki hastalıklarının adları ortaktır ama **etkenleri ve tedavileri farklıdır.**
Ölçülen çakışmalar (çilek kütüğü ↔ fındık için önerilen sınıflar):

| Ad | Çilekte | Fındıkta |
|---|---|---|
| **Leaf Spot** | *Mycosphaerella fragariae* | *Piggotia coryli* — farklı mantar |
| **Anthracnose** | *Colletotrichum acutatum*, meyvede | dalda ve kabukta |
| **Powdery Mildew** | *Podosphaera aphanis* | *Phyllactinia guttata* |
| **Spider Mite / Aphid** | tür farklı | tür farklı |

Tek kütükte toplansaydı: aynı ada tek kayıt düşerdi, fındık yaprağındaki
lekeye **çilek tedavisi** önerilirdi. Bu sessiz bir hatadır — sistem hata
vermez, sadece yanlış ilacı söyler.

**Bu yüzden hem `datasets/` hem `configs/urunler/` bitkiye göre klasörlenir.**
Aynı adı taşıyan iki hastalık iki ayrı dosyada yaşar, birbirini hiç görmez.

```
datasets/cilek/leaf_disease/     "Leaf Spot" → çilek yaprak lekesi
datasets/findik/leaf_disease/    "Leaf Spot" → fındık yaprak lekesi
```

Dataset'ler asla ortak bir havuzda birleştirilmez; birleştirme girişimi
yukarıdaki tabloyu geri getirir.

## Ama her şey ürüne bağlı DEĞİL — ortak kapsam

Yukarıdaki kural hastalıklar içindir. **Böcek türü için durum tersidir:**

| varlık | ürüne bağlı mı? | neden |
|---|---|---|
| `Leaf Spot` | ✅ evet | çilekte *Mycosphaerella*, fındıkta *Piggotia* |
| `Mole Cricket` | ❌ hayır | her bitkide *Gryllotalpa* — **aynı tür** |

Böcek teşhis akışı makro fotoğraftan **canlıyı** tanır; bitkiyi hiç görmez.
Ürün başına kopyalansaydı:

- aynı 20 MB'lık ağırlık her ürün klasörüne tekrar konurdu
- aynı dataset n kez saklanırdı
- yeni bitki eklenince böcek kütüğü elle kopyalanır, er geç biri unutulurdu

Bu yüzden ortak varlıklar **kapsamsız kökte** durur:

```
configs/ortak/     modeller.yaml · siniflar.yaml · tedavi_onerileri.yaml
models/            bocek_teshis.pt
datasets/          bocek_teshis/
```

**Birleşme kuralı:** ortak kütük önce yüklenir, ürününki üstüne yazılır.
Aynı ad iki yerdeyse **ürününki kazanır** — bir bitki için özelleştirme
gerekirse mümkün olsun diye. (`Spider Mites` böyle: ortak kütükte var, ama
çilek kütüğündeki kayıt onu eziyor.)

**Karar ölçütü — bir varlık ne zaman ortaktır?**

> Ancak **aynı gerçek nesneyi** gösteriyorsa.
> Hastalık *adı* ortak olabilir; **hastalık** ortak değildir.

Organlar, olgunluk ve hastalıklar ürüne özgü kalır. Model kütüğünde
`ortak: true` yazan tanım, ağırlığını `models/<urun>/` altında değil
`models/` kökünde arar.

Uygulama: `app/urunler.py` → *ORTAK KAPSAM*.
Test: `test_urun_kapsami.py`, `test_urunler.py::test_ortak_model_urun_klasorunde_ARANMAZ`

## İkinci risk: sınıf ID çakışması

Sınıf ID'leri etiket dosyalarında **sayı** olarak saklanır. Tek kütükte
kalsaydı, domates eklendiğinde iki kötü seçenekten biri olurdu:

| Seçenek | Sonuç |
|---------|-------|
| Domatesin "Leaf Spot"u çileğinkiyle **aynı ID** | Model iki farklı hastalığı tek sınıf sanar |
| **Yeni ID** alır | Her model diğer bitkinin sınıflarını da taşır, gereksiz şişer |

Çözüm: her ürünün **kendi kütüğü**. ID'ler ürün içinde `0..n-1`'dir; ürünler
arası çakışma kavramsal olarak imkânsızdır — uzman dataset'lerde yaptığımızın
aynısı.

## Dizin düzeni

```
configs/urunler/<urun>/
    siniflar.yaml            sınıf kütüğü (ad, grup, ID, eşik, aktif)
    modeller.yaml            model kütüğü (dosya, rol, tetik organ)
    tedavi_onerileri.yaml    ürüne özgü tedavi metinleri
    veri.yaml                eğitim dataset yapılandırması
    class_aliases.yaml       kaynak sınıf adı eşleştirmeleri

models/<urun>/               organ.pt · leaf_disease.pt · ... · best.pt
datasets/<urun>/             organ_detection/ · leaf_disease/ · ...
```

**Ürün-bağımsız kalanlar:** `train_config.yaml`, `finetune_config.yaml` —
eğitim hiperparametreleri bitkiden bağımsızdır.

## Ürün nereden belirlenir?

```
1. Açıkça verilirse (fonksiyon parametresi)
2. Seranın `urun` alanı          ← asıl kaynak; üretici zaten giriyor
3. VARSAYILAN_URUN ortam değişkeni
4. 'cilek'
```

**Bitki türünü görüntüden tespit etmek birincil yol DEĞİLDİR.** Kare tamamen
yaprakla dolduğunda çilek/domates ayrımı güvenilmezdir; oysa "hangi sera"
bilgisi bedava ve kesindir. Görüntüden tespit ileride yalnızca **doğrulama**
için eklenebilir: *"Bu sera çilek kayıtlı ama görüntü domates gibi görünüyor."*

`Sera.urun` alanı serbest metindir ('Çilek'); `urunler.slug()` bunu klasör
anahtarına çevirir (`cilek`). Türkçe karakterler ASCII'ye indirgenir.

## Kayıtlar

`Analiz.urun` sütunu **kaydın hangi bitkinin model setiyle üretildiğini**
saklar. Sera sonradan başka ürüne geçse bile geçmiş kayıt doğru kalır.
Mevcut kayıtlar otomatik geçişle `cilek` olarak işaretlendi.

## Yeni bitki ekleme

```bash
# 1. Yapılandırma klasörü
mkdir -p configs/urunler/domates models/domates datasets/domates
cp configs/urunler/cilek/modeller.yaml configs/urunler/domates/

# 2. Sınıf kütüğü — ID'ler SIFIRDAN başlar, çilekle ilgisi yoktur
cat > configs/urunler/domates/siniflar.yaml <<'YAML'
Early Blight:
  tr: Erken Yanıklık
  grup: hastalik
  id: 0
Late Blight:
  tr: Geç Yanıklık
  grup: hastalik
  id: 1
YAML

# 3. Dataset ve eğitim (çilekteki akışın aynısı)
# 4. Modelleri models/domates/ altına koy
# 5. Serayı 'Domates' ürünüyle tanımla → kayıtlar otomatik o kapsama düşer
```

Kod değişikliği **neredeyse hiç** gerekmez. Docker'da `configs/urunler`
klasörü tek parça bağlıdır; yeni bitki eklendiğinde `docker-compose.yml`
değişmez.

### Fındık eklenirken kodda değişen tek yer

`app/sonuc_ozeti.py: ORGAN_GORUNUM` — organ adının arayüzdeki simgesi ve
Türkçe başlığı. Fındık için `nut`, `husk`, `branch` eklendi; olmasaydı
üçü de "📄 Organ ayrımı yapılmadan" başlığına düşecekti (çökme değil,
sessiz kalite kaybı).

Bu sözlük **ürün adına değil ORGAN adına** bakar: `leaf` hem çilekte hem
fındıkta aynı satırı kullanır. Yeni bitkinin organları mevcutlarla
örtüşüyorsa dokunmak gerekmez.

### Fındıktan çıkan somut ders

`hazelnut detection v9` paketi adı doğru olduğu için boru hattına
bağlanacaktı. Ölçüm durdurdu: görüntü başına **en fazla 1 kutu**, kutular
kadraj ortasında — bu bahçe verisi değil, hasat sonrası ayıklama verisi.
`rol: tekil, tetik: []` ile ayrı akışa alındı.
Bkz. [HATA-YONETIMI.md](HATA-YONETIMI.md) § 2.6.

## Geriye dönük uyumluluk

Ürün klasörü yoksa eski (kapsamsız) yollara düşülür:
`configs/urunler/domates/train_config.yaml` yoksa `configs/train_config.yaml`
kullanılır. Böylece bu yapı mevcut kurulumu bozmadan devreye girdi —
doğrulandı: aynı görüntü yine aynı 2 tespiti veriyor, kayıtta `urun='cilek'`.

## Sınırlar

- **Organ modeli ürün başınadır.** Çilek yaprağı ile domates yaprağı görsel
  olarak farklıdır. İleride tek çok bitkili organ modeli denenebilir, ama
  başlangıçta ayrı tutmak daha güvenlidir.
- **Tedavi önerileri ürüne özgüdür** — aynı hastalık adı farklı bitkide farklı
  müdahale gerektirebilir.
- Arayüzde ürün seçici henüz yok; ürün seradan geliyor. Birden çok bitkiyle
  çalışıldığında geçmiş/panel için ürün filtresi eklenmelidir.

---

## İlgili belgeler

- [MIMARI.md](MIMARI.md) — model kütüğü ve boru hattı
- [VERI-ALMA.md](VERI-ALMA.md) — yeni bitkinin dataset'ini hazırlama
- [EGITIM.md](EGITIM.md) — eğitim ve model kurulumu
