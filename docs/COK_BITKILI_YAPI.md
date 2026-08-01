# Çok Bitkili Kurulum — Ürün Kapsamı

Bugün yalnızca çilek var. Bu belge, ikinci bitki eklendiğinde **hiçbir şeyin
karışmaması** için kurulan yapıyı ve yeni bitki ekleme adımlarını anlatır.

## Çözülen asıl risk: sınıf ID çakışması

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

Kod değişikliği **gerekmez**. Docker'da `configs/urunler` klasörü tek parça
bağlıdır; yeni bitki eklendiğinde `docker-compose.yml` değişmez.

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
