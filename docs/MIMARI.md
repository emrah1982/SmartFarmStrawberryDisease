# Mimari

> Bu belge sistemin **bugünkü** halini anlatır. Tasarım kararlarının
> gerekçeleri de yazılıdır — "neden böyle yapmıştık" sorusuna dönüp
> bakabilmek ve aynı hataya geri dönmemek için.

---

## 1. Temel karar: hiyerarşik çok modelli yapı

Sistem **tek bir model** değil, **beş bağımsız model** kullanır. Görüntü şu
yoldan geçer:

```
görüntü
  │
  ├─ organ modeli            → Leaf / Fruit / Flower kutuları
  │
  ├─ her organ için ROI kırpma (%12 pay ile)
  │
  ├─ o organda TETİKLENEN uzman modeller
  │     Leaf  → yaprak_hastalik
  │     Fruit → meyve_hastalik + olgunluk
  │
  ├─ koordinatları orijinal görüntüye geri dönüştürme
  └─ eşik süzme + örtüşen kutuları birleştirme (NMS)
```

### Neden tek model değil?

Tek modelde bütün sınıflar birbirine karışıyordu. Yaşanan somut örnek:

> Olgunluk sınıfları **yaprakları** "olgunlaşmamış çilek" sanıyordu.
> Ölçüldü: gerçek meyve güven medyanı 0,58 · yaprak yanlış pozitifleri
> 0,69-0,79. **İki grup üst üste biniyordu — ayıran bir eşik yoktu.**
> Eşiği yükseltmek gerçek meyveleri de eliyordu.

Hiyerarşide olgunluk modeli **yalnızca meyve kırpıntısı** görür. Bir yaprağı
olgunlaşmamış meyve diye işaretlemesi **yapısal olarak imkânsızdır**. Sorun
eşikle bastırılmaz, ortadan kalkar.

### İkinci fayda: bağımsız eğitim

Yeni bir zararlı eklendiğinde yalnızca zararlı modeli yeniden eğitilir;
hastalık modelleri dokunulmadan kalır. Tek modelde her ekleme, tüm sistemin
yeniden eğitilmesini gerektiriyordu.

### Üçüncü fayda: SAĞLIKLI durumu ayrı sınıf gerektirmez

**Organ modeli aynı zamanda "sağlıklı mı" sorusunun cevabıdır.**

```
organ modeli yaprağı buldu  +  yaprak hastalık modeli orada bir şey bulamadı
                            ↓
                    yaprak SAĞLIKLIDIR
```

Arayüz bunu açıkça yazar: *"🌿 Yapraklarda (5 adet) — Bakıldı, bulgu yok"*.

Bu yüzden dataset'lerde **`Healthy Leaf` / `Healthy Nut` diye bir sınıf
yoktur** ve olmamalıdır. Sağlıklı örnekler etiketsiz (background) görüntü
olarak eğitime girer.

**Neden sınıf yapılmıyor?**

| Sağlıklıyı sınıf yapmak | Organ modelinden türetmek |
|---|---|
| Her sağlıklı yaprağı kutulamak gerekir — etiketleme maliyeti katlanır | Organ zaten kutulanıyor, ek iş yok |
| "Sağlıklı" tanımı belirsiz: hafif leke sağlıklı mı? | Karar hastalık modelinde, eşikle ayarlanabilir |
| Hastalıklı yaprak hem `Leaf Spot` hem `Healthy` kutusuna girer — çelişki | Çelişki imkânsız |
| Yeni hastalık eklenince "sağlıklı" tanımı değişir, eski etiketler bozulur | Etkilenmez |

Ayrıca ayrım korunur: *"5 yaprak gördüm, hastalık bulmadım"* ≠ *"hiç yaprak
görmedim"*. Bu bilgi `Analiz.boru_izi` içinde saklanır (§ 6).

---

## 2. Model kütüğü — tek yetkili liste

Hangi modelin ne zaman çalışacağı **kodda değil**
`configs/urunler/<urun>/modeller.yaml` içinde yazılıdır.

| Model | rol | tetik | imgsz | eşik | Sınıflar |
|---|---|---|---|---|---|
| `organ` | organ | — | 640 | 0,20 | Flower, Fruit, Leaf |
| `yaprak_hastalik` | yaprak_hastalik | leaf | 256 | 0,20 | Angular Leafspot, Leaf Spot, Powdery Mildew Leaf, Gray Mold |
| `meyve_hastalik` | meyve_hastalik | fruit | 192 | 0,20 | Anthracnose Fruit Rot, Powdery Mildew Fruit, Blossom Blight, Gray Mold |
| `olgunluk` | olgunluk | fruit | 128 | 0,30 | unripe / semi_ripe / ripe |
| `zararli` | zararli | leaf, fruit | — | 0,25 | Thrips, Spider Mites, … (**veri yok, kapalı**) |
| `bocek_teshis` | **tekil** | **boş** | — | 0,35 | 6 böcek türü |
| `miras` | miras | — | — | 0,20 | eski 10 sınıflı model (**kapalı**) |

### `tetik: []` neden kritik?

`bocek_teshis` modeli 416×416 **makro böcek fotoğraflarıyla** eğitildi; kutu
alanı medyanı karenin %14,8'i. ROI boru hattı ise saha görüntüsünden kırpılmış
yaprak parçası verir — orada bir zararlı kırpıntının %1'inden azını kaplar.

Bu model ROI akışına bağlanırsa **yanlış ölçekte çalışır ve olmayan böcekler
bulur**. Boş tetik listesi bunu imkânsız kılar; `tests/test_tekil_model.py`
her organ için bunu doğrular.

### Model başına `imgsz` — ölçülerek belirlendi

Tek bir genel `imgsz` (1024) bütün modellere dayatılıyordu. Uzman modeller
**tam görüntüyü değil ROI kırpıntısını** görür (60-250 piksel); 1024'e
büyütmek 4-15 kat blowup demektir. Ölçüm (27 fotoğraf, 83 ROI):

| Model | imgsz 1024 | Ölçülen en iyi |
|---|---|---|
| olgunluk | 34/50 ROI · ort 0,380 | **128** → 47/50 · 0,459 |
| meyve_hastalik | 12/50 ROI · ort 0,236 | **192** → 36/50 · 0,533 |
| yaprak_hastalik | 17/33 ROI · ort 0,497 | **256** → 27/33 · 0,518 |

Tek bir sera fotoğrafında sonuç: **1 tespit → 4 tespit**.

> ⚠️ Bu ölçüm **güveni** optimize eder, **doğruluğu** değil — etiketli saha
> ROI'si yok. Yön kesin (1024 yanlış), tam optimum 128-256 aralığında ve
> etiketli veriyle yeniden ölçülmeli.

### Organ eşiği neden cömert (0,20)?

Asimetrik risk: organ modeli yalnızca **ROI seçer**. Yanlış bir ROI
zararsızdır (uzman model orada bir şey bulamaz), ama **kaçırılan organ tüm
zinciri keser** — meyve bulunmazsa olgunluk da hastalık da hiç çalışmaz.

---

## 3. Katmanlar ve bağımlılık yönü

```
                       app/main.py  (FastAPI, HTTP, şablon)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
      app/pipeline.py  app/detector.py  app/moduller/*
              │             │
              ▼             ▼
      app/modeller.py  app/takip.py
              │
              ▼
   app/siniflar.py · app/urunler.py · app/tedavi.py
   app/sonuc_ozeti.py · app/ag.py · app/cizim.py
```

**Kural:** ok yönü hep aşağı. Alt katman üstünü bilmez.

Bu kural testle sabitlenmiştir — şu modüller `fastapi`, `sqlalchemy`,
`jinja2`, `app.main` ithal **edemez**:

| Bileşen | Satır | İş |
|---|---|---|
| `app/takip.py` | 348 | Kareler arası nesne takibi, benzersiz sayım |
| `app/sonuc_ozeti.py` | 223 | Sonucu organa göre gruplama |
| `app/tedavi.py` | 86 | Tedavi kütüğü, organ bazlı çözümleme |
| `app/ag.py` | 135 | Sunucu adresleri (mDNS + IP) |
| `app/moduller/bocek/servis.py` | 273 | Böcek teşhis mantığı |

Faydası somut: `takip.py`'yi başka bir projeye kopyalasan çalışır. Drone
akışı eklendiğinde bu dosyaya dokunulmayacak, sadece kare kaynağı bağlanacak.

---

## 4. Modül sistemi

İsteğe bağlı yetenekler `app/moduller/` altında kendi klasöründe durur:

| Modül | Ne yapar | Kendi tablosu |
|---|---|---|
| `canli` | WebSocket canlı akış, oturum kaydı | — |
| `bocek` | Böcek teşhis (ayrı akış) | `bocek_kayitlari` |
| `konum` | EXIF GPS, blok/sıra, yaygınlık haritası | `analiz_konumlari` |
| `veritabani` | Tarayıcıdan veritabanı inceleme | — |

Her modül şunu sağlar: `ad`, `baslik`, `yol`, `router`,
`tablolar_olustur(engine)`. Kapatmak için
`app/moduller/__init__.py` → `yuklu_moduller()` listesinden çıkarmak yeterli.

**Neden ayrı tablo?** Modül silindiğinde çekirdek şema etkilenmesin. Ayrıca
böcek kayıtları `Analiz` tablosuna yazılsaydı hastalık istatistikleri ve
yaygınlık haritası bozulurdu — "şu serada 12 tespit" sayısı anlamını yitirirdi.

---

## 5. Ürün (bitki) kapsamı

Çok bitkili kuruluma hazır. Her ürünün **kendi** yapılandırması ve modelleri:

```
configs/urunler/<urun>/    siniflar.yaml · modeller.yaml · veri.yaml · tedavi_onerileri.yaml
models/<urun>/             organ.pt · leaf_disease.pt · …
datasets/<urun>/           organ_detection/ · leaf_disease/ · …
```

Sınıf ID'leri **ürün içinde** 0..n-1'dir; ürünler arası çakışma imkânsızdır.
Ayrıntı: [COK_BITKILI_YAPI.md](COK_BITKILI_YAPI.md)

---

## 6. Sonuç sunumu — organa göre

Tespitler **sınıf adına göre değil organa göre** gruplanır. Sebep:

> `Gray Mold` hem yaprak hem meyve modelinde tanımlı. Sınıfa göre
> gruplayınca "Gray Mold 5 adet" çıkıyordu — 3'ü yaprakta, 2'si meyvede
> olduğu görünmüyordu. **Meyvedeki kurşuni küf acil hasat/imha, yapraktaki
> havalandırma demektir.** Kozmetik değil, yanlış tarımsal karar doğuruyordu.

Tedavi önerileri de organa göre çözülür
(`configs/urunler/<urun>/tedavi_onerileri.yaml` içinde isteğe bağlı `organ:`
bloğu). Aciliyet bile organa göre değişir:
yapraktaki botrytis `orta`, meyvedeki `yuksek`.

`Analiz.boru_izi` (JSON) sütunu "ne GÖRÜLDÜ" bilgisini saklar — tespit listesi
yalnızca "ne BULUNDU"yu içerir. İkisi olmadan *"5 yaprak gördüm, hastalık
bulmadım"* ile *"hiç yaprak görmedim"* ayırt edilemez.

---

## 7. Veri modeli

```
Üretici → Sera → Kamera → Analiz → Tespit
                              └──→ AnalizKonum   (konum modülü)
BocekKaydi                                       (böcek modülü, bağımsız)
```

`Analiz` üzerindeki sayım alanları — **karıştırılmamalı**:

| Alan | Anlamı |
|---|---|
| `tespit_sayisi` | Ham kutu sayısı. Videoda aynı nesneyi her karede tekrar sayar |
| `kare_basina_en_cok` | Tek karede görülen en çok nesne (güvenilir alt sınır) |
| `benzersiz_sayi` | **Kullanıcıya gösterilen sayı** — kareler arası takip sonrası |

---

## 8. Geriye dönük uyum

Sistem, eksik modelle **çökmez**:

- Organ modeli yoksa → miras (tek) modele düşer
- Uzman model yoksa → o organ için tespit üretilmez, akış sürer
- Hiç model yoksa → 400 döner ve **ne yapılacağını söyler**

Bu, mimariye kademeli geçişi mümkün kılan tasarımdır. Bugün miras model
kapalı (dört uzman hazır) ama yol açık duruyor.

---

## İlgili belgeler

- [HATA-YONETIMI.md](HATA-YONETIMI.md) — hata yönetimi ve **tekrarlayan tuzaklar**
- [VERI-ALMA.md](VERI-ALMA.md) — Roboflow/Kaggle veri alma kuralları
- [GORUNTU-KAYNAKLARI.md](GORUNTU-KAYNAKLARI.md) — fotoğraf/video/drone/canlı koşulları
- [EGITIM.md](EGITIM.md) — eğitim ve model kurulumu
- [COK_BITKILI_YAPI.md](COK_BITKILI_YAPI.md) — yeni bitki ekleme
