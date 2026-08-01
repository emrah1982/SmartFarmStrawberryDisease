# Görüntü Kaynakları — Fotoğraf, Video, Drone, Canlı Akış

> Her kaynağın **neyi doğru verdiği** ve **neyi veremediği** yazılıdır.
> Özellikle sayım: aynı sayı her kaynakta aynı şeyi ifade etmez.

---

## 1. Özet tablo

| Kaynak | Sayım doğru mu? | Takip | Çizgi sayımı | Böcek yedeği |
|---|---|---|---|---|
| **Fotoğraf** | ✅ Her nesne bir kez | gereksiz | — | ✅ |
| **IP kamera (anlık)** | ✅ Tek kare | gereksiz | — | ✅ |
| **Video** | ⚠️ Takiple benzersiz | ✅ | isteğe bağlı | ❌ |
| **Drone (video)** | ⚠️ Takiple benzersiz | ✅ | ✅ önerilir | ❌ |
| **Canlı akış** | ⚠️ Tur boyunca benzersiz | ✅ | isteğe bağlı | ❌ |

---

## 2. Fotoğraf — en güvenilir

Her nesne bir kez sayılır, sayım tartışmasızdır. Model kararı burada
verilmelidir.

### Çekim koşulları — iki mod

Kullanıcıya **"yaprak mı meyve mi çekiyorsun"** diye sorulmaz. Sistem tek
karede yaprağı, çiçeği ve meyveyi ayrı ayrı bulup her birini kendi
uzmanına yönlendirir. Belirlenmesi gereken tek şey **mesafe**:

| Mod | Ne zaman | Mesafe | Ne verir |
|---|---|---|---|
| **Tarama** | Sıra boyunca yürürken | 80-150 cm | Çok organ, "nerede sorun var" |
| **Teşhis** | Şüpheli bölge görünce | 30-60 cm | Tek organ, yüksek doğruluk |

Taramada lezyonlar küçük göründüğü için güven düşer; erken evre hastalık
kaçabilir. **Bir şey çıkarsa yaklaşıp ikinci fotoğraf çekin.**

### Her modda geçerli

- **Işık:** Gündüz, dengeli. Doğrudan güneş ve sert gölge renk ipuçlarını
  bozar — kahverengi leke / gri küf / beyaz külleme ayrımı **renge** dayanır.
- **Arkadan ışık almayın:** Işık yapraktan geçince lekeler saydamlaşır;
  model bu görünümü eğitimde görmedi. Güneş **arkanızda** olsun.
- **Sade arka plan:** Gökyüzü/ağaç yerine toprak veya düz zemin.
- **Sabitlik:** Çekim anında bir saniye durun. Hareket bulanıklığı en sık
  doğruluk kaybı sebebidir.
- **Odak:** Telefonda hedefe dokunup odaklandığından emin olun.

### Böcek yedeği

Bitki analizi **hiçbir şey bulamazsa** görüntü böcek teşhis modeline de
sorulur ve sonuç **öneri** olarak gösterilir (eşik 0,55 — normal teşhiste
0,20). Yalnızca fotoğrafta çalışır.

---

## 3. Video — sayım dikkat ister

### Örnekleme SÜREYE göre

```
VIDEO_ORNEK_ARALIK_SN = 0.5      # her yarım saniyede bir kare
adım = fps × aralık
```

Sabit kare adımı (her 15. kare) 30 fps'te 0,5 sn, 60 fps'te 0,25 sn
demektir — aynı ayar farklı videolarda farklı davranır ve takibin arama
penceresi kayar. Ölçüldü, 3 saniyelik video:

```
30 fps → 6 kare işlendi
60 fps → 6 kare işlendi     (önce 6 ve 12 olurdu)
```

### Bulanık kare atlanır

Her örneklenen karenin keskinliği ölçülür (Laplacian varyansı); eşiğin
altındakiler modele verilmez. Kaç karenin atlandığı sonuç sayfasında yazar.
Tüm kareler bulanıksa **en keskin olan yine de işlenir** — kullanıcı boş
dönmesin.

### Sayım: `tespit_sayisi` ≠ nesne sayısı

Ölçülen gerçek olay:

```
4 meyveli SABİT sahne, 4 kare örneklendi  →  11 kutu
```

Aynı meyve her karede yeniden sayılıyordu. Artık `app/takip.py` kareler
arası eşleştirme yapıyor:

```
17 kutu  →  3 BENZERSİZ nesne
```

Arayüzde **`benzersiz_sayi`** gösterilir. `tespit_sayisi` ham kutu sayısı
olarak saklanır (etiketleme ve geçmiş için gerekli).

### Video çekim koşulları

- 15-20 saniye yeterlidir; uzun video işlemeyi yavaşlatır
- Yavaş yürüyün, **2-3 adımda bir yarım saniye durun** — duraklama
  anındaki kareler keskin olur
- Telefonu göğüs hizasında, bitkiye dönük ve sabit tutun
- Video **tarama modudur**: fotoğrafa göre düşük doğruluk verir. Şüpheli
  bölge görürseniz durup teşhis fotoğrafı çekin

---

## 4. Kareler arası takip (`app/takip.py`)

### Nasıl çalışır

Her kutuya kalıcı kimlik verilir. Eşleştirme ölçütleri:

| Ölçüt | Kural | Neden |
|---|---|---|
| Sınıf | **Aynı sınıf** olmalı | Olgun/olgunlaşmamış birleşirse sahte süreklilik olur |
| Örtüşme | IoU ≥ 0,30 | Konum yakınlığı tek başına yetmez (yan yana iki çilek) |
| Mesafe | Süreyle **büyüyen** pencere | 2 sn önce görülen nesne daha uzağa gitmiş olabilir |
| Kayıp | 1,5 sn tolerans | Yaprak arkasına giren meyve iki nesne sayılmasın |

Hız varsayımı: nesne saniyede kadrajın en çok **%35'ini** kat eder
(yürüyerek çekimde ölçülen tipik değer). Drone/hızlı pan için artırılabilir.

### Neden ByteTrack/Kalman değil?

O kütüphaneler **ardışık kare** bekler. Biz aralıklı örnekliyoruz (her
kareyi işlemek ~15 kat pahalı); aradaki hareket büyük olduğu için Kalman
öngörüsü zaten güvenilmez olur.

### Sınırı

Kamera çok hızlı hareket ederse veya nesneler sık ve benzerse (sıra sıra
çilek) eşleştirme hata yapabilir. Sonuç **"kesin sayı" değil "benzersiz
tahmin"** olarak sunulur.

---

## 5. Çizgi (tripwire) sayımı — drone ve transekt için

Ekrana sanal çizgi konur; nesne çizgiyi geçtiği anda sayılır.

### Ne zaman İŞE YARAR

Kamera **tek yönde ve düzenli** ilerlerken:
- Drone transekti
- Sıra boyunca sabit hızla yürüyüş
- Bant üstü ürün

Uzun taramalarda benzersiz-iz sayımından **daha sağlamdır**: iz kopup
yeniden kurulsa bile (bulanıklık, yaprak arkası) nesne çizgiyi bir kez
geçmiştir.

### Ne zaman İŞE YARAMAZ — önemli

| Durum | Sonuç |
|---|---|
| **Sabit çekim** | Hiçbir şey geçmez, sayı **0 kalır** (kadrajda 5 meyve olsa bile) |
| **Dur-kalk yürüyüş** | Yön tutarsızlaşır; nesne çizgi üstünde gidip gelirse yanlış sayılır* |
| **Seyrek örnekleme** | Nesne kadraja girip iki örnek arasında çıkarsa hiç görülmez |

\* Aynı iz **yalnızca bir kez** sayılır — bu koruma var.

> Çekim rehberimiz "2-3 adımda bir yarım saniye durun" diyor. Bu, çizgi
> sayımıyla çelişir. Drone gibi düzenli ilerleyen kaynaklarda kullanın.

### Sabit/hareketli ayrımı ÖLÇÜLÜR

Kullanıcıya "video mu sabit mi" diye sorulmaz. İzlerin kayması ölçülür:
saniyede kadrajın **%4'ünden** fazla kayma varsa kamera ilerliyordur.

```
SABİT video : 17 kutu → 3 benzersiz  | kamera: sabit
KAYAN video : 14 kutu → 5 benzersiz  | kamera: hareketli, kayma 0.109
```

`sayim_onerisi()` hangi sayımın geçerli olduğunu söyler:

| Kamera | Önerilen | Neden |
|---|---|---|
| sabit | benzersiz iz | çizgi 0 verir |
| hareketli | çizgi (açıksa) | iz sayımı kopmalardan şişebilir |

Çizgi sayacı **varsayılan kapalıdır**. Canlı akış sayfasından açılır:
yön (dikey/yatay) ve konum (%10-%90) seçilir, çizgi tuvalde kesikli sarı
olarak görünür. Sabit çekimde arayüz uyarır:

```
📏 çizgiden geçen: 0 (kamera sabit — çizgi sayımı bu çekimde işe yaramaz)
```

---

## 6. Canlı akış

### Nasıl çalışır

- Kare tarayıcıda küçültülür, WebSocket ile gönderilir
- Sunucu modeli çalıştırır, **yalnızca kutu koordinatlarını** döner
- Çizim tarayıcıda yapılır — ağdan resim taşınmaz
- **Geri basınç:** Bir sonraki kare, öncekinin sonucu gelmeden
  gönderilmez. Sunucu yavaşsa kuyruk birikmez, kare/sn kendiliğinden düşer
- WebSocket engelliyse aynı kare REST'e POST edilir (davranış aynı)

### Oturum boyunca benzersiz sayım

Canlı akışta kareler saniyede birkaç kez gelir; aynı meyve onlarca karede
görünür. Kutuları toplamak "300 çilek gördüm" gibi anlamsız bir sayı
üretirdi.

```
3 tespit (ekranda) · 🍓 turda 12 farklı nesne · 💾 4/50 kayıt
```

> Kareler **düzensiz** aralıklarla gelir (ağ gecikmesi, telefon gücü).
> Bu yüzden takipçiye kare numarası değil **gerçek zaman** verilir
> (`ekle_zamanli`). Arama penceresi süreye bağlı olduğu için bu şarttır.

### Kayıt modları

| Mod | Ne kaydeder | Ne zaman |
|---|---|---|
| `akilli` | Yalnızca kararlı bulgular | Günlük kullanım, depolama dostu |
| `tespitli` | Tespit içeren her kare | Sera turunun dökümü |
| `hepsi` | Tespit olmayanlar dahil | **Eğitim verisi toplamanın en hızlı yolu** |

`tespitli`/`hepsi` modlarında iki koruma var: kareler arası en az aralık ve
oturum başına azami kayıt. İkisi olmasa 10 dakikalık tur binlerce kayıt açar.

### HTTPS şartı

Tarayıcı kamerayı **yalnızca güvenli bağlamda** açar. Bu yüzden:

- Bilgisayarda: `http://localhost:8000/canli` yeterli (localhost güvenli sayılır)
- Telefonda: `https://<MAKINE-ADI>.local:8443/canli` gerekir

Adres ve QR kodu için `/baglan` sayfasını kullanın. **Makine adı IP gibi
değişmez** — router yeni IP verse bile çalışır.

---

## 7. Drone

Drone ayrı bir kod yolu **değildir**: ürettiği görüntü fotoğraf ya da
videodur, o akışlardan geçer.

### Koşullar

| Konu | Kural |
|---|---|
| **Yükseklik** | Lezyon kaynak görüntüde **en az 16 piksel** olmalı. `imgsz_oner.py` ile ölçün — "sorun veride" uyarısı çıkıyorsa alçalın |
| **Hız** | Nesne **en az iki örnek karede** görünmeli. 0,5 sn aralıkta: nesne kadrajda ≥1 sn kalmalı |
| **Çizgi sayımı** | Düzenli transektte **önerilir**. Uçuş yönüne dik çizgi seçin |
| **Takip hızı** | Varsayılan %35/sn yürüyüş içindir. Hızlı uçuşta `Takipci(hiz=…)` artırılmalı |
| **Titreşim** | Bulanık kare atlanır; gimbal yoksa atlama oranı yüksek olur |

### Drone için önerilen akış

1. Düşük irtifada bir **test videosu** çekin
2. `imgsz_oner.py` ile nesne boyutunu ölçün
3. Yeterliyse transekt uçuşu yapın, **çizgi sayımını açın**
4. Sonuçta `sayim_onerisi()`'nin "hareketli" dediğini doğrulayın

> Drone akışı canlı olarak bağlanacaksa `app/takip.py` içindeki
> `Takipci`/`CizgiSayaci` doğrudan kullanılabilir — yeni sayım kodu
> yazmak gerekmez, sadece kare kaynağı bağlanır.

---

## 8. Hangi sayıya güvenmeli?

```
Tek nesne sayısı mı istiyorsun?
├── Fotoğraf çektiysen        → tespit sayısı doğrudur
├── Video/canlı ve kamera SABİT → benzersiz_sayi
├── Video/canlı ve kamera HAREKETLİ
│   ├── çizgi açıksa          → çizgiden geçen
│   └── çizgi kapalıysa       → benzersiz_sayi (kopmalardan şişebilir)
└── Kesin sayım şartsa        → tek fotoğraf çekin
```

---

## İlgili belgeler

- [MIMARI.md](MIMARI.md) — boru hattı ve model kütüğü
- [HATA-YONETIMI.md](HATA-YONETIMI.md) § 4 — çıkarım hataları
- [VERI-ALMA.md](VERI-ALMA.md) — veri seti hazırlama
