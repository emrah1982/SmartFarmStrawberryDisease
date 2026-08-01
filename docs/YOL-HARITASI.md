# Yol Haritası ve Notlar

Projenin bugünkü durumu ve sonraki adımlar. Kararların **gerekçesi** de yazılıdır;
"neden böyle yapmıştık" sorusuna dönüp bakabilmek için.

---

## Bugün nerede

| Alan | Durum |
|---|---|
| Mimari | **Hiyerarşik çok modelli**: organ → ROI → uzman model ([MIMARI.md](MIMARI.md)) |
| Modeller | 5 model eğitildi ve kuruldu; `zararli` saha verisi bekliyor |
| Veri | Model başına ayrı dataset; sızıntı denetimi ve etiket temizliği araçlı |
| Eğitim | Colab Pro+ (A100); epoch/imgsz/önbellek **ölçülerek** seçiliyor |
| Sayım | Kareler arası takip (`app/takip.py`) — video/drone/canlıda benzersiz sayım |
| Arayüz | FastAPI + SQLite; telefon/webcam/IP kamera/canlı akış, QR ile bağlantı |
| İşletme yapısı | Üretici → Sera → Kamera → Analiz |
| Ürün kapsamı | Çilek; çok bitkili iskelet hazır ([COK_BITKILI_YAPI.md](COK_BITKILI_YAPI.md)) |
| Dağıtım | Docker, tek süreçte http:8000 + https:8443 |
| Test | 528 test |

### Bu turda tamamlananlar

- Hiyerarşik mimariye geçiş tamamlandı; miras (tek) model devre dışı
- Model başına çıkarım `imgsz`'i ölçülerek belirlendi → bir sera fotoğrafında
  1 tespit → 4 tespit
- Sonuçlar **organa göre** gruplanıyor; tedavi önerileri organa özel
- Böcek teşhis modülü (ayrı akış, kapalı küme uyarılı)
- Video/drone/canlı için kareler arası takip + isteğe bağlı çizgi sayımı
- `/baglan` sayfası: mDNS makine adı + QR (IP değişse de çalışır)

---

## 1. Kullanıcı girişi ve müşteri izolasyonu ⭐ öncelikli

**Durum:** Giriş sistemi yok. Yerel ağdaki herkes tüm üreticilerin verisini görür.
Tek işletme için sorun değil; **birden fazla müşteriye satış** düşünülüyorsa gerekli.

**Altyapı hazır:**

- `app/database.py` → `Kullanici` tablosu tanımlı (`kullanici_adi`, `parola_hash`,
  `rol`, `uretici_id`). Tablo baştan oluşturulur ki geçişte veritabanı göçü gerekmesin.
- `app/yetki.py` → **tek geçiş noktası**. Rotalar veriye doğrudan değil buradaki
  yardımcılar (`analiz_sorgusu`, `gorunur_seralar`, `gorunur_kameralar`,
  `erisebilir_mi`) üzerinden erişir.

**Yapılacak (yalnızca `yetki.py` + giriş sayfası):**

1. `aktif_kullanici()` oturumdan kullanıcıyı okusun (çerez/JWT)
2. Giriş/çıkış sayfaları, parola doğrulama (`passlib[bcrypt]`)
3. Rotalarda başka değişiklik gerekmez — izolasyon otomatik uygulanır

**Neden şimdiden bu yapı:** İzolasyonu sonradan eklemek her sorguyu bulup filtre
eklemek demektir; **bir tanesini atlamak bir müşterinin verisini başkasına gösterir.**
Tek noktadan geçiş bu riski ortadan kaldırır.

---

## 2. HTTPS ✅ tamamlandı

Telefonda tarayıcı içi kamera **güvenli bağlam** ister. Çözüldü:

- Tek süreçte iki dinleyici: `http:8000` + `https:8443`
- `scripts/https_sertifika.py` kendinden imzalı sertifika üretir; makine
  adını (`<ad>.local`) ve mevcut IP'leri **her zaman** kapsar
- `/baglan` sayfası adresleri QR koduyla verir, sertifika kapsamını denetler

**Kalan:** internete açılacaksa gerçek sertifika (Let's Encrypt) + madde 1'deki
kimlik doğrulama şart.

---

## 3. Bildirim ve otomatik izleme

Bugün kamera **isteğe bağlı** çekim yapıyor. Sıradaki seçenekler:

- **Zamanlanmış çekim:** her 30 dk otomatik analiz → trend takibi
- **Eşik alarmı:** belirli hastalık > N tespit → e-posta/Telegram bildirimi
- **Sürekli izleme:** canlı akış; en çok kaynak tüketen seçenek

Not: Sürekli izleme GPU olmadan pahalıdır; önce zamanlanmış çekim önerilir.

---

## 4. Model iyileştirme döngüsü

Kurulu ve çalışıyor: düşük güvenli kayıtlar → inceleme kuyruğu → **tarayıcıda
etiketleme** (Roboflow gerekmez) → eğitim formatında dışa aktarım (`images/`,
`labels/`, `data.yaml`) → `merge_datasets.py` → yeniden eğitim.

**Ölçülen durum:** `leaf_disease` 0,40 ile en zayıf halka. Sebep bulundu —
en küçük %10 kutu **kaynak görüntüde zaten 3 piksel**. Bu imgsz sorunu değil
veri sorunudur: ya hatalı etiket, ya çok uzaktan çekim, ya küçültülmüş
dışa aktarım. Çözüm augmentasyon değil **etiket temizliği + gerçek saha verisi**.

**Sıradaki:**
- `leaf_disease` etiketlerini `etiket_temizle.py` ile denetle, 3 piksellik
  kutuları ayıkla, yeniden eğit
- Ölçülen `imgsz` değerleriyle yeniden eğitim (10 kat hız kazancı)
- Dondurulmuş test seti — model sürümleri adil karşılaştırılsın
  (`model_karsilastir.py` hazır)

---

## 5. Ticari/hukuki

- **Lisans:** YOLO26/Ultralytics **AGPL-3.0**. Kapalı kaynak ticari üründe ya kod
  açılır ya Ultralytics Enterprise lisansı alınır. Alternatif Apache-2.0 modeller:
  RF-DETR, YOLOX, D-FINE.
- **Tedavi önerileri:** `configs/urunler/<urun>/tedavi_onerileri.yaml` içinde
  **ilaç adı/dozu yoktur**
  — bilinçli tercih. Ruhsatlı ilaçlar ülkeye/ürüne/döneme göre değişir; yanlış
  tavsiye yasal sorumluluk ve ürün kaybı doğurur. Metinler kültürel önlem +
  "uzmana danışın" yönlendirmesidir.
- **Veri sahipliği:** Müşteri serasından toplanan görüntülerin eğitimde kullanımı
  için sözleşmede açık madde bulunmalı.

---

## 6. Dağıtım seçenekleri

| Senaryo | Yaklaşım |
|---|---|
| Tek işletme, yerel | Bugünkü kurulum (Docker, yerel ağ) |
| Birden çok müşteri | Bulut sunucu + HTTPS + kullanıcı girişi (madde 1-2) |
| İnternetsiz sera | Jetson/mini PC + TensorRT export |
| Mobil uygulama | Mevcut API kullanılır; `detector.py` arayüzden bağımsız yazıldı |

---

## Küçük notlar

- `storage/` ve `models/*.pt` git'e girmez; dataset (640 MB) da öyle — Drive üzerinden taşınır.
- Windows'ta 260 karakter yol sınırı: Roboflow'un uzun dosya adları sorun çıkardı,
  28 dosya kısaltıldı. Yeni veri eklerken kontrol edin (README'de komut var).
- Analiz kayıtlarında görüntüler **dosya** olarak tutulur, veritabanında yalnızca yol
  vardır; BLOB kullanmak yedeklemeyi ve servisi ağırlaştırırdı.
- Böcek yedek teşhisi yalnızca **fotoğrafta** çalışır. Videoda hangi karenin
  "böcek fotoğrafı" sayılacağı ayrı bir problem — bilinçli erteleme.
- Çizgi sayımı arayüzden açılabilir ama **varsayılan kapalı**: sabit çekimde
  0 verir, kendiliğinden açılması yanıltıcı olurdu.

---

## İlgili belgeler

- [MIMARI.md](MIMARI.md) · [HATA-YONETIMI.md](HATA-YONETIMI.md) ·
  [VERI-ALMA.md](VERI-ALMA.md) · [GORUNTU-KAYNAKLARI.md](GORUNTU-KAYNAKLARI.md) ·
  [EGITIM.md](EGITIM.md)
