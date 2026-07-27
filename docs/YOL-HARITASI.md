# Yol Haritası ve Notlar

Projenin bugünkü durumu ve sonraki adımlar. Kararların **gerekçesi** de yazılıdır;
"neden böyle yapmıştık" sorusuna dönüp bakabilmek için.

---

## Bugün nerede

| Alan | Durum |
|---|---|
| Model | YOLO26 (`yolo26s`), 10 sınıf: 7 hastalık + 3 olgunluk |
| Veri | 4 kaynak birleşik, ~9.300 train görüntüsü, sınıf hedefli augmentasyon |
| Eğitim | Colab Pro+ (A100), GPU'ya göre otomatik ayar, kaldığı yerden devam |
| Arayüz | FastAPI + SQLite, yerel ağ; telefon/webcam/IP kamera |
| İşletme yapısı | Üretici → Sera → Kamera → Analiz |
| Dağıtım | Docker (CPU imajı 697 MB), GPU seçeneği hazır |
| Sürekli iyileştirme | İnceleme kuyruğu, tarayıcıda etiketleme, tek birikimli eğitim havuzu |
| Konum/yaygınlık | Modül (`app/moduller/konum`): EXIF GPS, kamera konumu, elle blok/sıra, ısı haritası |

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

## 2. HTTPS (telefonda tarayıcı içi kamera için)

Telefondan `http://192.168.x.x` ile bağlanıldığında tarayıcı `getUserMedia`'yı
**engeller** (güvenli bağlam şartı). Bu yüzden telefonda cihazın kamera uygulaması
kullanılır — pratikte yeterlidir.

Tarayıcı içi kamera/canlı önizleme istenirse: `mkcert` ile yerel ağa güvenilir
sertifika üretilip uvicorn `--ssl-keyfile/--ssl-certfile` ile başlatılır.
İnternete açılacaksa zaten HTTPS + kimlik doğrulama şart.

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

**Sıradaki ölçüm:** eğitim bitince sınıf bazlı recall'a bakılacak. Az örnekli
sınıflar (Anthracnose 326 kutu, Powdery Mildew Fruit 590) zayıf kalırsa çözüm
augmentasyon değil **gerçek saha verisi**.

Ayrıca: dondurulmuş bir test seti tutulmalı ki model sürümleri adil karşılaştırılsın.

---

## 5. Ticari/hukuki

- **Lisans:** YOLO26/Ultralytics **AGPL-3.0**. Kapalı kaynak ticari üründe ya kod
  açılır ya Ultralytics Enterprise lisansı alınır. Alternatif Apache-2.0 modeller:
  RF-DETR, YOLOX, D-FINE.
- **Tedavi önerileri:** `configs/tedavi_onerileri.yaml` içinde **ilaç adı/dozu yoktur**
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
