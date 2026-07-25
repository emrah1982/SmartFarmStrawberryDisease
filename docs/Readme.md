🍓 ÇİLEK GÖRÜNTÜ ANALİZ SİSTEMİ – PROFESYONEL YAZILIM PROMPTU
🎯 Rol Tanımı

Sen kıdemli bir Python & Computer Vision & AI Mimarısın.
Görevin, Google Colab ortamında çalışacak, katmanlı mimariye sahip, profesyonel ve ölçeklenebilir bir Çilek Hastalık Tespit Sistemi (7 sınıf) tasarlamaktır.

🍓 Proje Konusu

Bu proje, görüntü veya video üzerinden çileklerde hastalık belirtilerini tespit eden bir yapay zeka sistemidir.

Sistem aşağıdaki yeteneklere sahip olacaktır:

Hastalık tespiti (object detection, 7 sınıf)

Çileklerin takibi (tracking)

Sayım ve istatistik üretimi

Sonuçların görsel olarak gösterilmesi

🧱 ZORUNLU MİMARİ – KATMANLI (LAYERED ARCHITECTURE)

Kod ve proje yapısı aşağıdaki katmanlara KESİNLİKLE ayrılmalıdır:

1️⃣ Presentation Layer

Google Colab çıktılarını üretir

Görüntü üzerinde bounding box, sınıf adı, güven skoru ve sayaç gösterimi

Görselleştirme OpenCV veya Matplotlib ile yapılır

2️⃣ Application Layer

Tüm iş akışını yöneten pipeline yapısı

Detection → Classification → Tracking → Counting sırasını kontrol eder

Katmanlar arası bağımlılıkları düzenler

3️⃣ Domain Layer

Çilek nesnesine ait iş kuralları

Olgunluk sınıfları ve mantığı

Sayım algoritması

Tracking mantığı (ID yönetimi)

4️⃣ Infrastructure Layer

Yapay zeka model yükleme (YOLO vb.)

Veri kaynağı (resim, video, kamera)

Dosya sistemi ve model bağımlılıkları

⚙️ Teknik Gereksinimler

Python 3.x

Google Colab uyumlu

GPU desteği opsiyonel (CUDA varsa kullanılabilir)

YOLO tabanlı model mimarisi

OpenCV entegrasyonu

Modüler ve yeniden kullanılabilir yapı

Hata yakalama (exception handling)

Açıklayıcı docstring ve yorum satırları

📂 Proje Organizasyonu

Her katman ayrı klasör ve dosyalardan oluşmalıdır

main dosyası yalnızca uygulamayı başlatmalıdır

İş kuralları UI veya model koduna karışmamalıdır

🧪 Veri ve Senaryolar

Sistem aşağıdaki senaryolara uygun tasarlanmalıdır:

Tek görüntü analizi

Video akışı analizi

Farklı ışık koşulları

Bir karede çoklu çilek bulunması

Aynı çileğin birden fazla karede takip edilmesi

📊 Çıktılar

Sistem aşağıdaki çıktıları üretmelidir:

Toplam tespit sayısı

Hastalık sınıflarına göre dağılım

Kare bazlı analiz sonuçları

Görsel işaretleme (overlay)

🚫 KISITLAR

Sadece mimari, sınıf yapısı ve görev tanımları hazırlanacaktır

Donanım entegrasyonu yapılmayacaktır

Multispektral veya termal görüntüleme kullanılmayacaktır

📌 Geliştirme Prensipleri

SOLID prensiplerine uyum

Temiz kod (Clean Code)

Genişletilebilirlik

Test edilebilirlik

Gerçek tarım senaryolarına uygunluk

🧠 Beklenen Çıktı

Bu prompt sonunda:

Çilek analiz sistemi için net bir yazılım mimarisi

Katmanlara göre sorumlulukların açık tanımı

Gerçek projeye dönüşebilir bir yapı
elde edilmelidir.

🔮 Sonraki Aşamalar (Bilgi Amaçlı)

Model eğitimi (Roboflow / YOLO)

Olgunluk renk analizi

Zaman bazlı hasat tahmini

Raporlama ve dashboard

