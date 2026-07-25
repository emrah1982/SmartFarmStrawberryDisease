# Mimari Tasarım (Katmanlı Mimari)

Bu doküman, Strawberry Vision projesinin katmanlı mimarisini, katmanların sorumluluklarını ve aralarındaki bağımlılık kurallarını açıklar.

## Katmanlar ve Sorumluluklar

- Presentation Layer
  - Çıktı üretimi ve görselleştirme (OpenCV/Matplotlib).
  - UI/Notebook etkileşimi.
  - İş kuralı içermez, yalnızca sunum/çizim.

- Application Layer
  - Uygulama akışı/pipeline koordinasyonu.
  - Detection → Classification → Tracking → Counting sırasını yönetir.
  - Katmanlar arası orkestrasyon ve bağımlılıkların enjekte edilmesi (DI).
  - Dış dünya ile Domain arasındaki sınırı korur.

- Domain Layer
  - Saf iş kuralları ve model (Entities, Value Objects, Services).
  - `Detection`, `Strawberry` ve (opsiyonel) `Ripeness` gibi varlıklar.
  - `TrackingService`, `CountingService` gibi saf mantık.
  - Harici kütüphanelere bağımlı olmamalıdır.

- Infrastructure Layer
  - Model yükleme ve çalıştırma (YOLO).
  - Veri kaynakları (resim, video, kamera).
  - Dosya sistemi, üçüncü parti entegrasyonlar.
  - Domain veya Application tarafından tanımlı arayüzlerin implementasyonları.

## Bağımlılık Kuralları

- Presentation → Application/Domain (sadece okuma, iş kuralı yok).
- Application → Domain ve Infrastructure (uygulama akışını kurar).
- Domain → Hiçbir katmana (bağımsızdır).
- Infrastructure → Domain (entity kullanımı) ama Domain Infrastructure'a bağlı olmaz.

## Veri Akışı

1. Source (Infrastructure) kare/görüntü üretir.
2. Detector (Infrastructure) ile aday çilekler bulunur.
3. Hastalık sınıf etiketleri modelden alınır; (opsiyonel) olgunluk analizi yapılabilir.
4. TrackingService (Domain) ID ataması yapar.
5. CountingService (Domain) sayım yapar.
6. Visualizer (Presentation) sonuçları çizer.
7. Application Layer sonuçları özetler/raporlar.

## Genişletilebilirlik Noktaları

- Detector stratejisi: YOLO varyantı, farklı çerçeveler.
- Ripeness sınıflandırma: Renk heuristiği → CNN/transformer.
- Kaynaklar: görüntü/video/kamera/RTSP.
- Çıktı: çizim, dosyaya kaydetme, metrik raporlama, REST girişi/çıkışı.

## Test Edilebilirlik

- Domain saf Python: birim test kolay ve hızlı.
- Application: bağımlılık enjeksiyonu ile sahte (fake) detector/visualizer kullanılabilir.
- Infrastructure: entegrasyon testleri ve dummy fallback.

## Hata Yönetimi

- Infrastructure bileşenlerinde try/except ile güvenli düşüş (graceful fallback).
- Application seviyesi: adımlar arası hata izolasyonu ve özet rapor.
- Log ve metrikler ileride eklenecek şekilde arabirimler üzerinden tasarlanabilir.
