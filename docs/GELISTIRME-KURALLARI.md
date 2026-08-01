# Geliştirme Kuralları

> Bu kurallar teorik değil: her biri bu projede yaşanan bir sorundan
> çıktı. Gerekçeleri yazılıdır.

---

## 1. Bağımlılık yönü

```
main.py (HTTP, şablon)  →  pipeline / detector / moduller
                        →  modeller / takip
                        →  siniflar · urunler · tedavi · sonuc_ozeti · ag · cizim
```

**Ok yönü hep aşağı.** Alt katman üstünü bilmez.

Somut kural: şu modüller `fastapi`, `sqlalchemy`, `jinja2`, `app.main`
ithal **edemez** — `takip.py`, `sonuc_ozeti.py`, `tedavi.py`, `ag.py`,
`moduller/bocek/servis.py`.

**Neden:** `takip.py`'yi başka projeye kopyalayabilmek için değil sadece —
asıl fayda test edilebilirlik. HTTP olmadan 37 test çalışıyor.

---

## 2. Modül mü, çekirdek mi?

**Modül yap** (`app/moduller/<ad>/`) eğer:
- Kapatılabilir bir yetenekse
- Kendi tablosu/sayfası varsa
- Çekirdek onsuz çalışabiliyorsa

**Çekirdeğe koy** (`app/<ad>.py`) eğer:
- Analiz akışının parçasıysa
- Birden çok modül kullanacaksa

Modül kendi tablosunu açar, çekirdek şemaya sütun **eklemez**. Böcek
kayıtları `Analiz` tablosuna yazılsaydı hastalık istatistikleri ve
yaygınlık haritası bozulurdu.

---

## 3. Yapılandırma koda gömülmez

Hangi model ne zaman çalışır, hangi sınıf hangi eşikle kabul edilir, hangi
tedavi önerilir — hepsi `configs/urunler/<urun>/` altındadır.

Yeni bir zararlı eklemek **kod değiştirmez**.

---

## 4. Sessiz hata yasak

Bu projedeki hataların çoğu çökmedi; yanlış sonuç üretti ve görünmedi.

| Yasak | Yerine |
|---|---|
| Boş sonuç dönüp susmak | Logla ve kullanıcıya söyle |
| Anlamı belirsiz sayı vermek | Ne olduğunu yaz ("N kutu, M ayrı nesne") |
| Tanımsız CSS değişkenine yedek vermek | Değişkeni tanımla — yedek hatayı gizler |
| `except: pass` | En az `logger.warning` |

Gerçek örnek: `background: var(--zemin2, #fafafa)` — `--zemin2` hiç
tanımlı değildi, yedek devreye girdi, koyu temada metin okunmaz oldu ve
hiçbir hata çıkmadı.

---

## 5. Ölç, varsayma

| Varsayım | Ölçüm |
|---|---|
| "Küçük lezyonlar var, imgsz 1024 olsun" | Hiçbir dataset 640'tan büyük değil |
| "Sınıf dengesizliği sorun" | 15:1 oranda en az veri en iyi sonucu verdi |
| "Model kötü" | Çıkarım imgsz'i yanlıştı: 1 tespit → 4 tespit |
| "Eğitim öldü" | Drive kopyası koşuyu ikiye bölmüş, 200/200 bitmişti |

Ölçüm araçları: `imgsz_oner.py`, `epoch_oner.py`, `egitim_izle.py`,
`model_karsilastir.py`, `harici_paket_duzelt.py --kuru`

---

## 6. Test yazma

**Testin ne olduğunu değil NEDEN olduğunu yaz.** Docstring'de gerçek
olayı anlat:

```python
def test_uzman_model_baska_modelin_kosusunu_gormez(self):
    """ASIL HATA: organ eğitimi, birleşik modelin koşusunu kendi sanıyordu."""
```

Bir hata ikinci kez yaşandıysa **mutlaka** test ekleyin. Bu projedeki
testlerin çoğu böyle doğdu.

Test, gerçek kodu çalıştırmalı — kopyasını değil. Notebook testleri hücre
kodunu dosyadan okuyup `exec` eder; ayrı bir simülasyon yazmak, simülasyonu
test etmek olurdu.

---

## 7. Yorumlar

Yorum **neden**i anlatır, neyi değil. Kod ne yaptığını zaten söyler.

```python
# Organ eşiği cömert: yanlış ROI zararsızdır (uzman model bir şey bulamaz),
# ama KAÇIRILAN organ tüm zinciri keser.
esik: 0.20
```

Bir karar ölçümle alındıysa **sayıyı yaz**:

```python
# ÖLÇÜLDÜ (sera fotoğrafı, 640x640 kaynak):
#   imgsz 1024 → 2 meyve, güvenler 0.738 / 0.669 / 0.339
#   imgsz  640 → 3 meyve, güvenler 0.841 / 0.793 / 0.608
```

---

## 8. Dosya boyutu

300-500 satırı geçen dosyayı bölün. `main.py` 1062 satır — bölünmeyi
bekliyor; yeni özellik oraya değil modüle veya bileşene gitmeli.

---

## 9. Sürüm ve dağıtım

- Commit mesajı **ne + neden** içerir; ölçüm varsa sayıyı yazın
- `app/` Docker imajına gömülüdür — değiştirince `docker compose build web`
  şart, sadece `restart` yetmez
- `configs/urunler`, `models`, `storage`, `certs` bağlanmıştır; onlar
  yeniden derleme istemez
- Testler geçmeden push etmeyin (bilinen kararsız `test_get_metrics` hariç)

---

## 10. Windows/PowerShell tuzakları

Bu projede tekrar tekrar zaman kaybettiren şeyler:

- PowerShell here-string içinde backtick kaçış karakteridir; `` `t `` sekmeye
  dönüşür. Çok satırlı metin için dosyaya yazıp `-F` ile verin
- `Set-Content` ANSI kod sayfası kullanır → Türkçe bozulur.
  `[System.IO.File]::WriteAllText` veya `Out-File -Encoding utf8` kullanın
- `cv2.imread` Türkçe karakterli yolda çalışmaz → `cv2.imdecode` + `np.fromfile`
- Alt süreç çıktısı cp1254 ile çözülür → `encoding='utf-8'` ve
  `PYTHONIOENCODING=utf-8` verin
