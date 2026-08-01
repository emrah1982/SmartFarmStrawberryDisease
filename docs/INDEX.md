# Belgeler

Çilek hastalık tespit sistemi — hiyerarşik çok modelli görüntü analizi,
FastAPI arayüzü, yerel ağda çalışır.

---

## Ne arıyorsun?

| Soru | Belge |
|---|---|
| Sistem nasıl çalışıyor? Neden böyle kurulmuş? | [MIMARI.md](MIMARI.md) |
| Bir hata aldım / aynı hata tekrar ediyor | [HATA-YONETIMI.md](HATA-YONETIMI.md) |
| Roboflow'dan/Kaggle'dan veri indirdim, nasıl eklerim? | [VERI-ALMA.md](VERI-ALMA.md) |
| Video/drone/canlı akışta sayım neden farklı? | [GORUNTU-KAYNAKLARI.md](GORUNTU-KAYNAKLARI.md) |
| Model nasıl eğitilir ve kurulur? | [EGITIM.md](EGITIM.md) |
| Yeni bir bitki (domates vb.) eklemek istiyorum | [COK_BITKILI_YAPI.md](COK_BITKILI_YAPI.md) |
| Kod yazarken neye dikkat etmeliyim? | [GELISTIRME-KURALLARI.md](GELISTIRME-KURALLARI.md) |
| Sırada ne var? | [YOL-HARITASI.md](YOL-HARITASI.md) |

---

## Bugünkü durum

| Alan | Durum |
|---|---|
| Mimari | Hiyerarşik: organ → ROI → uzman model |
| Modeller | 5 model eğitildi ve kuruldu; `zararli` bekliyor (veri yok) |
| Ürün kapsamı | Çilek. Çok bitkili iskelet hazır |
| Arayüz | FastAPI + SQLite, yerel ağ; telefon/webcam/IP kamera/canlı akış |
| Dağıtım | Docker (http:8000 + https:8443 tek süreçte) |
| Test | 528 test |

### Model durumu

| Model | mAP50-95 | Durum |
|---|---|---|
| `organ` | 0,8326 | ✅ kurulu |
| `olgunluk` | 0,6157 | ✅ kurulu |
| `bocek_teshis` | 0,5712 | ✅ kurulu (ayrı akış) |
| `meyve_hastalik` | 0,5559 | ✅ kurulu |
| `yaprak_hastalik` | 0,4000 | ✅ kurulu — veri kalitesi sınırlı |
| `zararli` | — | ⏳ saha verisi toplanacak |

---

## Hızlı başlangıç

```bash
# Çalıştır
docker compose up -d

# Bilgisayardan
http://localhost:8000

# Telefondan (adres ve QR için)
http://localhost:8000/baglan
```

Kamera kullanmak için telefonda **https** gerekir; `/baglan` sayfası
makine adıyla kalıcı adresi verir.

### Sık kullanılan komutlar

```bash
python scripts/model_kur.py --listele                 # hangi modeller kurulu?
python scripts/egitim_izle.py --bekle                 # eğitim izle
python scripts/imgsz_oner.py --hepsi                  # çözünürlük ölç
python scripts/harici_paket_duzelt.py <zip> --kuru    # yeni veri setini denetle
python -m pytest tests/ -q                            # testler
```

---

## Belge yazma kuralı

Bu belgelerde **karar + gerekçe** birlikte yazılır. "Neden böyle
yapmıştık" sorusuna dönüp bakabilmek ve aynı hataya geri dönmemek için.

Yeni bir hata bulunca [HATA-YONETIMI.md](HATA-YONETIMI.md) dosyasına
**belirti / sebep / koruma** üçlüsüyle ekleyin. Testsiz koruma, koruma
değildir.
