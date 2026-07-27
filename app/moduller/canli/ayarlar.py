"""Canlı akış parametreleri — tek yerden ayarlanır, ortam değişkeniyle ezilir."""

import os

# --- Hız ---------------------------------------------------------------------
# Canlıda çözünürlük düşürülür: 1024 yerine 640 ile tespit birkaç kat hızlanır,
# akış izlenebilir kalır. Yakın çekimde (30-60 cm) doğruluk kaybı azdır; uzak
# çekimde tek kare analizini (Ayrıntılı analiz) kullanın.
CANLI_IMGSZ = int(os.environ.get('CANLI_IMGSZ', '640'))

# Tarayıcının göndereceği karenin genişliği (piksel). Ağ üzerinden taşınan
# veriyi belirler; Wi-Fi zayıfsa düşürün.
GONDERIM_GENISLIK = int(os.environ.get('CANLI_GONDERIM_GENISLIK', '640'))
GONDERIM_KALITE = float(os.environ.get('CANLI_GONDERIM_KALITE', '0.6'))   # JPEG

# İki kare arasındaki en az bekleme (ms). Sunucu çok hızlıysa CPU'yu boş yere
# doldurmamak için. Asıl hız zaten geri basınçla belirlenir (bkz. akis.js).
EN_AZ_ARALIK_MS = int(os.environ.get('CANLI_EN_AZ_ARALIK_MS', '120'))

# --- Otomatik kayıt ----------------------------------------------------------
# Canlı akışta her kareyi kaydetmek anlamsız (saniyede birkaç kayıt). Bir
# bulgu ancak KARARLI hale gelince kaydedilir: aynı sınıf üst üste N karede,
# yeterli güvenle görülürse. Böylece tek karelik yanlış tespitler kayda geçmez.
OTOMATIK_KAYIT = os.environ.get('CANLI_OTOMATIK_KAYIT', '1') not in ('0', 'false', 'False')
KARARLILIK_KARE = int(os.environ.get('CANLI_KARARLILIK_KARE', '3'))
KAYIT_GUVEN = float(os.environ.get('CANLI_KAYIT_GUVEN', '0.60'))

# Aynı bulgu için bekleme (saniye): sera içinde aynı lekeye 10 saniye bakınca
# onlarca kayıt oluşmasın.
BEKLEME_SN = float(os.environ.get('CANLI_BEKLEME_SN', '20'))

# Bu değerin altındaki keskinlikte kare modele verilmez (hareket bulanıklığı).
# 0 = kapalı. Canlıda eşik, video işlemedekinden gevşektir: yürürken çekimde
# çok sert eşik akışı tamamen durdurur.
BULANIKLIK_ESIGI = float(os.environ.get('CANLI_BULANIKLIK_ESIGI', '25'))
