"""Uygulama ayarları — ortam değişkenleriyle geçersiz kılınabilir."""

import os
from pathlib import Path

# Proje kökü (app/ dizininin bir üstü)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Model -----------------------------------------------------------------
# Eğitilmiş model yolu. Drive'dan indirdiğiniz best.pt'yi buraya koyun veya
# MODEL_PATH ortam değişkeniyle gösterin.
MODEL_PATH = os.environ.get('MODEL_PATH', str(BASE_DIR / 'models' / 'best.pt'))

CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', '0.25'))
IMGSZ = int(os.environ.get('IMGSZ', '1024'))

# Bu güvenin altındaki tespitler "incelenecek" kuyruğuna düşer.
# Sürekli iyileştirme döngüsünü besler (bkz. README).
REVIEW_THRESHOLD = float(os.environ.get('REVIEW_THRESHOLD', '0.55'))

# Laplacian varyansı bu değerin altındaki kareler bulanık sayılır ve videoda
# atlanır. Yürürken çekimde hareket bulanıklığı yaygındır; bulanık kareyi
# modele vermek yanlış/eksik tespit üretir.
BULANIKLIK_ESIGI = float(os.environ.get('BULANIKLIK_ESIGI', '60'))

# --- Ayrıntılı analiz (büyük saha fotoğrafları) ------------------------------
# Tek ölçekli tahmin, çekim ölçeği eğitim verisinden farklıysa kararsızdır.
# Bu ölçeklerde tahmin yapılıp sonuçlar birleştirilir.
DETAYLI_OLCEKLER = [int(x) for x in os.environ.get('DETAYLI_OLCEKLER', '640,1024').split(',')]
DILIM_ESIGI = int(os.environ.get('DILIM_ESIGI', '1600'))      # bu boyutu aşan görüntü dilimlenir
DILIM_BOYUTU = int(os.environ.get('DILIM_BOYUTU', '1024'))
DILIM_ORTUSME = float(os.environ.get('DILIM_ORTUSME', '0.2'))
NMS_IOU = float(os.environ.get('NMS_IOU', '0.5'))             # örtüşen kutuları birleştirme eşiği

# --- Depolama --------------------------------------------------------------
STORAGE_DIR = Path(os.environ.get('STORAGE_DIR', str(BASE_DIR / 'storage')))
UPLOAD_DIR = STORAGE_DIR / 'uploads'      # yüklenen orijinaller
RESULT_DIR = STORAGE_DIR / 'results'      # kutulanmış çıktılar
# Dış araçlara (Roboflow vb.) gönderilecek paket. Tarihli anlık görüntü
# BİRİKTİRİLMEZ: her aktarımda temizlenip güncel veritabanından yeniden
# yazılır, böylece içerik veritabanıyla her zaman tutarlı kalır.
INCELEME_DIR = STORAGE_DIR / 'inceleme_paketi'

# Elle etiketlenen kayıtların BİRİKTİĞİ tek klasör.
# Her dışa aktarımda yeni klasör açmak yerine buraya eklenir: eğitim öncesinde
# tek bir yol verilir, klasörleri elle toplamak gerekmez.
EGITIM_DIR = STORAGE_DIR / 'egitim_verisi'

# --- Veritabanı ------------------------------------------------------------
# SQLite tek dosyadır, kurulum gerektirmez. Büyürseniz DATABASE_URL'i
# PostgreSQL'e çevirmek yeterli (kod değişikliği gerekmez).
DATABASE_URL = os.environ.get('DATABASE_URL', f"sqlite:///{STORAGE_DIR / 'kayitlar.db'}")

# --- Video ------------------------------------------------------------------
# Videolarda her kareyi işlemek gereksizdir; bu aralıkla örnekleme yapılır.
VIDEO_FRAME_STEP = int(os.environ.get('VIDEO_FRAME_STEP', '15'))
VIDEO_MAX_FRAMES = int(os.environ.get('VIDEO_MAX_FRAMES', '40'))

# --- Sunucu -----------------------------------------------------------------
HOST = os.environ.get('HOST', '0.0.0.0')   # 0.0.0.0 = yerel ağdaki telefonlar erişebilir
PORT = int(os.environ.get('PORT', '8000'))

for d in (UPLOAD_DIR, RESULT_DIR, INCELEME_DIR, EGITIM_DIR):
    d.mkdir(parents=True, exist_ok=True)
