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

# --- Depolama --------------------------------------------------------------
STORAGE_DIR = Path(os.environ.get('STORAGE_DIR', str(BASE_DIR / 'storage')))
UPLOAD_DIR = STORAGE_DIR / 'uploads'      # yüklenen orijinaller
RESULT_DIR = STORAGE_DIR / 'results'      # kutulanmış çıktılar
EXPORT_DIR = STORAGE_DIR / 'exports'      # etiketleme için dışa aktarım

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

for d in (UPLOAD_DIR, RESULT_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)
