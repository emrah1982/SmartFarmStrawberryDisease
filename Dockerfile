# Çilek Hastalık Tespit — web arayüzü imajı
#
# Varsayılan: CPU. GPU için TORCH_INDEX'i CUDA sürümüyle değiştirin:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libglib2.0-0: opencv-headless'ın tek sistem bağımlılığı
# ffmpeg: RTSP/video kod çözme güvenilirliği için
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# torch ayrı kurulur: CPU tekerleği ~200 MB, CUDA sürümü ~2.5 GB.
# Bu satır ayrı katmanda olduğu için kod değişince yeniden indirilmez.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app ./app
COPY configs ./configs
# Yardimci scriptler (ornegin db_incele.py) konteyner icinden de calissin
COPY scripts ./scripts

# Model ve veriler imaja gömülmez; volume olarak bağlanır
ENV MODEL_PATH=/app/models/best.pt \
    STORAGE_DIR=/app/storage \
    PORT=8000

EXPOSE 8000 8443

# Ultralytics'in yazma denediği dizinler (root olmayan kullanıcıda da çalışsın)
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# app.main uzerinden baslatilir: sertifika (certs/) varsa https ile acilir.
# Canli kamera (getUserMedia) yalnizca https/localhost uzerinde calisir.
CMD ["python", "-m", "app.main"]
