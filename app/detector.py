"""Model yükleme ve tahmin katmanı.

Model bir kez yüklenip bellekte tutulur (her istekte yeniden yüklemek
saniyeler sürer). Fotoğraf, video ve IP kamera kaynakları aynı arayüzden
işlenir; çıktı her zaman kutulanmış görsel + tespit listesidir.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2

from app import config

logger = logging.getLogger(__name__)


@dataclass
class Kutu:
    sinif_id: int
    sinif_adi: str
    guven: float
    x: float
    y: float
    w: float
    h: float
    kare: int = 0


@dataclass
class Sonuc:
    kutular: List[Kutu] = field(default_factory=list)
    sonuc_yolu: str = ''
    islenen_kare: int = 1
    sure_ms: int = 0

    @property
    def min_guven(self) -> float:
        return min((k.guven for k in self.kutular), default=0.0)

    @property
    def ort_guven(self) -> float:
        return sum(k.guven for k in self.kutular) / len(self.kutular) if self.kutular else 0.0

    @property
    def inceleme_gerekli(self) -> bool:
        """Tespit yoksa veya en düşük güven eşiğin altındaysa uzman baksın.

        Bu kayıtlar sürekli iyileştirme döngüsünün girdisidir: modelin
        zorlandığı örnekler etiketlenince en çok kazanımı onlar sağlar.
        """
        if not self.kutular:
            return True
        return self.min_guven < config.REVIEW_THRESHOLD


class Detector:
    """Ultralytics modelini sarmalar. İlk kullanımda yüklenir (lazy)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.MODEL_PATH
        self._model = None
        self._names = {}

    @property
    def hazir(self) -> bool:
        return Path(self.model_path).exists()

    def yukle(self):
        if self._model is not None:
            return self._model
        if not self.hazir:
            raise FileNotFoundError(
                f'Model bulunamadı: {self.model_path}\n'
                'Eğitilmiş best.pt dosyasını models/ klasörüne koyun veya '
                'MODEL_PATH ortam değişkenini ayarlayın.'
            )
        from ultralytics import YOLO           # ağır import — yalnızca gerektiğinde
        logger.info(f'Model yükleniyor: {self.model_path}')
        self._model = YOLO(self.model_path)
        self._names = self._model.names
        return self._model

    @property
    def siniflar(self) -> dict:
        if not self._names and self.hazir:
            self.yukle()
        return self._names

    # ---------------------------------------------------------------- görüntü
    def goruntu(self, kaynak_yol: str, cikti_yol: str) -> Sonuc:
        """Tek görüntüyü işler, kutulanmış görseli kaydeder."""
        model = self.yukle()
        t0 = time.time()
        r = model(kaynak_yol, conf=config.CONF_THRESHOLD, imgsz=config.IMGSZ, verbose=False)[0]

        kutular = self._kutulari_al(r)
        cv2.imwrite(cikti_yol, r.plot())
        return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol,
                     islenen_kare=1, sure_ms=int((time.time() - t0) * 1000))

    # ------------------------------------------------------------------ video
    def video(self, kaynak_yol: str, cikti_yol: str) -> Sonuc:
        """Videoyu örnekleyerek işler.

        Her kareyi işlemek gereksiz ve yavaştır; VIDEO_FRAME_STEP aralığıyla
        örneklenir. En çok tespit içeren kare önizleme olarak kaydedilir.
        """
        model = self.yukle()
        cap = cv2.VideoCapture(kaynak_yol)
        if not cap.isOpened():
            raise RuntimeError(f'Video açılamadı: {kaynak_yol}')

        t0 = time.time()
        kutular: List[Kutu] = []
        en_iyi_kare, en_iyi_sayi = None, -1
        idx = islenen = 0

        while islenen < config.VIDEO_MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % config.VIDEO_FRAME_STEP == 0:
                r = model(frame, conf=config.CONF_THRESHOLD, imgsz=config.IMGSZ, verbose=False)[0]
                kare_kutulari = self._kutulari_al(r, kare=idx)
                kutular.extend(kare_kutulari)
                if len(kare_kutulari) > en_iyi_sayi:
                    en_iyi_sayi, en_iyi_kare = len(kare_kutulari), r.plot()
                islenen += 1
            idx += 1

        cap.release()
        if en_iyi_kare is not None:
            cv2.imwrite(cikti_yol, en_iyi_kare)
        return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol if en_iyi_kare is not None else '',
                     islenen_kare=islenen, sure_ms=int((time.time() - t0) * 1000))

    # ----------------------------------------------------------------- kamera
    def kamera(self, url: str, cikti_yol: str, kaynak_kaydet: Optional[str] = None) -> Sonuc:
        """IP kameradan tek kare alıp işler (isteğe bağlı anlık çekim)."""
        cap = cv2.VideoCapture(url)
        try:
            if not cap.isOpened():
                raise RuntimeError(
                    f'Kameraya bağlanılamadı: {url}\n'
                    'URL doğru mu? RTSP için rtsp://kullanici:parola@ip:554/... biçimi kullanılır.'
                )
            # İlk kare bazen bozuk gelir; birkaç kare atlayıp taze görüntü alınır
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError('Kameradan görüntü alınamadı (bağlantı var ama kare yok).')
        finally:
            cap.release()

        if kaynak_kaydet:
            cv2.imwrite(kaynak_kaydet, frame)

        model = self.yukle()
        t0 = time.time()
        r = model(frame, conf=config.CONF_THRESHOLD, imgsz=config.IMGSZ, verbose=False)[0]
        cv2.imwrite(cikti_yol, r.plot())
        return Sonuc(kutular=self._kutulari_al(r), sonuc_yolu=cikti_yol,
                     islenen_kare=1, sure_ms=int((time.time() - t0) * 1000))

    # ------------------------------------------------------------------ yardım
    def _kutulari_al(self, r, kare: int = 0) -> List[Kutu]:
        names = r.names if hasattr(r, 'names') else self._names
        out = []
        for b in r.boxes:
            x, y, w, h = b.xywhn[0].tolist()
            cid = int(b.cls[0])
            out.append(Kutu(sinif_id=cid, sinif_adi=names[cid], guven=float(b.conf[0]),
                            x=x, y=y, w=w, h=h, kare=kare))
        return out


detector = Detector()
