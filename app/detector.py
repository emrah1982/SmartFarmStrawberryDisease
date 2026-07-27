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
    keskinlik: float = 0.0        # Laplacian varyansı — düşükse bulanık
    bulanik_kare: int = 0         # videoda atlanan bulanık kare sayısı
    kalite_notu: str = ''         # kullanıcıya gösterilecek uyarı

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


def keskinlik_olc(frame) -> float:
    """Laplacian varyansı: yüksek = keskin, düşük = bulanık.

    Odak/hareket bulanıklığının standart ölçüsüdür. Bulanık kareyi modele
    vermek yanlış veya eksik tespit üretir; özellikle yürürken çekilen
    videolarda karelerin bir kısmı kullanılamaz durumda olur.
    """
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gri, cv2.CV_64F).var())


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

        kare = cv2.imread(kaynak_yol)
        keskinlik = keskinlik_olc(kare) if kare is not None else 0.0
        not_ = ''
        if keskinlik and keskinlik < config.BULANIKLIK_ESIGI:
            not_ = (f'Görüntü bulanık (keskinlik {keskinlik:.0f}, '
                    f'eşik {config.BULANIKLIK_ESIGI}). Tespitler eksik olabilir; '
                    'sabit tutarak ve iyi ışıkta tekrar çekin.')

        return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol, islenen_kare=1,
                     sure_ms=int((time.time() - t0) * 1000),
                     keskinlik=keskinlik, kalite_notu=not_)

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
        idx = islenen = bulanik = 0
        keskinlikler = []
        en_keskin_frame, en_keskin_deger = None, -1.0

        while islenen < config.VIDEO_MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % config.VIDEO_FRAME_STEP == 0:
                # Bulanık kareyi modele vermek yanlış tespit üretir — atla.
                # Yürürken çekimde karelerin bir kısmı hareket bulanıklığı taşır.
                k = keskinlik_olc(frame)
                keskinlikler.append(k)
                if k > en_keskin_deger:
                    en_keskin_deger, en_keskin_frame = k, frame.copy()
                if k < config.BULANIKLIK_ESIGI:
                    bulanik += 1
                    idx += 1
                    continue

                r = model(frame, conf=config.CONF_THRESHOLD, imgsz=config.IMGSZ, verbose=False)[0]
                kare_kutulari = self._kutulari_al(r, kare=idx)
                kutular.extend(kare_kutulari)
                if len(kare_kutulari) > en_iyi_sayi:
                    en_iyi_sayi, en_iyi_kare = len(kare_kutulari), r.plot()
                islenen += 1
            idx += 1

        cap.release()

        # Tüm kareler bulanıksa yine de en keskin olanı işle — kullanıcı boş dönmesin
        if islenen == 0 and en_keskin_frame is not None:
            r = model(en_keskin_frame, conf=config.CONF_THRESHOLD,
                      imgsz=config.IMGSZ, verbose=False)[0]
            kutular.extend(self._kutulari_al(r, kare=0))
            en_iyi_kare = r.plot()
            islenen = 1
        if en_iyi_kare is not None:
            cv2.imwrite(cikti_yol, en_iyi_kare)

        ort_keskinlik = sum(keskinlikler) / len(keskinlikler) if keskinlikler else 0.0
        toplam = bulanik + islenen
        not_ = ''
        if toplam and bulanik / toplam > 0.4:
            not_ = (f'{toplam} karenin {bulanik} tanesi bulanık olduğu için atlandı. '
                    'Yürürken çekimde hareket bulanıklığı olağandır; daha yavaş '
                    'yürüyüp kısa duraklamalarla çekerseniz tespit doğruluğu artar.')
        elif bulanik:
            not_ = f'{bulanik} bulanık kare atlandı; kalan {islenen} kare işlendi.'

        return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol if en_iyi_kare is not None else '',
                     islenen_kare=islenen, sure_ms=int((time.time() - t0) * 1000),
                     keskinlik=ort_keskinlik, bulanik_kare=bulanik, kalite_notu=not_)

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
