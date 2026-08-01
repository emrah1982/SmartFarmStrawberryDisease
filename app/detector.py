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

from app import config, siniflar

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
    # Bu tespitin BULUNDUĞU organ (hiyerarşik boru hattında dolar).
    #
    # NEDEN GEREKLİ? Aynı sınıf birden çok uzman modelde olabilir: "Gray Mold"
    # hem yaprak hem meyve modelinde tanımlıdır. Organ bilgisi taşınmazsa
    # sonuç ekranı ikisini tek satırda birleştirir ve kullanıcı küfün yaprakta
    # mı meyvede mi olduğunu göremez. İkisinin tarımsal karşılığı farklıdır:
    # meyvedeki kurşuni küf acil hasat/imha, yapraktaki havalandırma demektir.
    #
    # Tek modelli (miras) akışta ve elle etiketlemede boş kalır.
    organ: str = ''


@dataclass
class Sonuc:
    kutular: List[Kutu] = field(default_factory=list)
    sonuc_yolu: str = ''
    islenen_kare: int = 1
    sure_ms: int = 0
    keskinlik: float = 0.0        # Laplacian varyansı — düşükse bulanık
    bulanik_kare: int = 0         # videoda atlanan bulanık kare sayısı
    kalite_notu: str = ''         # kullanıcıya gösterilecek uyarı
    # Videoda AYNI nesne her örneklenen karede yeniden sayılır. `kutular`
    # bu yüzden "kaç nesne var" sorusunun cevabı DEĞİLDİR: 4 meyveli sabit
    # bir sahneden 4 kare örneklendiğinde 11 kutu birikir (ölçüldü).
    #
    # Kareler arası eşleştirme (takip) yapılmadığı sürece dürüst olan alt
    # sınırı vermektir: EN AZ bu kadar nesne vardır.
    kare_basina_en_cok: int = 0
    # Kareler arası TAKİP sonrası benzersiz nesne sayısı (app/takip.py).
    # Asıl cevap budur: aynı meyve kaç karede görünürse görünsün bir sayılır.
    benzersiz_sayi: int = 0
    takip_izi: dict = field(default_factory=dict)
    # Boru hattının ne yaptığı: hangi organlar görüldü, hangi modeller çalıştı.
    # Tespit ÜRETİLMEYEN organlar da burada durur — "5 yaprak gördüm, hastalık
    # bulmadım" ile "hiç yaprak görmedim" arasındaki fark yalnızca burada saklı.
    iz: dict = field(default_factory=dict)

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


def _nms(kutular, iou_esigi: float):
    """Örtüşen kutuları teke indirir (sınıf bazında, güvene göre).

    Çok ölçekli ve dilimli tarama aynı lezyonu birden çok kez bulur; bunlar
    birleştirilmezse tek hastalık 5-6 tespit gibi görünür.
    """
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        kesisim = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if kesisim <= 0:
            return 0.0
        alan_a = (a[2] - a[0]) * (a[3] - a[1])
        alan_b = (b[2] - b[0]) * (b[3] - b[1])
        return kesisim / (alan_a + alan_b - kesisim)

    secili = []
    for k in sorted(kutular, key=lambda x: -x[4]):
        if all(iou(k, s) <= iou_esigi for s in secili if s[5] == k[5]):
            secili.append(k)
    return secili


def _ciz(frame, kutular, cikti_yol: str):
    """Sonuç görselini kullanıcının dilindeki etiketlerle yazar."""
    from app import cizim, dil
    cizim.sonuc_yaz(frame, kutular, cikti_yol, ad_cevir=dil.sinif_adi)


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
        """Tek görüntüyü işler, kutulanmış görseli kaydeder.

        Hiyerarşik mimari (organ → ROI → uzman model) kuruluysa oraya
        yönlendirilir; değilse tek modelle çalışır. Çıktı biçimi aynıdır,
        bu yüzden çağıranların hiçbiri değişmez.
        """
        from app import modeller as model_kutugu
        if model_kutugu.hiyerarsik_hazir():
            from app import pipeline
            return pipeline.goruntu(kaynak_yol, cikti_yol)

        model = self.yukle()
        t0 = time.time()
        r = model(kaynak_yol, conf=siniflar.en_dusuk_esik(), imgsz=config.IMGSZ,
                  verbose=False)[0]

        kutular = self._kutulari_al(r)
        # Etiketler kullanıcının dilinde ve YALNIZCA kabul edilen kutular çizilir
        # (r.plot() İngilizce yazar ve elenen kutuları da çizerdi)
        kare = cv2.imread(kaynak_yol)
        _ciz(kare, kutular, cikti_yol)
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
        from app import modeller as model_kutugu
        hiyerarsik = model_kutugu.hiyerarsik_hazir()
        model = None if hiyerarsik else self.yukle()

        cap = cv2.VideoCapture(kaynak_yol)
        if not cap.isOpened():
            raise RuntimeError(f'Video açılamadı: {kaynak_yol}')

        # Örnekleme SÜREYE göre: sabit kare adımı farklı fps'lerde farklı
        # zaman aralığı demektir ve takip penceresi kayar (bkz. app/takip.py).
        from app import takip as takip_modulu
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps and fps > 0:
            adim = takip_modulu.ornekleme_adimi(fps, config.VIDEO_ORNEK_ARALIK_SN)
        else:
            fps, adim = 30.0, config.VIDEO_FRAME_STEP
        takipci = takip_modulu.Takipci(fps=fps)

        t0 = time.time()
        kutular: List[Kutu] = []
        en_iyi_kare, en_iyi_sayi, en_iyi_kutular = None, -1, []
        idx = islenen = bulanik = 0
        keskinlikler = []
        en_keskin_frame, en_keskin_deger = None, -1.0

        while islenen < config.VIDEO_MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % adim == 0:
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

                # Hiyerarşik mimari kuruluysa her örneklenen kare de organ →
                # ROI → uzman model yolundan geçer; yoksa tek model kullanılır.
                if hiyerarsik:
                    from app import pipeline
                    kare_kutulari = pipeline.calistir(frame)[0]
                    for k in kare_kutulari:
                        k.kare = idx
                else:
                    r = model(frame, conf=siniflar.en_dusuk_esik(), imgsz=config.IMGSZ,
                              verbose=False)[0]
                    kare_kutulari = self._kutulari_al(r, kare=idx)
                # Kareler arası eşleştirme: aynı nesne bir daha sayılmasın
                takipci.ekle(idx, kare_kutulari)
                kutular.extend(kare_kutulari)
                if len(kare_kutulari) > en_iyi_sayi:
                    en_iyi_sayi = len(kare_kutulari)
                    en_iyi_kare, en_iyi_kutular = frame.copy(), kare_kutulari
                islenen += 1
            idx += 1

        cap.release()

        # Tüm kareler bulanıksa yine de en keskin olanı işle — kullanıcı boş dönmesin
        if islenen == 0 and en_keskin_frame is not None:
            if hiyerarsik:
                from app import pipeline
                en_iyi_kutular = pipeline.calistir(en_keskin_frame)[0]
            else:
                r = model(en_keskin_frame, conf=siniflar.en_dusuk_esik(),
                          imgsz=config.IMGSZ, verbose=False)[0]
                en_iyi_kutular = self._kutulari_al(r, kare=0)
            kutular.extend(en_iyi_kutular)
            en_iyi_kare = en_keskin_frame
            islenen = 1
        if en_iyi_kare is not None:
            _ciz(en_iyi_kare, en_iyi_kutular, cikti_yol)

        ort_keskinlik = sum(keskinlikler) / len(keskinlikler) if keskinlikler else 0.0
        toplam = bulanik + islenen
        not_ = ''
        if toplam and bulanik / toplam > 0.4:
            not_ = (f'{toplam} karenin {bulanik} tanesi bulanık olduğu için atlandı. '
                    'Yürürken çekimde hareket bulanıklığı olağandır; daha yavaş '
                    'yürüyüp kısa duraklamalarla çekerseniz tespit doğruluğu artar.')
        elif bulanik:
            not_ = f'{bulanik} bulanık kare atlandı; kalan {islenen} kare işlendi.'

        en_cok = max(en_iyi_sayi, 0)
        benzersiz = takipci.benzersiz_toplam
        if islenen > 1 and len(kutular) > benzersiz:
            not_ = (f'{not_} Video {islenen} karede örneklendi ({adim} karede bir, '
                    f'~{adim / fps:.1f} sn). Toplam {len(kutular)} kutu bulundu; '
                    f'kareler arası takiple bunlar {benzersiz} AYRI nesneye '
                    'indirgendi.').strip()

        return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol if en_iyi_kare is not None else '',
                     islenen_kare=islenen, sure_ms=int((time.time() - t0) * 1000),
                     keskinlik=ort_keskinlik, bulanik_kare=bulanik, kalite_notu=not_,
                     kare_basina_en_cok=en_cok,
                     benzersiz_sayi=benzersiz, takip_izi=takipci.ozet())

    # ------------------------------------------------------- ayrıntılı analiz
    def goruntu_detayli(self, kaynak_yol: str, cikti_yol: str) -> Sonuc:
        """Çok ölçekli + dilimli tarama (ölçek kaynaklı kaçırmaları azaltır).

        HİYERARŞİK MİMARİDE: organ modeli zaten her organı bulup ROI olarak
        kırpıyor; uzman model lezyonu kendi çözünürlüğünde görüyor. Yani
        yakınlaştırma etkisi boru hattının içinde var, ayrıca dilimlemeye
        gerek kalmıyor. Bu yüzden hiyerarşi kuruluysa boru hattına yönlendirilir.
        """
        from app import modeller as model_kutugu
        if model_kutugu.hiyerarsik_hazir():
            from app import pipeline
            return pipeline.goruntu(kaynak_yol, cikti_yol)
        return self._goruntu_detayli_tek_model(kaynak_yol, cikti_yol)

    def _goruntu_detayli_tek_model(self, kaynak_yol: str, cikti_yol: str) -> Sonuc:
        """Çok ölçekli + dilimli analiz (büyük saha fotoğrafları için).

        NEDEN: Tek ölçekli tahmin, çekim ölçeği eğitim verisinden farklı olduğunda
        kararsız davranır — aynı fotoğraf bir çözünürlükte doğru sınıfı bulurken
        diğerinde başka sınıfa kayabilir. Burada görüntü hem birkaç ölçekte hem de
        (büyükse) örtüşen dilimler halinde işlenir; sonuçlar NMS ile birleştirilir.
        Bedeli: birkaç kat daha uzun sürer.
        """
        model = self.yukle()
        t0 = time.time()
        img = cv2.imread(kaynak_yol)
        if img is None:
            raise RuntimeError(f'Görüntü okunamadı: {kaynak_yol}')
        h, w = img.shape[:2]

        ham: List[tuple] = []   # (x1,y1,x2,y2,conf,cls)

        def topla(r, ofs_x=0, ofs_y=0):
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                ham.append((x1 + ofs_x, y1 + ofs_y, x2 + ofs_x, y2 + ofs_y,
                            float(b.conf[0]), int(b.cls[0])))

        # 1) Tam görüntü, birkaç ölçekte
        for boyut in config.DETAYLI_OLCEKLER:
            topla(model(img, imgsz=boyut, conf=siniflar.en_dusuk_esik(),
                        verbose=False)[0])

        # 2) Büyük görüntüde örtüşen dilimler — lezyon kendi çözünürlüğünde görünür
        if max(h, w) > config.DILIM_ESIGI:
            d = config.DILIM_BOYUTU
            adim = int(d * (1 - config.DILIM_ORTUSME))
            for y in range(0, h, adim):
                for x in range(0, w, adim):
                    parca = img[y:min(y + d, h), x:min(x + d, w)]
                    if parca.shape[0] < 200 or parca.shape[1] < 200:
                        continue
                    topla(model(parca, imgsz=d, conf=siniflar.en_dusuk_esik(),
                                verbose=False)[0], x, y)

        secili = _nms(ham, config.NMS_IOU)

        isimler = self._names or model.names
        # Sınıf bazlı eşik burada da uygulanır: dilimli tarama tek karelik
        # analizden daha çok aday üretir, kapalı sınıflar elenmezse yığılır
        kutular = [Kutu(sinif_id=cid, sinif_adi=isimler[cid], guven=guven,
                        x=((x1 + x2) / 2) / w, y=((y1 + y2) / 2) / h,
                        w=(x2 - x1) / w, h=(y2 - y1) / h)
                   for x1, y1, x2, y2, guven, cid in secili
                   if siniflar.kabul_edilir_mi(isimler[cid], guven)]
        _ciz(img, kutular, cikti_yol)

        keskinlik = keskinlik_olc(img)
        notlar = [f'Ayrıntılı analiz: {len(config.DETAYLI_OLCEKLER)} ölçek'
                  + (' + dilimli tarama' if max(h, w) > config.DILIM_ESIGI else '')
                  + f' ({len(ham)} aday → {len(secili)} tespit).']
        if keskinlik < config.BULANIKLIK_ESIGI:
            notlar.append('Görüntü bulanık; sabit tutarak tekrar çekin.')

        return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol, islenen_kare=1,
                     sure_ms=int((time.time() - t0) * 1000),
                     keskinlik=keskinlik, kalite_notu=' '.join(notlar))

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

        t0 = time.time()
        from app import modeller as model_kutugu
        if model_kutugu.hiyerarsik_hazir():
            from app import pipeline
            kutular = pipeline.calistir(frame)[0]
        else:
            r = self.yukle()(frame, conf=siniflar.en_dusuk_esik(),
                             imgsz=config.IMGSZ, verbose=False)[0]
            kutular = self._kutulari_al(r)
        _ciz(frame, kutular, cikti_yol)
        return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol,
                     islenen_kare=1, sure_ms=int((time.time() - t0) * 1000))

    # ------------------------------------------------- canlı akış (tek kare)
    def kare(self, frame, imgsz: Optional[int] = None) -> Sonuc:
        """Bellekteki tek kareyi işler; DOSYA YAZMAZ.

        Canlı akışta saniyede birkaç kare gelir: her karede diske görsel
        yazmak (goruntu()/kamera() gibi) hem yavaşlatır hem depolamayı
        şişirir. Kutular JSON olarak tarayıcıya gönderilir, çizimi tarayıcı
        yapar. Kayıt yalnızca kullanıcı istediğinde/otomatik kural
        tetiklendiğinde ayrıca yapılır.

        imgsz: canlıda küçük tutulur (hız); tek kare analizindeki değerden
        bağımsızdır.
        """
        from app import modeller as model_kutugu
        if model_kutugu.hiyerarsik_hazir():
            from app import pipeline
            return pipeline.kare(frame, imgsz=imgsz)

        model = self.yukle()
        t0 = time.time()
        r = model(frame, conf=siniflar.en_dusuk_esik(),
                  imgsz=imgsz or config.IMGSZ, verbose=False)[0]
        return Sonuc(kutular=self._kutulari_al(r), islenen_kare=1,
                     sure_ms=int((time.time() - t0) * 1000))

    # ------------------------------------------------------------------ yardım
    def _kutulari_al(self, r, kare: int = 0) -> List[Kutu]:
        names = r.names if hasattr(r, 'names') else self._names
        out = []
        for b in r.boxes:
            x, y, w, h = b.xywhn[0].tolist()
            cid = int(b.cls[0])
            ad, guven = names[cid], float(b.conf[0])
            # Sınıf bazlı eşik/kapatma: gürültülü bir sınıfı tek başına
            # sıkılaştırmak, genel eşiği yükseltip her şeyi kaybetmekten iyidir
            if not siniflar.kabul_edilir_mi(ad, guven):
                continue
            out.append(Kutu(sinif_id=cid, sinif_adi=ad, guven=guven,
                            x=x, y=y, w=w, h=h, kare=kare))
        return out


detector = Detector()
