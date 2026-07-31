"""Hiyerarşik çıkarım boru hattı: organ → ROI → uzman model → birleştirme.

AKIŞ
    görüntü
      ↓  organ modeli          (leaf / fruit / flower / stem / runner)
      ↓  ROI kırpma (pay ile)
      ↓  o organda tetiklenen uzman modeller
      ↓  koordinatları orijinal görüntüye geri dönüştürme
      ↓  eşik süzme + örtüşen kutuları birleştirme
      ↓  tek Sonuc

NEDEN BÖYLE?
    Tek modelde bütün sınıflar birbirine karışıyordu. Somut örnek: olgunluk
    sınıfları yaprakları "olgunlaşmamış çilek" sanıyordu ve iki grubun güven
    aralıkları üst üste bindiği için ayıran bir eşik yoktu.

    Burada olgunluk modeli YALNIZCA meyve ROI'si görür. Yaprağı olgunlaşmamış
    meyve diye işaretlemesi yapısal olarak imkânsızdır — sorun bastırılmaz,
    ortadan kalkar.

GERİYE DÖNÜK UYUMLULUK
    Çıktı yine `Sonuc`/`Kutu`dur. Veritabanı, etiketleme, dışa aktarım ve
    harita hiç değişmeden çalışır. Uzman modeller eksikse boru hattı MİRAS
    modele düşer; yani mimariye kademeli geçilir.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from app import config, modeller, siniflar
from app.detector import Kutu, Sonuc, _nms, keskinlik_olc

logger = logging.getLogger(__name__)

# ROI kırpılırken bırakılan pay (kutu kenarının oranı). Lezyon çoğu zaman
# organ kutusunun tam kenarındadır; paysız kırpma onu ikiye böler.
ROI_PAY = 0.12

# Bu boyutun altındaki ROI'de uzman model anlamlı çalışmaz (piksel yetersiz).
EN_KUCUK_ROI = 32


@dataclass
class Iz:
    """Boru hattının ne yaptığının kaydı — arayüzde ve hata ayıklamada kullanılır."""
    organlar: List[Tuple[str, float]]
    calisan_modeller: List[str]
    roi_sayisi: int
    miras_kullanildi: bool
    sure_ms: int = 0

    def organ_sayilari(self) -> dict:
        """{organ: adet} — tespit üretmemiş organlar da burada."""
        out = {}
        for o, _ in self.organlar:
            out[o] = out.get(o, 0) + 1
        return out

    def sozluk(self) -> dict:
        """Kaydedilebilir/şablona verilebilir biçim.

        Sonuç ekranı "5 yaprak gördüm, hastalık bulmadım" ile "hiç yaprak
        görmedim" arasındaki farkı ancak buradan bilebilir; tespit listesi
        yalnızca BULUNANLARI içerir.
        """
        return {
            'organlar': self.organ_sayilari(),
            'modeller': list(self.calisan_modeller),
            'roi': self.roi_sayisi,
            'miras': self.miras_kullanildi,
            'sure_ms': self.sure_ms,
        }

    def ozet(self) -> str:
        if self.miras_kullanildi and not self.organlar:
            return 'Tek model (miras) — organ modeli yok'
        organ = ', '.join(f'{a}×{n}' for a, n in self.organ_sayilari().items())
        return (f'{len(self.organlar)} organ ({organ}) → '
                f'{self.roi_sayisi} ROI → {", ".join(self.calisan_modeller) or "—"}')


def _kirp(frame, kutu: Kutu, pay: float = ROI_PAY):
    """Kutuyu paylı kırpar. Returns: (kirpik, x_ofset, y_ofset) veya None."""
    h, w = frame.shape[:2]
    yariw, yarih = kutu.w * (1 + pay) / 2, kutu.h * (1 + pay) / 2
    x1 = max(0, int((kutu.x - yariw) * w))
    y1 = max(0, int((kutu.y - yarih) * h))
    x2 = min(w, int((kutu.x + yariw) * w))
    y2 = min(h, int((kutu.y + yarih) * h))
    if x2 - x1 < EN_KUCUK_ROI or y2 - y1 < EN_KUCUK_ROI:
        return None
    return frame[y1:y2, x1:x2], x1, y1


def roi_kutusunu_cevir(kutu_roi: Kutu, roi_genislik: int, roi_yukseklik: int,
                       ofset_x: int, ofset_y: int,
                       tam_genislik: int, tam_yukseklik: int,
                       organ: str = '') -> Kutu:
    """ROI koordinatındaki kutuyu ORİJİNAL görüntü koordinatına çevirir.

    Uzman model kırpıntı üzerinde çalışır ve kutularını 0-1 aralığında
    KIRPINTIYA GÖRE verir. Geri dönüştürülmezse tespit görüntünün yanlış
    yerinde görünür — sessiz ve fark edilmesi zor bir hatadır, bu yüzden
    ayrı fonksiyon ve testle sabitlenmiştir.

    `organ` da burada damgalanır: kutu hangi organın kırpıntısından çıktıysa
    o bilgi tek yerde, dönüşümün yapıldığı noktada eklenir.
    """
    px = kutu_roi.x * roi_genislik + ofset_x
    py = kutu_roi.y * roi_yukseklik + ofset_y
    pw = kutu_roi.w * roi_genislik
    ph = kutu_roi.h * roi_yukseklik
    return Kutu(sinif_id=kutu_roi.sinif_id, sinif_adi=kutu_roi.sinif_adi,
                guven=kutu_roi.guven,
                x=px / tam_genislik, y=py / tam_yukseklik,
                w=pw / tam_genislik, h=ph / tam_yukseklik,
                kare=kutu_roi.kare, organ=organ or kutu_roi.organ)


def _tahmin(model, goruntu, esik: float, imgsz: Optional[int] = None) -> List[Kutu]:
    """Bir modeli çalıştırıp Kutu listesi döner."""
    try:
        r = model(goruntu, conf=esik, imgsz=imgsz or config.IMGSZ, verbose=False)[0]
    except Exception as e:
        logger.warning(f'Tahmin başarısız: {e}')
        return []
    adlar = r.names
    out = []
    for b in r.boxes:
        x, y, w, h = b.xywhn[0].tolist()
        cid = int(b.cls[0])
        out.append(Kutu(sinif_id=cid, sinif_adi=adlar[cid], guven=float(b.conf[0]),
                        x=x, y=y, w=w, h=h))
    return out


def _birlestir(kutular: List[Kutu]) -> List[Kutu]:
    """Örtüşen kutuları teke indirir.

    Aynı lezyon birden çok ROI'de görülebilir (organ kutuları paylı kırpıldığı
    için üst üste biner). Birleştirilmezse tek hastalık birkaç tespit gibi görünür.
    """
    if len(kutular) < 2:
        return kutular
    ham = [(k.x - k.w / 2, k.y - k.h / 2, k.x + k.w / 2, k.y + k.h / 2,
            k.guven, k.sinif_adi) for k in kutular]
    secili = _nms(ham, config.NMS_IOU)
    secili_kume = {(round(s[0], 6), round(s[1], 6), s[4], s[5]) for s in secili}
    return [k for k in kutular
            if (round(k.x - k.w / 2, 6), round(k.y - k.h / 2, 6),
                k.guven, k.sinif_adi) in secili_kume]


def _kabul(kutular: List[Kutu], esik: float, urun=None) -> List[Kutu]:
    """Sınıf bazlı eşik + kapalı sınıf elemesi (ürünün siniflar.yaml'ı)."""
    return [k for k in kutular
            if k.guven >= max(esik, siniflar.esik(k.sinif_adi, urun))
            and siniflar.aktif_mi(k.sinif_adi, urun)]


def calistir(frame, imgsz: Optional[int] = None,
             urun: Optional[str] = None) -> Tuple[List[Kutu], Iz]:
    """Bir kareyi hiyerarşik boru hattından geçirir.

    Returns: (kutular, iz)
    """
    t0 = time.time()
    h, w = frame.shape[:2]
    organ_tanimlari = modeller.rol_ile('organ', urun)

    # --- Organ modeli yoksa: mirasa düş -----------------------------------
    if not organ_tanimlari:
        miras = modeller.yukle('miras', urun)
        if miras is None:
            return [], Iz([], [], 0, True, int((time.time() - t0) * 1000))
        t = modeller.tanim('miras', urun)
        kutular = _kabul(_tahmin(miras, frame, t.esik, imgsz), t.esik, urun)
        return kutular, Iz([], ['miras'], 0, True, int((time.time() - t0) * 1000))

    # --- 1) Organ tespiti --------------------------------------------------
    organ_tanim = organ_tanimlari[0]
    organ_model = modeller.yukle(organ_tanim.ad, urun)
    organlar = _tahmin(organ_model, frame, organ_tanim.esik, imgsz) if organ_model else []

    sonuc: List[Kutu] = []
    calisan: List[str] = []
    roi_sayisi = 0

    # --- 2) Her organ için tetiklenen uzman modeller -----------------------
    for organ in organlar:
        uzmanlar = modeller.tetiklenen(organ.sinif_adi, urun)
        if not uzmanlar:
            continue
        kirpma = _kirp(frame, organ)
        if kirpma is None:
            continue
        roi, ofs_x, ofs_y = kirpma
        roi_h, roi_w = roi.shape[:2]
        roi_sayisi += 1

        for u in uzmanlar:
            m = modeller.yukle(u.ad, urun)
            if m is None:
                continue
            if u.ad not in calisan:
                calisan.append(u.ad)
            for k in _tahmin(m, roi, u.esik, imgsz):
                sonuc.append(roi_kutusunu_cevir(k, roi_w, roi_h, ofs_x, ofs_y, w, h,
                                                organ=organ.sinif_adi))

    # --- 3) Hiç uzman çalışmadıysa mirasla tamamla ------------------------
    miras_kullanildi = False
    if not calisan:
        miras = modeller.yukle('miras', urun)
        if miras is not None:
            t = modeller.tanim('miras', urun)
            sonuc.extend(_tahmin(miras, frame, t.esik, imgsz))
            miras_kullanildi = True
            calisan.append('miras')

    sonuc = _birlestir(_kabul(sonuc, config.CONF_THRESHOLD, urun))
    iz = Iz([(o.sinif_adi, o.guven) for o in organlar], calisan, roi_sayisi,
            miras_kullanildi, int((time.time() - t0) * 1000))
    return sonuc, iz


def goruntu(kaynak_yol: str, cikti_yol: str, urun: Optional[str] = None) -> Sonuc:
    """Tek görüntüyü boru hattından geçirip kutulanmış görseli yazar."""
    from app import cizim, dil

    frame = cv2.imread(kaynak_yol)
    if frame is None:
        raise RuntimeError(f'Görüntü okunamadı: {kaynak_yol}')

    t0 = time.time()
    kutular, iz = calistir(frame, urun=urun)
    cizim.sonuc_yaz(frame, kutular, cikti_yol, ad_cevir=dil.sinif_adi)

    keskinlik = keskinlik_olc(frame)
    # kalite_notu YALNIZCA görüntü kalitesi içindir. Boru hattı izi eskiden
    # buraya yazılıyordu ve arayüzde "Görüntü kalitesi" kutusunda
    # "2 organ (Fruit×2) → 2 ROI" gibi geliştirici metni çıkıyordu.
    kalite = ''
    if keskinlik < config.BULANIKLIK_ESIGI:
        kalite = 'Görüntü bulanık; sabit tutarak tekrar çekin.'

    return Sonuc(kutular=kutular, sonuc_yolu=cikti_yol, islenen_kare=1,
                 sure_ms=int((time.time() - t0) * 1000),
                 keskinlik=keskinlik, kalite_notu=kalite, iz=iz.sozluk())


def kare(frame, imgsz: Optional[int] = None, urun: Optional[str] = None) -> Sonuc:
    """Bellekteki kare (canlı akış) — dosya yazmaz."""
    t0 = time.time()
    kutular, iz = calistir(frame, imgsz=imgsz, urun=urun)
    return Sonuc(kutular=kutular, islenen_kare=1,
                 sure_ms=int((time.time() - t0) * 1000), iz=iz.sozluk())
