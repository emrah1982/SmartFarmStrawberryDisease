"""Canlı akışın saf mantığı — sunucu/veritabanı bilmez, bu yüzden test edilebilir.

Buradaki hiçbir fonksiyon HTTP, WebSocket veya DB'ye dokunmaz; girdi alır,
çıktı verir. Kayıt kararı gibi asıl kritik davranış (yanlış tespitler kayda
geçmesin) böylece kamerasız test edilebiliyor.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.detector import Kutu, Sonuc, keskinlik_olc
from app.moduller.canli import ayarlar


def kare_coz(veri: bytes):
    """JPEG baytlarını OpenCV karesine çevirir. Bozuksa None."""
    import cv2
    dizi = np.frombuffer(veri, dtype=np.uint8)
    return cv2.imdecode(dizi, cv2.IMREAD_COLOR)


def _detector():
    """Çekirdeğin dedektörü — testlerde app.main.detector değiştirilebilsin diye
    modül yüklenirken değil, çağrı anında alınır."""
    from app import main
    return main.detector


def tespit(frame, imgsz: Optional[int] = None) -> Sonuc:
    """Tek kareyi işler. Bulanık kare modele verilmez."""
    keskinlik = keskinlik_olc(frame) if ayarlar.BULANIKLIK_ESIGI else 0.0
    if ayarlar.BULANIKLIK_ESIGI and keskinlik < ayarlar.BULANIKLIK_ESIGI:
        # Bulanık kare yanlış/eksik tespit üretir; boş sonuç dönüp bir sonraki
        # kareyi beklemek daha doğru — kullanıcıya "sabit tutun" denir.
        return Sonuc(kutular=[], islenen_kare=0, keskinlik=keskinlik,
                     kalite_notu='bulanik')
    sonuc = _detector().kare(frame, imgsz=imgsz or ayarlar.CANLI_IMGSZ)
    sonuc.keskinlik = keskinlik
    return sonuc


def kutu_sozlugu(k: Kutu) -> dict:
    """Kutuyu tarayıcının çizeceği biçime çevirir (0-1 normalize koordinat)."""
    return {'sinif_id': k.sinif_id, 'ad': k.sinif_adi, 'guven': round(k.guven, 3),
            'x': round(k.x, 4), 'y': round(k.y, 4),
            'w': round(k.w, 4), 'h': round(k.h, 4)}


@dataclass
class KayitKarari:
    """Bir bulgunun kaydedilmeye değer olup olmadığına karar verir.

    NEDEN GEREKLİ?
        Canlı akışta saniyede birkaç kare gelir. Her tespiti kaydetmek
        veritabanını saniyede birkaç kayıtla doldurur ve tek karelik yanlış
        tespitler de kayda geçer.

    KURAL
        1. Güven eşiği: düşük güvenli tespit tek başına kayıt açmaz.
        2. Kararlılık: aynı sınıf üst üste N karede görülmeli — gerçek bir
           lezyon birkaç kare boyunca durur, gürültü durmaz.
        3. Bekleme: aynı sınıf tekrar kaydedilmeden önce N saniye geçmeli.
    """
    kararlilik_kare: int = ayarlar.KARARLILIK_KARE
    guven_esigi: float = ayarlar.KAYIT_GUVEN
    bekleme_sn: float = ayarlar.BEKLEME_SN

    _ardisik: Dict[int, int] = field(default_factory=dict)
    _son_kayit: Dict[int, float] = field(default_factory=dict)

    def degerlendir(self, kutular: List[Kutu], simdi: float) -> Optional[Kutu]:
        """Kaydedilecek kutuyu döner; yoksa None.

        simdi: monotonik saniye (dışarıdan verilir → test zamana bağlı kalmaz).
        """
        gorulen = set()
        aday = None
        for k in sorted(kutular, key=lambda x: -x.guven):
            if k.guven < self.guven_esigi:
                continue
            gorulen.add(k.sinif_id)
            self._ardisik[k.sinif_id] = self._ardisik.get(k.sinif_id, 0) + 1
            if self._ardisik[k.sinif_id] < self.kararlilik_kare:
                continue
            son = self._son_kayit.get(k.sinif_id)
            if son is not None and simdi - son < self.bekleme_sn:
                continue
            if aday is None or k.guven > aday.guven:
                aday = k

        # Karede görünmeyen sınıfların sayacı sıfırlanır: "üst üste" şartı
        # gerçekten ardışık kareleri ifade etsin.
        for sid in list(self._ardisik):
            if sid not in gorulen:
                self._ardisik[sid] = 0

        if aday is not None:
            self._son_kayit[aday.sinif_id] = simdi
        return aday


def ciz(frame, kutular: List[Kutu]):
    """Kaydedilecek kareye kutuları çizer (arşiv görseli için).

    Tarayıcı zaten canlıda çiziyor; bu yalnızca kaydedilen karede, sonradan
    geçmişte bakıldığında ne bulunduğunun görünmesi için.
    """
    import cv2
    y_boy, x_boy = frame.shape[:2]
    kopya = frame.copy()
    for k in kutular:
        x1 = int((k.x - k.w / 2) * x_boy)
        y1 = int((k.y - k.h / 2) * y_boy)
        x2 = int((k.x + k.w / 2) * x_boy)
        y2 = int((k.y + k.h / 2) * y_boy)
        cv2.rectangle(kopya, (x1, y1), (x2, y2), (0, 0, 220), 2)
        etiket = f'{k.sinif_adi} {k.guven:.2f}'
        cv2.putText(kopya, etiket, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
    return kopya
