"""Kareler arası nesne takibi — video/drone/canlı akışta BENZERSİZ sayım.

SORUN
    Video işlenirken her örneklenen karenin kutuları biriktiriliyordu ve
    kareler arası eşleştirme yoktu. Aynı meyve her karede yeniden sayıldığı
    için sayı şişiyordu. Ölçüldü:

        4 meyveli SABİT sahne, 4 kare örneklendi  →  11 kutu

    Kullanıcı bunu "11 hastalıklı meyve" diye okursa yanlış tarımsal karar
    verir: gereksiz ilaçlama, yanlış hasat planı, boşuna imha.

ÇÖZÜM
    Her kutuya KALICI KİMLİK verilir. Bir sonraki karede aynı nesne
    bulunduğunda aynı kimliği alır; benzersiz sayım kimlik sayısıdır.

NEDEN FPS ÖNEMLİ — KULLANICININ TESPİTİ
    Örnekleme sabit KARE aralığıyla yapılıyordu (her 15. kare). Bu 30 fps'te
    0,5 saniye, 60 fps'te 0,25 saniye demektir — aynı ayar farklı videolarda
    farklı davranır.

    Takip için önemli olan kare sayısı değil GEÇEN SÜREDİR: bir nesne iki
    örnek arasında ne kadar yer değiştirebilir? Yürüyerek çekimde saniyede
    kabaca kadrajın üçte biri kadar. Bu yüzden eşleştirme penceresi
    saniyeye göre hesaplanır, kareye göre değil.

NEDEN KALMAN/ByteTrack DEĞİL?
    Bu kütüphaneler ARDIŞIK kare bekler. Biz aralıklı örnekliyoruz (her
    kareyi işlemek 15 kat pahalı); aradaki hareket büyük olduğu için
    Kalman öngörüsü zaten güvenilmez olur. Burada yapılan şey daha basit
    ve bu veri için daha dürüst: sınıf + konum yakınlığı ile eşleştirme,
    kayıp toleransı ve süreye göre genişleyen arama penceresi.

    Sınırı açıkça söylüyoruz: hızlı hareket eden kamerada veya çok sık
    nesnede (sıra sıra çilek) eşleştirme hata yapabilir. Bu yüzden sonuç
    "kesin sayı" değil "benzersiz tahmin" olarak sunulur.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Nesne saniyede kadrajın en çok bu kadarını kat edebilir (merkez kayması).
# Yürüyerek çekimde ölçülen tipik değer; drone/hızlı pan için artırılabilir.
VARSAYILAN_HIZ = 0.35

# İki kutu bu orandan fazla örtüşüyorsa aynı nesne sayılır (konum yakınlığı
# tek başına yetmez: yan yana iki çilek merkezce yakındır ama örtüşmez).
IOU_ESIGI = 0.30

# Nesne bu kadar saniye görünmezse izi kapatılır. Yaprak arkasına giren bir
# meyve birkaç karede kaybolup geri gelebilir; hemen kapatmak onu iki ayrı
# nesne sayardı.
KAYIP_TOLERANS_SN = 1.5


def _iou(a, b) -> float:
    """İki normalize kutunun kesişim/birleşim oranı."""
    ax1, ay1 = a.x - a.w / 2, a.y - a.h / 2
    ax2, ay2 = a.x + a.w / 2, a.y + a.h / 2
    bx1, by1 = b.x - b.w / 2, b.y - b.h / 2
    bx2, by2 = b.x + b.w / 2, b.y + b.h / 2

    kx1, ky1 = max(ax1, bx1), max(ay1, by1)
    kx2, ky2 = min(ax2, bx2), min(ay2, by2)
    if kx2 <= kx1 or ky2 <= ky1:
        return 0.0
    kesisim = (kx2 - kx1) * (ky2 - ky1)
    birlesim = a.w * a.h + b.w * b.h - kesisim
    return kesisim / birlesim if birlesim > 0 else 0.0


def _mesafe(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


@dataclass
class Iz:
    """Bir nesnenin kareler boyunca izi."""
    kimlik: int
    sinif_adi: str
    ilk_zaman: float                     # saniye
    son_zaman: float
    kutu: object                         # son görülen kutu
    onceki_kutu: Optional[object] = None  # çizgi geçişi için gerekli
    ilk_kutu: Optional[object] = None     # hareket ölçümü için başlangıç konumu
    gorulme: int = 1
    en_yuksek_guven: float = 0.0
    kareler: List[int] = field(default_factory=list)
    cizgiyi_gecti: bool = False

    @property
    def sure(self) -> float:
        return max(0.0, self.son_zaman - self.ilk_zaman)


class CizgiSayaci:
    """Sanal çizgi (tripwire) geçiş sayacı.

    NE ZAMAN İŞE YARAR?
        Kamera TEK YÖNDE ve DÜZENLİ ilerlerken: drone transekti, sıra
        boyunca sabit hızla yürüyüş, bant üstü ürün. Nesne çizgiyi geçtiği
        anda sayılır; kadrajda ne kadar kaldığı önemsizdir.

        Bu, uzun taramalarda benzersiz-iz sayımından DAHA SAĞLAMDIR: iz
        kopup yeniden kurulsa bile (bulanıklık, yaprak arkası) nesne
        çizgiyi bir kez geçmiştir.

    NE ZAMAN İŞE YARAMAZ — ÖNEMLİ
        1. SABİT çekim: kamera durunca hiçbir şey çizgiyi geçmez, sayı 0
           kalır. Oysa kadrajda 5 meyve olabilir.
        2. DUR-KALK yürüyüş: bizim çekim rehberimiz "2-3 adımda bir yarım
           saniye durun" diyor. Yön tutarsızlaşır, nesne çizgi üstünde
           gidip gelirse iki kez sayılır.
        3. SEYREK ÖRNEKLEME: nesne iki örnek arasında çizginin bir
           yanından öbürüne atlarsa geçiş görülür; ama kadraja girip
           çıkarsa hiç görülmez. Örnekleme aralığı, nesnenin kadrajda
           EN AZ İKİ kez görüleceği kadar kısa olmalı.

    Bu yüzden VARSAYILAN OLARAK KAPALIDIR ve benzersiz-iz sayımının yerine
    değil YANINDA sunulur. Hangisinin doğru olduğunu çekim biçimi belirler.
    """

    def __init__(self, eksen: str = 'x', konum: float = 0.5):
        if eksen not in ('x', 'y'):
            raise ValueError("eksen 'x' veya 'y' olmalı")
        self.eksen = eksen
        self.konum = min(max(konum, 0.0), 1.0)
        self.ileri: Dict[str, int] = {}      # soldan sağa / yukarıdan aşağıya
        self.geri: Dict[str, int] = {}

    def _deger(self, kutu) -> float:
        return kutu.x if self.eksen == 'x' else kutu.y

    def kontrol(self, iz: Iz) -> Optional[str]:
        """İz çizgiyi yeni geçtiyse yönü döner ('ileri'/'geri'), yoksa None.

        Bir iz YALNIZCA BİR KEZ sayılır: çizgi üstünde titreyen bir nesne
        (dur-kalk yürüyüşte olur) defalarca sayılmamalı.
        """
        if iz.cizgiyi_gecti or iz.onceki_kutu is None:
            return None
        onceki = self._deger(iz.onceki_kutu)
        simdiki = self._deger(iz.kutu)
        if onceki == simdiki:
            return None
        if (onceki < self.konum <= simdiki) or (simdiki < self.konum <= onceki):
            yon = 'ileri' if simdiki > onceki else 'geri'
            hedef = self.ileri if yon == 'ileri' else self.geri
            hedef[iz.sinif_adi] = hedef.get(iz.sinif_adi, 0) + 1
            iz.cizgiyi_gecti = True
            return yon
        return None

    @property
    def toplam(self) -> int:
        return sum(self.ileri.values()) + sum(self.geri.values())

    def ozet(self) -> dict:
        return {
            'eksen': self.eksen, 'konum': self.konum,
            'ileri': dict(self.ileri), 'geri': dict(self.geri),
            'toplam': self.toplam,
        }


class Takipci:
    """Kareler arası eşleştirme.

    Kullanım:
        t = Takipci(fps=30)
        for kare_no, kutular in ...:
            t.ekle(kare_no, kutular)
        t.benzersiz_sayim()   # {'strawberry_ripe': 4, ...}
    """

    def __init__(self, fps: float = 30.0, hiz: float = VARSAYILAN_HIZ,
                 iou_esigi: float = IOU_ESIGI,
                 kayip_tolerans_sn: float = KAYIP_TOLERANS_SN,
                 cizgi: Optional[CizgiSayaci] = None):
        self.fps = fps if fps and fps > 0 else 30.0
        self.hiz = hiz
        self.iou_esigi = iou_esigi
        self.kayip_tolerans_sn = kayip_tolerans_sn
        # İSTEĞE BAĞLI çizgi sayacı. Varsayılan kapalı: yalnızca tek yönde
        # düzenli ilerleyen çekimde anlamlıdır (bkz. CizgiSayaci).
        self.cizgi = cizgi
        self.izler: List[Iz] = []
        self._sonraki_kimlik = 1
        self._son_zaman = 0.0

    # ------------------------------------------------------------------
    def _zaman(self, kare_no: int) -> float:
        return kare_no / self.fps

    def _aday_izler(self, zaman: float) -> List[Iz]:
        """Hâlâ açık olan izler (kayıp toleransı içinde)."""
        return [i for i in self.izler
                if zaman - i.son_zaman <= self.kayip_tolerans_sn]

    def ekle(self, kare_no: int, kutular) -> List[int]:
        """Kare NUMARASIYLA ekler (video: zaman = kare / fps)."""
        return self.ekle_zamanli(self._zaman(kare_no), kutular, kare_no)

    def ekle_zamanli(self, zaman: float, kutular, kare_no: int = 0) -> List[int]:
        """Bir karenin kutularını işler; her kutunun iz kimliğini döner.

        CANLI AKIŞTA zaman doğrudan verilir: kareler düzensiz aralıklarla
        gelir (ağ gecikmesi, telefon gücü), kare numarası zaman anlamına
        gelmez. Arama penceresi süreye bağlı olduğu için gerçek zaman şart.

        Eşleştirme AÇGÖZLÜ: en iyi puanlı çift önce bağlanır. Macar
        algoritması daha iyi olurdu ama kare başına nesne sayısı küçük
        (onlarca) ve fark ölçülebilir değil; basitlik burada kazanıyor.
        """
        self._son_zaman = max(self._son_zaman, zaman)
        acik = self._aday_izler(zaman)
        atanan: Dict[int, int] = {}          # kutu indeksi → kimlik
        kullanilan = set()

        # Puan tablosu: yalnızca AYNI SINIF eşleşebilir. Farklı sınıfları
        # birleştirmek "olgunlaşmamış çilek olgunlaştı" gibi yanlış bir
        # süreklilik üretirdi.
        adaylar = []
        for ki, kutu in enumerate(kutular):
            for iz in acik:
                if iz.sinif_adi != kutu.sinif_adi:
                    continue
                gecen = max(zaman - iz.son_zaman, 1.0 / self.fps)
                # Arama penceresi süreyle büyür: 2 saniye önce görülen bir
                # nesne daha uzağa gitmiş olabilir.
                izin_verilen = self.hiz * gecen
                d = _mesafe(iz.kutu, kutu)
                ortusme = _iou(iz.kutu, kutu)
                if ortusme >= self.iou_esigi or d <= izin_verilen:
                    # Örtüşme daha güvenilir; mesafe ikincil ölçüt.
                    puan = ortusme + max(0.0, 1.0 - d / max(izin_verilen, 1e-6)) * 0.5
                    adaylar.append((puan, ki, iz))

        for _, ki, iz in sorted(adaylar, key=lambda x: -x[0]):
            if ki in atanan or iz.kimlik in kullanilan:
                continue
            atanan[ki] = iz.kimlik
            kullanilan.add(iz.kimlik)
            iz.onceki_kutu = iz.kutu          # çizgi geçişi bunu kullanır
            iz.kutu = kutular[ki]
            iz.son_zaman = zaman
            iz.gorulme += 1
            iz.en_yuksek_guven = max(iz.en_yuksek_guven, kutular[ki].guven)
            iz.kareler.append(kare_no)
            if self.cizgi is not None:
                self.cizgi.kontrol(iz)

        # Eşleşmeyenler yeni nesnedir
        kimlikler = []
        for ki, kutu in enumerate(kutular):
            if ki in atanan:
                kimlikler.append(atanan[ki])
                continue
            iz = Iz(kimlik=self._sonraki_kimlik, sinif_adi=kutu.sinif_adi,
                    ilk_zaman=zaman, son_zaman=zaman, kutu=kutu, ilk_kutu=kutu,
                    en_yuksek_guven=kutu.guven, kareler=[kare_no])
            self._sonraki_kimlik += 1
            self.izler.append(iz)
            kimlikler.append(iz.kimlik)
        return kimlikler

    # ------------------------------------------------------------------
    def benzersiz_sayim(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for iz in self.izler:
            out[iz.sinif_adi] = out.get(iz.sinif_adi, 0) + 1
        return out

    @property
    def benzersiz_toplam(self) -> int:
        return len(self.izler)

    # ------------------------------------------------------- hareket ölçümü
    # Çizgi sayımı YALNIZCA kamera ilerlerken anlamlıdır; sabit çekimde
    # hiçbir şey çizgiyi geçmez ve sayı 0 kalır. Kullanıcıya "video mu sabit
    # mi" diye SORMAK yerine ölçüyoruz: izler tutarlı biçimde kayıyorsa
    # kamera ilerliyordur.
    HAREKET_ESIGI = 0.04          # saniyede kadrajın %4'ü

    @property
    def ortalama_kayma(self) -> float:
        """İzlerin saniyede ortalama kayması (kadraj oranı).

        Yalnızca en az iki kez görülmüş izler sayılır: tek karelik izin
        kayması ölçülemez.
        """
        hizlar = []
        for iz in self.izler:
            if iz.gorulme < 2 or iz.ilk_kutu is None or iz.sure <= 0:
                continue
            yol = _mesafe(iz.ilk_kutu, iz.kutu)
            hizlar.append(yol / iz.sure)
        return sum(hizlar) / len(hizlar) if hizlar else 0.0

    def hareketli_mi(self, esik: float = None) -> bool:
        return self.ortalama_kayma >= (self.HAREKET_ESIGI if esik is None else esik)

    def sayim_onerisi(self) -> dict:
        """Bu çekimde HANGİ sayım geçerli? Ölçümle karar verilir.

        sabit    → benzersiz iz sayımı doğrudur; çizgi sayımı 0 verir
        hareketli→ çizgi sayımı da anlamlıdır (varsa); uzun taramada iz
                   sayımı, iz kopmalarından ötürü şişebilir
        """
        hareketli = self.hareketli_mi()
        onerilen = 'cizgi' if (hareketli and self.cizgi is not None) else 'benzersiz'
        return {
            'kamera': 'hareketli' if hareketli else 'sabit',
            'kayma': round(self.ortalama_kayma, 4),
            'esik': self.HAREKET_ESIGI,
            'onerilen': onerilen,
            'benzersiz': self.benzersiz_toplam,
            'cizgi': self.cizgi.toplam if self.cizgi is not None else None,
        }

    def ozet(self) -> dict:
        """Kaydedilebilir/şablona verilebilir biçim."""
        out = {
            'benzersiz': self.benzersiz_toplam,
            'sinif': self.benzersiz_sayim(),
            'fps': round(self.fps, 2),
            'hiz': self.hiz,
            'iou': self.iou_esigi,
        }
        out['oneri'] = self.sayim_onerisi()
        if self.cizgi is not None:
            out['cizgi'] = self.cizgi.ozet()
        return out


def ornekleme_adimi(fps: float, aralik_sn: float, en_az: int = 1) -> int:
    """Kaç karede bir örneklensin? SÜREYE göre hesaplanır.

    NEDEN? Sabit kare adımı (her 15. kare) 30 fps'te 0,5 sn, 60 fps'te
    0,25 sn demektir — aynı ayar farklı videolarda farklı davranır ve
    takip penceresi de kayar. Süre sabitlenirse davranış her videoda aynı.
    """
    if not fps or fps <= 0:
        fps = 30.0
    return max(en_az, int(round(fps * max(aralik_sn, 0.01))))
