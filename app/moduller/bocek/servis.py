"""Böcek teşhis modelinin çalıştırılması — saf mantık, HTTP bilmez.

NEDEN AYRI AKIŞ?
    Bu model 416x416 MAKRO böcek fotoğraflarıyla eğitildi; kutu alanı
    medyanı karenin %14,8'i. Hiyerarşik boru hattı ise saha görüntüsünden
    kırpılmış yaprak/meyve parçası verir — orada bir zararlı kırpıntının
    %1'inden azını kaplar. İki alan birbirine benzemez.

    Bu yüzden kütükte `tetik: []` ile durur ve ROI akışına HİÇ girmez
    (bkz. configs/urunler/<urun>/modeller.yaml, tests/test_tekil_model.py).
    Buradan açıkça çağrılır.

KAPALI KÜME SORUNU — ARAYÜZÜN ASIL İŞİ
    Model yalnızca 6 tür bilir ve "bilmiyorum" diyemez: gördüğü her böceği
    bu 6'dan birine sokar. Kullanıcı yaprak biti fotoğrafı çekerse
    "Toprak Larvası %70" cevabı alabilir ve buna güvenirse yanlış mücadele
    yapar.

    Bu yüzden tek bir cevap DÖNMÜYORUZ: ilk 3 aday güvenleriyle birlikte
    verilir ve aralarındaki fark küçükse "kararsız" işaretlenir. Kullanıcı
    modelin ne kadar emin olduğunu görerek karar verir.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

MODEL_ADI = 'bocek_teshis'

# Bu değerin altındaki hiçbir aday gösterilmez — model her karede bir şey
# bulmaya çalışır, düşük güvenli çöp tespitler kullanıcıyı yanıltır.
EN_DUSUK_GUVEN = 0.20

# İlk iki aday birbirine bu kadar yakınsa model KARARSIZDIR. Tek cevap
# göstermek burada yanıltıcı olur.
KARARSIZLIK_FARKI = 0.15


@dataclass
class Aday:
    ad: str                  # eğitimdeki ad (Mole Cricket)
    guven: float
    # Kutunun konumu (0-1 normalize). Kullanıcı böceği görüntünün NERESİNDE
    # bulunduğunu görmeli — özellikle karede birden çok canlı varsa "bu mu
    # yoksa şu mu" sorusu ancak kutuyla cevaplanır.
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    sinif_id: int = 0

    @property
    def yuzde(self) -> int:
        return int(round(self.guven * 100))

    @property
    def kutu_var(self) -> bool:
        return self.w > 0 and self.h > 0


@dataclass
class Sonuc:
    # TÜM tespitler — karede kaç birey varsa o kadar kutu.
    #
    # NEDEN AYRI? Önce yalnızca tür başına en iyi kutu tutuluyordu ("bu ne"
    # sorusuna cevap yeter diye). Ama 3 tırtıllı bir karede tek kutu
    # çiziliyordu ve kullanıcı diğerlerini göremiyordu. Zararlıda SAYI da
    # tarımsal bilgidir: 1 birey ile 20 birey aynı şey değildir.
    kutular: List[Aday] = field(default_factory=list)
    # Tür listesi (tablo için): tür başına EN İYİ kutu, güvene göre sıralı.
    adaylar: List[Aday] = field(default_factory=list)
    kararsiz: bool = False
    hata: str = ''

    @property
    def bulundu(self) -> bool:
        return bool(self.adaylar)

    @property
    def en_iyi(self) -> Optional[Aday]:
        return self.adaylar[0] if self.adaylar else None

    def sayim(self) -> dict:
        """{tür: kaç birey} — arayüz 'Bozkurt Tırtılı ×3' yazabilsin."""
        out = {}
        for k in self.kutular:
            out[k.ad] = out.get(k.ad, 0) + 1
        return out

    def adet(self, ad: str) -> int:
        return self.sayim().get(ad, 0)


def _kutu_coz(b):
    """Ultralytics kutusundan normalize (x, y, w, h).

    `xywhn` gerçek çalışmada bir tensördür, testte düz liste olabilir;
    ikisini de kabul ediyoruz. Okunamazsa sıfır döner ve kutu çizilmez —
    tür bilgisi yine de gösterilir, çizim eksikliği akışı kesmemeli.
    """
    try:
        ham = b.xywhn[0]
        ham = ham.tolist() if hasattr(ham, 'tolist') else list(ham)
        x, y, w, h = (float(v) for v in ham[:4])
        return x, y, w, h
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def hazir(urun=None) -> bool:
    """Model dosyası kurulu mu?"""
    from app import modeller
    t = modeller.tanim(MODEL_ADI, urun)
    return bool(t and t.aktif and t.var)


def taniyabildikleri(urun=None) -> List[str]:
    """Modelin bildiği türler — arayüz bunu AÇIKÇA göstermeli.

    Kapalı küme olduğu için kullanıcı listeyi görmeden fotoğraf çekerse,
    listede olmayan bir böcek için kendinden emin ama yanlış cevap alır.
    """
    from app import modeller
    t = modeller.tanim(MODEL_ADI, urun)
    return list(t.siniflar) if t else []


def tani(goruntu, urun=None, imgsz: int = 416) -> Sonuc:
    """Bir kareyi modelden geçirip aday listesi döner.

    goruntu : BGR numpy dizisi (cv2.imread / imdecode çıktısı)
    """
    from app import modeller

    model = modeller.yukle(MODEL_ADI, urun)
    if model is None:
        return Sonuc(hata='Model kurulu değil.')

    try:
        r = model(goruntu, conf=EN_DUSUK_GUVEN, imgsz=imgsz, verbose=False)[0]
    except Exception as e:
        logger.warning(f'Böcek teşhisi başarısız: {e}')
        return Sonuc(hata=f'Görüntü işlenemedi ({type(e).__name__}).')

    # HER kutu saklanır: karede birden çok birey olabilir ve kullanıcı
    # hepsini görmeli. Tür listesi bunlardan türetilir.
    adlar = r.names
    kutular = []
    for b in r.boxes:
        cid = int(b.cls[0])
        ad = adlar.get(cid, str(cid)) if isinstance(adlar, dict) else adlar[cid]
        kutular.append(Aday(ad, float(b.conf[0]), *_kutu_coz(b), sinif_id=cid))
    kutular.sort(key=lambda a: -a.guven)

    # Tür listesi: tür başına EN İYİ kutu. Aynı tür üç kez çıktıysa tabloda
    # üç satır olmamalı — soru "hangi türler var", "kaç kutu var" değil.
    en_iyi = {}
    for k in kutular:
        if k.guven > getattr(en_iyi.get(k.ad), 'guven', 0.0):
            en_iyi[k.ad] = k
    adaylar = sorted(en_iyi.values(), key=lambda a: -a.guven)[:3]

    kararsiz = (len(adaylar) >= 2
                and adaylar[0].guven - adaylar[1].guven < KARARSIZLIK_FARKI)
    return Sonuc(kutular=kutular, adaylar=adaylar, kararsiz=kararsiz)


# Bitki analizi boş dönünce çalışan YEDEK teşhis burada DAHA SIKI bir eşik
# kullanır. Sebep: kullanıcı bu fotoğrafı böcek sorusuyla yüklemedi. Model
# kapalı kümedir ve bulanık bir duvar fotoğrafına da "Toprak Larvası %70"
# diyebilir. Kendiliğinden ortaya çıkan bir öneri, kullanıcının açıkça
# sorduğu bir teşhisten daha yüksek çıta gerektirir.
YEDEK_EN_DUSUK_GUVEN = 0.55


def yedek_tani(goruntu, urun=None) -> dict:
    """Bitki analizi hiçbir şey bulamadığında çalışan tamamlayıcı teşhis.

    Boş sözlük döner = gösterilecek bir şey yok. Arayüz bunu ÖNERİ olarak
    sunar, karar olarak değil.
    """
    if not hazir(urun):
        return {}
    s = tani(goruntu, urun)
    if not s.adaylar:
        return {}
    en = s.adaylar[0]
    if en.guven < YEDEK_EN_DUSUK_GUVEN:
        return {}
    return {
        'ad': en.ad,
        'guven': en.guven,
        'kararsiz': s.kararsiz,
        'adet': len(s.kutular),
        'kutu': ({'x': en.x, 'y': en.y, 'w': en.w, 'h': en.h,
                  'sinif_id': en.sinif_id} if en.kutu_var else None),
        # Bütün kutular: yedek akışta da her birey çizilebilsin
        'kutular': [{'ad': k.ad, 'guven': k.guven, 'x': k.x, 'y': k.y,
                     'w': k.w, 'h': k.h, 'sinif_id': k.sinif_id}
                    for k in s.kutular if k.kutu_var],
        'adaylar': [{'ad': a.ad, 'guven': a.guven, 'adet': s.adet(a.ad)}
                    for a in s.adaylar],
    }


def adaylari_ciz(frame, sonuc: 'Sonuc', ad_cevir=None):
    """Teşhis görseli: BULUNAN HER BİREY kutu içine alınır.

    Önce yalnızca en iyi aday çiziliyordu; 3 tırtıllı bir karede tek kutu
    görünüyor, kullanıcı diğerlerini göremiyordu. Zararlıda sayı da bilgidir.

    Etiket "?" ile başlar: model KAPALI KÜMEDİR, kutu "burada bir şey var"
    demektir, "bu kesinlikle odur" demek değil.
    """
    if frame is None or not sonuc:
        return frame
    ciz = [k for k in (sonuc.kutular or []) if k.kutu_var]
    if not ciz:
        return frame

    from app import cizim
    from app.detector import Kutu

    kutular = [Kutu(sinif_id=k.sinif_id, sinif_adi=k.ad, guven=k.guven,
                    x=k.x, y=k.y, w=k.w, h=k.h) for k in ciz]

    def _etiket(ad):
        return '? ' + (ad_cevir(ad) if ad_cevir else ad)

    return cizim.kutulari_ciz(frame, kutular, _etiket)


def kutuyu_ciz(frame, bulgu: dict, ad_cevir=None):
    """Yedek teşhisin kutusunu görüntüye çizer.

    Kutu, bitki tespitleriyle AYNI görsele çizilir ama listeye eklenmez:
    `Tespit` tablosuna yazılsaydı hastalık istatistiklerine karışırdı.
    Etiket "?" ile başlar — bu bir ÖNERİ, kesin teşhis değil.
    """
    bulgu = bulgu or {}
    # Yeni biçimde bütün bireyler `kutular` içinde; eski kayıtlarda tek
    # `kutu` vardı — ikisi de desteklenir ki geçmiş kayıtlar bozulmasın.
    ham = bulgu.get('kutular') or ([dict(bulgu['kutu'], ad=bulgu.get('ad', ''),
                                         guven=bulgu.get('guven', 0))]
                                   if bulgu.get('kutu') else [])
    if frame is None or not ham:
        return frame

    from app import cizim
    from app.detector import Kutu

    kutular = [Kutu(sinif_id=int(k.get('sinif_id', 0)),
                    sinif_adi=k.get('ad') or bulgu.get('ad', ''),
                    guven=float(k.get('guven', 0)),
                    x=float(k['x']), y=float(k['y']),
                    w=float(k['w']), h=float(k['h'])) for k in ham]

    def _etiket(ad):
        return '? ' + (ad_cevir(ad) if ad_cevir else ad)

    return cizim.kutulari_ciz(frame, kutular, _etiket)


def oneri(sinif_adi: str, urun=None) -> dict:
    """Tür için mücadele önerisi.

    Organ verilmez: makro fotoğrafta böcek bitkinin neresinde bilinmez.
    tedavi.coz() bu durumda ORTAK metni döner (organ blokları kullanılmaz).
    """
    from app import tedavi
    return tedavi.coz(tedavi.yukle(urun), sinif_adi)
