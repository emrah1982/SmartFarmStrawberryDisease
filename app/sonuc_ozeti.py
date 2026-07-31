"""Analiz sonucunu ORGANA göre gruplar — sonuç ekranının veri katmanı.

NEDEN AYRI MODÜL?
    Gerçek sahnede yaprak, çiçek ve meyve bir aradadır. Boru hattı bunu
    zaten çözüyor (her organ kendi uzmanına gidiyor) ama sonuç ekranı
    tespitleri yalnızca SINIF ADINA göre gruplayınca organ bilgisi
    kayboluyordu. Somut sonucu:

        Gray Mold    5 adet    %81     ← 3'ü yaprakta, 2'si meyvede

    "Gray Mold" hem yaprak hem meyve modelinde tanımlı. Meyvedeki kurşuni
    küf acil hasat/imha, yapraktaki havalandırma demektir — kullanıcı
    hangisi olduğunu göremiyordu.

    Mantık burada saf fonksiyonlar halinde durur: veritabanı, FastAPI ve
    şablon bilmez. Böylece testi kolaydır ve sonuç ekranı büyüdükçe
    main.py şişmez.

İKİ AYRI KAYNAK, İKİSİ DE GEREKLİ
    tespitler → NE BULUNDU
    boru_izi  → NE GÖRÜLDÜ (tespit üretmeyen organlar dahil)

    İkisi olmadan "5 yaprak gördüm, hastalık bulmadım" ile "hiç yaprak
    görmedim" ayırt edilemez. Kullanıcı için bu fark kritiktir: ikincisinde
    yaprakları henüz kontrol etmemişizdir.
"""

import json
from dataclasses import dataclass, field
from typing import List

# Organ adları organ modelinin dataset'inden gelir (Flower/Fruit/Leaf).
# Görünüm bilgisi (simge, çoğul başlık) yalnızca sunuma aittir; sınıf
# kütüğüne değil buraya yazılır.
ORGAN_GORUNUM = {
    'leaf':   {'simge': '🌿', 'baslik': 'Yapraklarda', 'tekil': 'yaprak'},
    'fruit':  {'simge': '🍓', 'baslik': 'Meyvelerde', 'tekil': 'meyve'},
    'flower': {'simge': '🌸', 'baslik': 'Çiçeklerde', 'tekil': 'çiçek'},
    'stem':   {'simge': '🌱', 'baslik': 'Gövdede', 'tekil': 'gövde'},
}
BILINMEYEN = {'simge': '📄', 'baslik': 'Organ ayrımı yapılmadan',
              'tekil': 'bölge'}

# Aciliyet sıralaması — en acil grup en üstte gösterilir.
ACILIYET_SIRASI = {'yuksek': 0, 'orta': 1, 'bilgi': 2, '': 3}


def gorunum(organ: str) -> dict:
    return ORGAN_GORUNUM.get((organ or '').lower(), BILINMEYEN)


@dataclass
class SinifSatiri:
    ad: str                      # eğitimdeki ad (Gray Mold)
    adet: int = 0
    max_guven: float = 0.0
    tedavi: dict = field(default_factory=dict)
    # Öneri metni bu organ için ÖZELLEŞTİRİLDİ mi? Arayüzde rozet olarak
    # gösterilir; kullanıcı genel metni mi organa özel metni mi okuduğunu bilsin.
    organa_ozel: bool = False

    @property
    def aciliyet(self) -> str:
        return (self.tedavi or {}).get('aciliyet', '')


@dataclass
class OrganGrubu:
    organ: str                   # '' = organ bilgisi olmayan (miras/elle)
    gorulen: int = 0             # organ modelinin bulduğu adet
    siniflar: List[SinifSatiri] = field(default_factory=list)
    not_: str = ''               # 'değerlendirilmedi' gibi açıklama

    @property
    def simge(self) -> str:
        return gorunum(self.organ)['simge']

    @property
    def baslik(self) -> str:
        return gorunum(self.organ)['baslik']

    @property
    def tekil(self) -> str:
        """'yaprak' / 'meyve' — cümle içinde kullanılacak biçim."""
        return gorunum(self.organ)['tekil']

    @property
    def tespit_sayisi(self) -> int:
        return sum(s.adet for s in self.siniflar)

    @property
    def aciliyet(self) -> str:
        """Gruptaki en acil bulgunun aciliyeti — sıralama ve renk için."""
        if not self.siniflar:
            return ''
        return min((s.aciliyet for s in self.siniflar),
                   key=lambda a: ACILIYET_SIRASI.get(a, 3))


@dataclass
class Ozet:
    gruplar: List[OrganGrubu] = field(default_factory=list)
    organ_sayilari: dict = field(default_factory=dict)
    kontrol_edilmeyen: List[str] = field(default_factory=list)
    modeller: List[str] = field(default_factory=list)
    miras: bool = False

    @property
    def sahne_var(self) -> bool:
        """Organ tespiti yapıldı mı? Yapılmadıysa sahne özeti gösterilmez."""
        return bool(self.organ_sayilari)

    @property
    def sahne_metni(self) -> str:
        return ' · '.join(
            f"{gorunum(o)['simge']} {n} {gorunum(o)['tekil']}"
            for o, n in sorted(self.organ_sayilari.items(), key=lambda x: -x[1]))


def izi_coz(ham) -> dict:
    """boru_izi sütunu JSON metnidir; bozuksa akış kesilmemeli."""
    if isinstance(ham, dict):
        return ham
    if not ham:
        return {}
    try:
        d = json.loads(ham)
        return d if isinstance(d, dict) else {}
    except (ValueError, TypeError):
        return {}


def _kontrol_edilmeyenler(organ_sayilari: dict, urun=None) -> List[str]:
    """Bu karede HANGİ kontroller yapılamadı?

    Kullanıcı "hiçbir şey çıkmadı" ile "bakılmadı"yı karıştırmamalı. Karede
    meyve yoksa olgunluk ve meyve hastalığı modelleri hiç çalışmamıştır;
    bunu söylemezsek kullanıcı meyvelerinin sağlıklı olduğunu sanır.
    """
    from app import modeller

    gorulen = {o.lower() for o in organ_sayilari}
    eksik = []
    for t in modeller.tanimlar(urun).values():
        if not t.aktif or not t.tetik:
            continue                       # tekil/organ/miras modelleri atla
        tetikler = {x.lower() for x in t.tetik}
        if tetikler & gorulen:
            continue                       # çalıştı
        organlar = ', '.join(sorted(gorunum(x)['tekil'] for x in tetikler))
        durum = 'model henüz eğitilmedi' if not t.var else 'bu karede görülmedi'
        eksik.append(f'{t.aciklama.split(".")[0].strip() or t.ad} — '
                     f'{organlar} {durum}')
    return eksik


def kur(tespitler, boru_izi=None, tedavi_kutugu=None, urun=None) -> Ozet:
    """Tespitleri organa göre gruplar.

    tespitler       : `.sinif_adi`, `.guven`, `.organ` alanı olan nesneler
                      (veritabanı Tespit satırı ya da detector.Kutu)
    boru_izi        : Analiz.boru_izi (JSON metni) veya sözlük
    tedavi_kutugu   : {sinif_adi: {...}} tedavi önerileri
    """
    iz = izi_coz(boru_izi)
    organ_sayilari = {k: int(v) for k, v in (iz.get('organlar') or {}).items()}
    tedavi_kutugu = tedavi_kutugu or {}

    from app import tedavi as tedavi_modulu

    # Organ → sınıf → satır
    kova = {}
    for t in tespitler:
        organ = getattr(t, 'organ', '') or ''
        g = kova.setdefault(organ, {})
        s = g.get(t.sinif_adi)
        if s is None:
            # Öneri ORGANA göre çözülür: aynı sınıf yaprakta ve meyvede
            # farklı belirti/aciliyet taşıyabilir (bkz. app/tedavi.py).
            s = SinifSatiri(
                ad=t.sinif_adi,
                tedavi=tedavi_modulu.coz(tedavi_kutugu, t.sinif_adi, organ),
                organa_ozel=bool(organ) and tedavi_modulu.organa_ozel_mi(
                    tedavi_kutugu, t.sinif_adi))
            g[t.sinif_adi] = s
        s.adet += 1
        s.max_guven = max(s.max_guven, float(t.guven or 0))

    # Tespit üretmeyen ama GÖRÜLEN organlar da grup açar — "baktım, temiz".
    for organ in organ_sayilari:
        kova.setdefault(organ, {})

    gruplar = []
    for organ, siniflar_ in kova.items():
        satirlar = sorted(siniflar_.values(), key=lambda s: (-s.adet, -s.max_guven))
        grup = OrganGrubu(organ=organ, gorulen=organ_sayilari.get(organ, 0),
                          siniflar=satirlar)
        if not satirlar and grup.gorulen:
            from app import modeller
            if modeller.tetiklenen(organ, urun):
                grup.not_ = 'Bakıldı, bulgu yok'
            else:
                grup.not_ = 'Bu organ için uzman model yok — değerlendirilmedi'
        gruplar.append(grup)

    # En acil grup üstte; eşitse en çok tespit içeren
    gruplar.sort(key=lambda g: (ACILIYET_SIRASI.get(g.aciliyet, 3),
                                -g.tespit_sayisi, -g.gorulen))

    return Ozet(gruplar=gruplar, organ_sayilari=organ_sayilari,
                kontrol_edilmeyen=_kontrol_edilmeyenler(organ_sayilari, urun),
                modeller=list(iz.get('modeller') or []),
                miras=bool(iz.get('miras')))
