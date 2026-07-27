"""Modül (bileşen) altyapısı.

NEDEN MODÜLER?
    Konum/haritalama gibi yetenekler çekirdek uygulamaya karışmadan, kendi
    tabloları-rotaları-şablonlarıyla ayrı klasörde durur. Böylece:
      - başka bir projeye tek klasör kopyalanarak taşınabilir,
      - istenmeyen modül `MODULLER` listesinden çıkarılarak kapatılabilir,
      - çekirdek kod (app/main.py) modülün ayrıntısını bilmek zorunda kalmaz.

Her modül şu arayüzü sağlar:
    ad        : kısa kimlik (klasör adı)
    baslik    : menüde görünecek etiket
    yol       : menü bağlantısı
    router    : FastAPI APIRouter
    tablolar_olustur(engine) : (isteğe bağlı) kendi tablolarını kurar
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from fastapi import APIRouter, FastAPI


@dataclass
class Modul:
    ad: str
    baslik: str
    yol: str
    router: APIRouter
    tablolar_olustur: Optional[Callable] = None
    menude: bool = True
    grup: str = 'ayarlar'      # ana | model | ayarlar — menüde nereye düşeceği
    ikon: str = ''


def sablon_ayarla(templates):
    """Modül şablonlarına çekirdeğin filtre ve global'lerini taşır.

    Menü, tarih biçimi gibi ortak şeyler her modülde tekrar tanımlanmasın.
    """
    from app import main as cekirdek
    for ad, f in cekirdek.templates.env.filters.items():
        templates.env.filters.setdefault(ad, f)
    for ad, g in cekirdek.templates.env.globals.items():
        templates.env.globals.setdefault(ad, g)


def yuklu_moduller() -> List[Modul]:
    """Etkin modüller. Kapatmak için ilgili satırı yorumlamak yeterlidir."""
    from app.moduller.konum import modul as konum_modul
    from app.moduller.veritabani import modul as veritabani_modul
    return [konum_modul(), veritabani_modul()]


def kaydet(app: FastAPI, engine=None) -> List[Modul]:
    """Modülleri uygulamaya bağlar; menü için listesini döner."""
    moduller = yuklu_moduller()
    for m in moduller:
        app.include_router(m.router)
        if m.tablolar_olustur and engine is not None:
            m.tablolar_olustur(engine)
    return moduller
