"""Veritabanı görüntüleyici modülü (salt okunur).

Docker içinde çalışırken SQLite dosyasını açmak zahmetlidir; bu modül
tabloları ve satırları tarayıcıdan gösterir. YALNIZCA OKUR: serbest SQL
çalıştırılmaz, yalnızca tanımlı tablolar listelenip sayfalanır.
"""

from app.moduller.veritabani.rotalar import router


def modul():
    from app.moduller import Modul
    return Modul(ad='veritabani', baslik='Veritabanı', yol='/veritabani',
                 router=router)


__all__ = ['modul', 'router']
