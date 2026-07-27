"""Konum ve yaygınlık modülü.

Ne yapar:
  - Fotoğrafın EXIF GPS bilgisini okur (telefon veya drone görüntüsü)
  - IP kameraların sabit konumunu analize taşır
  - Konum elle girilebilir (blok/sıra) — serada GPS hassasiyeti çoğu zaman yetersizdir
  - "Hastalık nerede yoğunlaşmış" sorusunu yanıtlayan yaygınlık sayfası üretir

Kapatmak için: app/moduller/__init__.py içindeki yuklu_moduller() listesinden çıkarın.
"""

from app.moduller.konum.modeller import tablolar_olustur
from app.moduller.konum.rotalar import konum_ata, router


def modul():
    from app.moduller import Modul
    return Modul(
        ad='konum',
        baslik='Harita',
        yol='/konum/yayginlik',
        grup='ana', ikon='🗺️',
        router=router,
        tablolar_olustur=tablolar_olustur,
    )


__all__ = ['modul', 'konum_ata', 'router', 'tablolar_olustur']
