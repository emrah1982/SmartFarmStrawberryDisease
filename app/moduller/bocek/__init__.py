"""Böcek teşhis modülü — ayrı akış, hastalık boru hattına girmez.

NE YAPAR?
    Kullanıcı bulduğu böceğin YAKIN ÇEKİM fotoğrafını çeker, model türünü
    söyler ve mücadele önerisi gösterilir.

NEDEN ÇEKİRDEĞE DEĞİL MODÜLE?
    Fotoğraf analizi akışı "bitkiye bak, hastalık bul" der; bu akış
    "elimdeki böcek ne" der. İkisi aynı sayfada birleşirse kullanıcı hangi
    fotoğrafı nereye vereceğini karıştırır. Modül olarak durunca kendi
    rotası, şablonu ve menü girdisi olur; istenmezse yuklu_moduller()
    listesinden çıkarılarak kapatılır.

SINIRI AÇIKÇA SÖYLER
    Model KAPALI KÜMEDİR: yalnızca 6 tür bilir ve "bilmiyorum" diyemez.
    Arayüz bu yüzden tek cevap değil ilk 3 adayı gösterir, bildiği türleri
    listeler ve kararsızlık durumunu ayrıca uyarır. Ayrıntı: servis.py
"""

from app.moduller.bocek.modeller import tablolar_olustur
from app.moduller.bocek.rotalar import router


def modul():
    from app.moduller import Modul
    return Modul(
        ad='bocek',
        baslik='Böcek Teşhis',
        yol='/bocek',
        grup='ana', ikon='🐛',
        router=router,
        tablolar_olustur=tablolar_olustur,
    )


__all__ = ['modul', 'router', 'tablolar_olustur']
