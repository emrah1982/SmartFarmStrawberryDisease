"""Aday etiket inceleme modülü.

NE YAPAR?
    Otomatik ön-etiketleme (Grounding DINO / DINOv2 / CLIP) ADAY kutular
    üretir. Bu modül o adayları tarayıcıda gösterir: kutuyu gör, taşı,
    boyutlandır, sınıfını değiştir, sil, yeni ekle, onayla.

NEDEN GEREKLİ?
    Otomatik etiketi doğrudan eğitime vermek bu projede tekrar tekrar
    yakalanan "sessiz hata" desenidir: model hata vermez, yanlışı öğrenir.
    Ölçülen iki örnek: hazelnut ön-etiketlerinde 4310 kutunun HEPSİ aynı
    sabit değerdeydi; cilek/organ_detection görüntülerinin %32.7'sinde tek
    etiket tam-kadraj kutuydu. İkisi de gözle bakılsa hemen görülürdü.

    Bu modül o "gözle bakma" adımını akışın parçası yapar.

KATMAN
    servis.py  → saf mantık, fastapi/db/jinja bilmez, ayrı test edilir
    rotalar.py → HTTP
    static/    → tuval editörü, dış kütüphane YOK (sera internetsiz olabilir)
"""

from app.moduller.etiket.rotalar import router


def modul():
    from app.moduller import Modul
    from pathlib import Path
    return Modul(
        ad='etiket',
        baslik='Etiket İnceleme',
        yol='/etiket',
        grup='model', ikon='🏷️',
        router=router,
        statik=str(Path(__file__).parent / 'static'),
    )


__all__ = ['modul', 'router']
