"""Canlı kamera modülü — telefon/bilgisayar kamerasından anlık tespit.

NE YAPAR?
    Tarayıcı kameradan kare alır, sunucuya gönderir, dönen kutuları görüntünün
    üzerine çizer. Kullanıcı yürürken ekranda hastalıkları anında görür.

NEDEN AYRI MODÜL?
    Canlı akış, çekirdek "dosya yükle → analiz et → kaydet" akışından farklı
    çalışır (WebSocket, geri basınç, otomatik kayıt kuralı). Aynı dosyaya
    karışsaydı ikisi de kırılgan olurdu. Bu klasör silinse uygulama çalışmaya
    devam eder; menüden de kendiliğinden kalkar.

DOSYA DÜZENİ (her biri tek işten sorumlu)
    ayarlar.py   → eşikler/parametreler (ortam değişkenleriyle değiştirilir)
    servis.py    → saf mantık: kare çözme, tespit, otomatik kayıt kararı
    depo.py      → diske yazma + veritabanı kaydı (tek DB temas noktası)
    rotalar.py   → HTTP sayfası, WebSocket, REST yedeği
    static/      → tarayıcı bileşenleri: kamera.js, akis.js, cizim.js
"""

from pathlib import Path

from app.moduller.canli.rotalar import router


def modul():
    from app.moduller import Modul
    return Modul(ad='canli', baslik='Canlı', yol='/canli', router=router,
                 grup='ana', ikon='🔴',
                 statik=str(Path(__file__).parent / 'static'))


__all__ = ['modul', 'router']
