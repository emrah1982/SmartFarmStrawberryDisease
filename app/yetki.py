"""Kimlik doğrulama ve veri izolasyonu için TEK geçiş noktası.

DURUM: Şu an giriş sistemi yoktur — `aktif_kullanici()` None döner ve tüm
veriler görünür. Yerel ağda tek işletme için tasarlanan mevcut kullanım budur.

NEDEN BÖYLE BİR DOSYA VAR?
    Çok müşterili kullanıma (her üretici yalnızca kendi verisini görür)
    geçildiğinde YALNIZCA bu dosya değişir. Rotalar veriye doğrudan değil
    buradaki yardımcılar üzerinden eriştiği için izolasyon otomatik uygulanır;
    sayfaları tek tek elden geçirmek gerekmez.

    Bu ayrımı sonradan kurmak, her sorguyu bulup filtre eklemek demektir —
    bir tanesini atlamak bir müşterinin verisini başkasına gösterir.

GEÇİŞ ADIMLARI (bkz. docs/YOL-HARITASI.md):
    1. Kullanıcı tablosu zaten hazır (database.Kullanici)
    2. `aktif_kullanici()` oturumdan kullanıcıyı okusun
    3. Giriş/çıkış sayfaları ve parola doğrulama eklensin
    Rotalarda başka değişiklik gerekmez.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.database import Analiz, Kamera, Kullanici, Sera, Uretici


def aktif_kullanici(db: Session) -> Optional[Kullanici]:
    """Oturumdaki kullanıcı. Giriş sistemi eklenene kadar None.

    None = kimlik doğrulama kapalı; tüm veriler görünür.
    """
    return None


def _admin_mi(kullanici: Optional[Kullanici]) -> bool:
    return kullanici is None or kullanici.rol == 'admin'


def sera_kapsami(db: Session, kullanici: Optional[Kullanici]) -> Optional[List[int]]:
    """Kullanıcının görebileceği sera id'leri. None = sınırsız (tümü)."""
    if _admin_mi(kullanici):
        return None
    return [s.id for s in db.query(Sera).filter(Sera.uretici_id == kullanici.uretici_id)]


def analiz_sorgusu(db: Session, kullanici: Optional[Kullanici]):
    """Kullanıcının görebileceği analizler için hazır sorgu."""
    q = db.query(Analiz)
    kapsam = sera_kapsami(db, kullanici)
    if kapsam is not None:
        # Boş liste durumunda hiçbir kayıt dönmemeli
        q = q.filter(Analiz.sera_id.in_(kapsam or [0]))
    return q


def gorunur_seralar(db: Session, kullanici: Optional[Kullanici]) -> List[Sera]:
    q = db.query(Sera).filter(Sera.aktif == True)  # noqa: E712
    kapsam = sera_kapsami(db, kullanici)
    if kapsam is not None:
        q = q.filter(Sera.id.in_(kapsam or [0]))
    return q.all()


def gorunur_kameralar(db: Session, kullanici: Optional[Kullanici],
                      yalniz_aktif: bool = True) -> List[Kamera]:
    q = db.query(Kamera)
    if yalniz_aktif:
        q = q.filter(Kamera.aktif == True)  # noqa: E712
    kapsam = sera_kapsami(db, kullanici)
    if kapsam is not None:
        q = q.filter(Kamera.sera_id.in_(kapsam or [0]))
    return q.all()


def gorunur_ureticiler(db: Session, kullanici: Optional[Kullanici]) -> List[Uretici]:
    q = db.query(Uretici).filter(Uretici.aktif == True)  # noqa: E712
    if not _admin_mi(kullanici):
        q = q.filter(Uretici.id == kullanici.uretici_id)
    return q.all()


def erisebilir_mi(db: Session, kullanici: Optional[Kullanici], analiz: Analiz) -> bool:
    """Tek bir kaydın görüntülenmesine izin var mı?"""
    if _admin_mi(kullanici):
        return True
    return analiz.sera_id in (sera_kapsami(db, kullanici) or [])
