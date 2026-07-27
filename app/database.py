"""Veritabanı modelleri ve oturum yönetimi.

SQLAlchemy kullanılır; böylece SQLite → PostgreSQL geçişi yalnızca
DATABASE_URL değişikliğiyle yapılır, kod değişmez.
"""

from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine, func)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app import config

Base = declarative_base()

engine = create_engine(
    config.DATABASE_URL,
    connect_args={'check_same_thread': False} if config.DATABASE_URL.startswith('sqlite') else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def simdi():
    """Zaman damgası — UTC saklanır, arayüzde yerel saate çevrilir."""
    return datetime.now(timezone.utc)


class Kamera(Base):
    """Kayıtlı IP kamera. İsteğe bağlı anlık çekim için kullanılır."""
    __tablename__ = 'kameralar'

    id = Column(Integer, primary_key=True)
    ad = Column(String(120), nullable=False)
    url = Column(Text, nullable=False)          # rtsp://... veya http://.../snapshot
    konum = Column(String(200), default='')     # "Sera 1 - Kuzey blok"
    aktif = Column(Boolean, default=True)
    olusturma = Column(DateTime, default=simdi)

    analizler = relationship('Analiz', back_populates='kamera')


class Analiz(Base):
    """Tek bir analiz kaydı (fotoğraf, video veya kamera çekimi)."""
    __tablename__ = 'analizler'

    id = Column(Integer, primary_key=True)
    zaman = Column(DateTime, default=simdi, index=True)

    kaynak_tip = Column(String(20), index=True)   # foto | video | kamera
    kaynak_ad = Column(String(255), default='')   # dosya adı veya kamera adı
    kamera_id = Column(Integer, ForeignKey('kameralar.id'), nullable=True)

    dosya_yolu = Column(Text, default='')         # yüklenen orijinal (göreli)
    sonuc_yolu = Column(Text, default='')         # kutulanmış görsel (göreli)

    tespit_sayisi = Column(Integer, default=0)
    min_guven = Column(Float, default=0.0)
    ort_guven = Column(Float, default=0.0)
    islenen_kare = Column(Integer, default=1)     # video için örneklenen kare sayısı
    sure_ms = Column(Integer, default=0)

    # Düşük güvenli veya tespitsiz kayıtlar uzman incelemesine düşer
    inceleme_gerekli = Column(Boolean, default=False, index=True)
    incelendi = Column(Boolean, default=False, index=True)

    not_ = Column('not', Text, default='')

    tespitler = relationship('Tespit', back_populates='analiz',
                             cascade='all, delete-orphan')
    kamera = relationship('Kamera', back_populates='analizler')

    @property
    def ozet(self):
        """'Gray Mold x2, Leaf Spot x1' biçiminde kısa özet."""
        sayac = {}
        for t in self.tespitler:
            sayac[t.sinif_adi] = sayac.get(t.sinif_adi, 0) + 1
        if not sayac:
            return 'tespit yok'
        return ', '.join(f'{k} x{v}' for k, v in sorted(sayac.items(), key=lambda x: -x[1]))


class Tespit(Base):
    """Analizdeki tek bir kutu."""
    __tablename__ = 'tespitler'

    id = Column(Integer, primary_key=True)
    analiz_id = Column(Integer, ForeignKey('analizler.id'), index=True)

    sinif_id = Column(Integer)
    sinif_adi = Column(String(80), index=True)
    guven = Column(Float)

    # YOLO normalize koordinatlar — düzeltme/dışa aktarım için saklanır
    x = Column(Float)
    y = Column(Float)
    w = Column(Float)
    h = Column(Float)
    kare = Column(Integer, default=0)     # video ise kare numarası

    analiz = relationship('Analiz', back_populates='tespitler')


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI bağımlılığı."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
