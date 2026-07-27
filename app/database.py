"""Veritabanı modelleri ve oturum yönetimi.

Hiyerarşi:  Üretici → Sera → Kamera → Analiz

Bu ayrım baştan kurulur: "hangi hastalık, kimin serasında, hangi kamerada"
sorusunu sonradan eklemek tüm geçmiş kayıtları sahipsiz bırakırdı.

SQLAlchemy kullanılır; SQLite → PostgreSQL geçişi yalnızca DATABASE_URL
değişikliğidir, kod değişmez.
"""

from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine, inspect, text)
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


class Kullanici(Base):
    """Uygulama kullanıcısı — ŞU AN KULLANILMIYOR, altyapı olarak hazır.

    Giriş sistemi eklendiğinde (bkz. app/yetki.py ve docs/YOL-HARITASI.md):
    - rol='admin'    → tüm üreticilerin verisini görür
    - rol='uretici'  → yalnızca uretici_id'sine bağlı seraların verisini görür

    Tablo baştan tanımlanır ki çok müşterili kullanıma geçiş veritabanı
    göçü gerektirmesin.
    """
    __tablename__ = 'kullanicilar'

    id = Column(Integer, primary_key=True)
    kullanici_adi = Column(String(80), unique=True, nullable=False)
    parola_hash = Column(String(255), default='')
    ad = Column(String(160), default='')
    rol = Column(String(20), default='admin')          # admin | uretici
    uretici_id = Column(Integer, ForeignKey('ureticiler.id'), nullable=True)
    aktif = Column(Boolean, default=True)
    olusturma = Column(DateTime, default=simdi)


class Uretici(Base):
    """Sera sahibi / müşteri. Birden çok serası olabilir."""
    __tablename__ = 'ureticiler'

    id = Column(Integer, primary_key=True)
    ad = Column(String(160), nullable=False)
    telefon = Column(String(40), default='')
    eposta = Column(String(160), default='')
    notlar = Column(Text, default='')
    aktif = Column(Boolean, default=True)
    olusturma = Column(DateTime, default=simdi)

    seralar = relationship('Sera', back_populates='uretici',
                           order_by='Sera.ad')

    @property
    def aktif_seralar(self):
        return [s for s in self.seralar if s.aktif]


class Sera(Base):
    """Tek bir sera/tarla birimi. Bir üreticiye aittir, birden çok kamerası olabilir."""
    __tablename__ = 'seralar'

    id = Column(Integer, primary_key=True)
    uretici_id = Column(Integer, ForeignKey('ureticiler.id'), index=True)

    ad = Column(String(160), nullable=False)          # "Sera 1", "Kuzey blok"
    konum = Column(String(240), default='')           # adres / parsel
    urun = Column(String(120), default='Çilek')
    aktif = Column(Boolean, default=True)
    olusturma = Column(DateTime, default=simdi)

    uretici = relationship('Uretici', back_populates='seralar')
    kameralar = relationship('Kamera', back_populates='sera', order_by='Kamera.ad')
    analizler = relationship('Analiz', back_populates='sera')

    @property
    def tam_ad(self):
        """'Ahmet Yılmaz — Sera 1' biçiminde, listelerde kullanılır."""
        return f'{self.uretici.ad} — {self.ad}' if self.uretici else self.ad

    @property
    def aktif_kameralar(self):
        return [k for k in self.kameralar if k.aktif]


class Kamera(Base):
    """Bir seraya bağlı IP kamera."""
    __tablename__ = 'kameralar'

    id = Column(Integer, primary_key=True)
    sera_id = Column(Integer, ForeignKey('seralar.id'), index=True, nullable=True)

    ad = Column(String(120), nullable=False)          # "Giriş", "3. sıra"
    url = Column(Text, nullable=False)                # rtsp://... veya http://.../snapshot
    konum = Column(String(200), default='')           # sera içindeki yer: "A blok, 3. sıra"
    # Sabit kamera konumu — bu kameradan gelen analizler bu konumu devralır
    # (konum modülü kullanır; modül kapalıysa yalnızca boş dururlar)
    enlem = Column(Float, nullable=True)
    boylam = Column(Float, nullable=True)
    blok = Column(String(60), default='')
    sira = Column(String(30), default='')
    aktif = Column(Boolean, default=True)
    olusturma = Column(DateTime, default=simdi)

    sera = relationship('Sera', back_populates='kameralar')
    analizler = relationship('Analiz', back_populates='kamera')

    @property
    def tam_ad(self):
        """'Ahmet Yılmaz — Sera 1 / Giriş' — hangi kamera, hangi serada, kime ait."""
        return f'{self.sera.tam_ad} / {self.ad}' if self.sera else self.ad


class Analiz(Base):
    """Tek bir analiz kaydı (fotoğraf, video veya kamera çekimi)."""
    __tablename__ = 'analizler'

    id = Column(Integer, primary_key=True)
    zaman = Column(DateTime, default=simdi, index=True)

    kaynak_tip = Column(String(20), index=True)   # foto | video | kamera
    kaynak_ad = Column(String(255), default='')   # dosya adı veya kamera adı

    # Kamera analizlerinde kameradan türetilir; telefon yüklemelerinde kullanıcı seçer.
    # sera_id ayrıca saklanır: kamera silinse/değişse bile kaydın hangi seraya ait
    # olduğu kaybolmasın.
    kamera_id = Column(Integer, ForeignKey('kameralar.id'), nullable=True, index=True)
    sera_id = Column(Integer, ForeignKey('seralar.id'), nullable=True, index=True)

    dosya_yolu = Column(Text, default='')         # yüklenen orijinal (göreli)
    # Görüntü içeriğinin hash'i: aynı fotoğraf birden çok kez yüklenirse
    # eğitim havuzunda kopya oluşmasını engeller
    dosya_hash = Column(String(16), default='', index=True)
    sonuc_yolu = Column(Text, default='')         # kutulanmış görsel (göreli)

    tespit_sayisi = Column(Integer, default=0)
    min_guven = Column(Float, default=0.0)
    ort_guven = Column(Float, default=0.0)
    islenen_kare = Column(Integer, default=1)     # video için örneklenen kare sayısı
    sure_ms = Column(Integer, default=0)

    # Görüntü kalitesi: bulanık kare modele verilirse yanlış tespit üretir
    keskinlik = Column(Float, default=0.0)
    bulanik_kare = Column(Integer, default=0)
    kalite_notu = Column(Text, default='')

    # Düşük güvenli veya tespitsiz kayıtlar uzman incelemesine düşer
    inceleme_gerekli = Column(Boolean, default=False, index=True)
    incelendi = Column(Boolean, default=False, index=True)

    # Uzman kutuları elle düzelttiyse bu kayıt EĞİTİM VERİSİ olarak kullanılabilir
    elle_etiketlendi = Column(Boolean, default=False, index=True)
    disa_aktarildi = Column(Boolean, default=False, index=True)

    not_ = Column('not', Text, default='')

    tespitler = relationship('Tespit', back_populates='analiz',
                             cascade='all, delete-orphan')
    kamera = relationship('Kamera', back_populates='analizler')
    sera = relationship('Sera', back_populates='analizler')

    @property
    def ozet(self):
        """'Gray Mold x2, Leaf Spot x1' biçiminde kısa özet."""
        sayac = {}
        for t in self.tespitler:
            sayac[t.sinif_adi] = sayac.get(t.sinif_adi, 0) + 1
        if not sayac:
            return 'tespit yok'
        return ', '.join(f'{k} x{v}' for k, v in sorted(sayac.items(), key=lambda x: -x[1]))

    @property
    def yer(self):
        """Kaydın nereye ait olduğu: 'Ahmet Yılmaz — Sera 1 / Giriş'."""
        if self.kamera:
            return self.kamera.tam_ad
        if self.sera:
            return self.sera.tam_ad
        return 'atanmamış'


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


def _eksik_sutunlari_ekle():
    """Mevcut veritabanına sonradan eklenen sütunları ekler.

    Hiyerarşi (üretici/sera) sonradan geldiği için eski kayıt dosyalarında bu
    sütunlar yoktur. Veritabanını silmeye gerek kalmadan güncellenir.
    """
    denetci = inspect(engine)
    if 'kameralar' not in denetci.get_table_names():
        return
    yeni = {
        'kameralar': {'sera_id': 'INTEGER', 'enlem': 'FLOAT', 'boylam': 'FLOAT',
                      'blok': 'VARCHAR(60)', 'sira': 'VARCHAR(30)'},
        'analizler': {'sera_id': 'INTEGER', 'keskinlik': 'FLOAT',
                      'bulanik_kare': 'INTEGER', 'kalite_notu': 'TEXT',
                      'elle_etiketlendi': 'BOOLEAN', 'disa_aktarildi': 'BOOLEAN',
                      'dosya_hash': 'VARCHAR(16)'},
    }
    with engine.begin() as baglanti:
        for tablo, sutunlar in yeni.items():
            mevcut = {s['name'] for s in denetci.get_columns(tablo)}
            for ad, tip in sutunlar.items():
                if ad not in mevcut:
                    baglanti.execute(text(f'ALTER TABLE {tablo} ADD COLUMN {ad} {tip}'))


def init_db():
    Base.metadata.create_all(engine)
    _eksik_sutunlari_ekle()


def get_db():
    """FastAPI bağımlılığı."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
