"""Konum modülünün kendi tablosu.

Çekirdek `Analiz` tablosuna sütun EKLENMEZ: konum bilgisi ayrı tabloda tutulur.
Böylece modül kapatıldığında çekirdek şema etkilenmez ve modül başka bir
projeye taşınırken tek klasör yeterli olur.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import backref, relationship

from app.database import Base


class AnalizKonum(Base):
    """Bir analizin nerede çekildiği.

    İki tür konum desteklenir ve ikisi birlikte de kullanılabilir:
      - GPS (enlem/boylam): telefon EXIF'i, drone görüntüsü veya kameranın
        sabit koordinatı
      - Mantıksal konum (blok/sıra): serada GPS hassasiyeti yetersiz kalır;
        "A blok, 3. sıra" gibi bir adres pratikte daha kullanışlıdır
    """
    __tablename__ = 'analiz_konumlari'

    id = Column(Integer, primary_key=True)
    analiz_id = Column(Integer, ForeignKey('analizler.id', ondelete='CASCADE'),
                       index=True, unique=True)

    enlem = Column(Float, nullable=True)
    boylam = Column(Float, nullable=True)
    yukseklik = Column(Float, nullable=True)      # drone irtifası (m)

    blok = Column(String(60), default='')         # "A blok", "Kuzey"
    sira = Column(String(30), default='')         # "3", "3-B"

    kaynak = Column(String(20), default='')       # exif | kamera | elle
    olusturma = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # uselist=False backref'e verilmeli: aksi halde analiz.konum bir LİSTE döner
    # ve şablonlarda a.konum.etiket çalışmaz.
    analiz = relationship('Analiz', backref=backref('konum', uselist=False))

    @property
    def gps_var(self) -> bool:
        return self.enlem is not None and self.boylam is not None

    @property
    def etiket(self) -> str:
        """İnsan tarafından okunur konum: 'A blok / 3' veya koordinat."""
        if self.blok or self.sira:
            return ' / '.join(x for x in (self.blok, self.sira) if x)
        if self.gps_var:
            return f'{self.enlem:.5f}, {self.boylam:.5f}'
        return 'konum yok'


def tablolar_olustur(engine):
    Base.metadata.create_all(engine, tables=[AnalizKonum.__table__])
