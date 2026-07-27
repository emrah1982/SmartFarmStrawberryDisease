"""Konum modülünün sayfaları ve API'si."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app import config, yetki
from app.database import Analiz, Sera, get_db
from app.moduller.konum import servis
from app.moduller.konum.modeller import AnalizKonum

router = APIRouter(prefix='/konum', tags=['konum'])

# Modül kendi şablonlarını taşır (çekirdek templates/ klasörüne karışmaz), ama
# ortak yerleşim için çekirdeğin base.html'ini de görebilmeli — bu yüzden iki
# dizin birden aranır.
templates = Jinja2Templates(directory=[
    str(Path(__file__).parent / 'templates'),
    str(config.BASE_DIR / 'app' / 'templates'),
])


def _ortak_ayarlar():
    from app.moduller import sablon_ayarla
    sablon_ayarla(templates)


@router.get('/yayginlik', response_class=HTMLResponse)
def yayginlik(request: Request, sera_id: int = 0, db: Session = Depends(get_db)):
    """Hastalık yaygınlığı: nerede yoğunlaşmış?"""
    _ortak_ayarlar()
    kullanici = yetki.aktif_kullanici(db)
    sorgu = yetki.analiz_sorgusu(db, kullanici)
    if sera_id:
        sorgu = sorgu.filter(Analiz.sera_id == sera_id)
    kayitlar = sorgu.order_by(Analiz.id.asc()).all()

    # Panel ile aynı kural: aynı görüntünün son kaydı sayılır
    benzersiz = {}
    for a in kayitlar:
        benzersiz[a.dosya_hash or f'id{a.id}'] = a
    sayilan = list(benzersiz.values())

    bolgeler = servis.yaygınlık_hesapla(sayilan)
    noktalar = servis.gps_noktalari(sayilan)
    konumsuz = sum(1 for a in sayilan if not getattr(a, 'konum', None))

    return templates.TemplateResponse(request, 'konum/yayginlik.html', {
        'request': request, 'bolgeler': bolgeler, 'noktalar': noktalar,
        'konumsuz': konumsuz, 'toplam': len(sayilan),
        'seralar': yetki.gorunur_seralar(db, kullanici), 'sera_id': sera_id,
    })


@router.post('/kayit/{analiz_id}')
def konum_kaydet(analiz_id: int, blok: str = Form(''), sira: str = Form(''),
                 enlem: Optional[str] = Form(None), boylam: Optional[str] = Form(None),
                 db: Session = Depends(get_db)):
    """Bir kaydın konumunu elle belirler/günceller."""
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')
    if not yetki.erisebilir_mi(db, yetki.aktif_kullanici(db), a):
        raise HTTPException(403, 'Bu kayda erişim yetkiniz yok')

    def sayi(x):
        try:
            return float(x) if x not in (None, '') else None
        except ValueError:
            raise HTTPException(400, f'Geçersiz koordinat: {x}')

    k = db.query(AnalizKonum).filter(AnalizKonum.analiz_id == analiz_id).first()
    if not k:
        k = AnalizKonum(analiz_id=analiz_id)
        db.add(k)
    k.blok, k.sira = blok.strip(), sira.strip()
    k.enlem, k.boylam = sayi(enlem), sayi(boylam)
    k.kaynak = 'elle'
    db.commit()
    return RedirectResponse(f'/kayit/{analiz_id}', status_code=303)


def konum_ata(db: Session, analiz: Analiz, dosya: Path, kamera=None) -> Optional[AnalizKonum]:
    """Analiz kaydedilirken konumu otomatik belirler.

    Öncelik: fotoğrafın EXIF GPS'i → kameranın sabit konumu.
    Hiçbiri yoksa konum atanmaz; kullanıcı sonradan elle girebilir.
    """
    kaynak, enlem, boylam, yukseklik, blok, sira = '', None, None, None, '', ''

    gps = servis.exif_gps(dosya) if dosya.exists() else None
    if gps:
        enlem, boylam, yukseklik = gps
        kaynak = 'exif'
    elif kamera is not None and getattr(kamera, 'enlem', None) is not None:
        enlem, boylam = kamera.enlem, kamera.boylam
        blok, sira = getattr(kamera, 'blok', '') or '', getattr(kamera, 'sira', '') or ''
        kaynak = 'kamera'
    elif kamera is not None and (getattr(kamera, 'blok', '') or getattr(kamera, 'sira', '')):
        blok, sira = kamera.blok or '', kamera.sira or ''
        kaynak = 'kamera'

    if not kaynak:
        return None

    k = AnalizKonum(analiz_id=analiz.id, enlem=enlem, boylam=boylam,
                    yukseklik=yukseklik, blok=blok, sira=sira, kaynak=kaynak)
    db.add(k)
    db.commit()
    return k
