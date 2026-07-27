"""Salt okunur veritabanı tarayıcısı."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from app import config
from app.database import Base, get_db

router = APIRouter(prefix='/veritabani', tags=['veritabani'])

templates = Jinja2Templates(directory=[
    str(Path(__file__).parent / 'templates'),
    str(config.BASE_DIR / 'app' / 'templates'),
])


def _ortak_ayarlar():
    from app.moduller import sablon_ayarla
    sablon_ayarla(templates)


def _tablolar():
    """Yalnızca uygulamanın tanımladığı tablolar — serbest erişim yok."""
    return {ad: t for ad, t in Base.metadata.tables.items()}


@router.get('', response_class=HTMLResponse)
def liste(request: Request, db: Session = Depends(get_db)):
    _ortak_ayarlar()
    bilgi = []
    for ad, tablo in sorted(_tablolar().items()):
        try:
            adet = db.execute(select(func.count()).select_from(tablo)).scalar() or 0
        except Exception:
            adet = -1
        bilgi.append({'ad': ad, 'adet': adet,
                      'sutunlar': [c.name for c in tablo.columns]})

    yol = config.DATABASE_URL.replace('sqlite:///', '')
    boyut = Path(yol).stat().st_size / 1024 if Path(yol).exists() else 0
    return templates.TemplateResponse(request, 'veritabani/liste.html', {
        'request': request, 'tablolar': bilgi,
        'db_yolu': yol, 'db_boyut': round(boyut, 1),
        'db_url': config.DATABASE_URL,
    })


@router.get('/{tablo_adi}', response_class=HTMLResponse)
def tablo(tablo_adi: str, request: Request, sayfa: int = 1,
          db: Session = Depends(get_db)):
    _ortak_ayarlar()
    tablolar = _tablolar()
    if tablo_adi not in tablolar:
        raise HTTPException(404, f'Tablo yok: {tablo_adi}')

    t = tablolar[tablo_adi]
    boyut = 50
    sayfa = max(1, sayfa)
    toplam = db.execute(select(func.count()).select_from(t)).scalar() or 0

    birincil = list(t.primary_key.columns)
    sorgu = select(t)
    if birincil:
        sorgu = sorgu.order_by(birincil[0].desc())      # en yeni üstte
    satirlar = db.execute(sorgu.limit(boyut).offset((sayfa - 1) * boyut)).mappings().all()

    return templates.TemplateResponse(request, 'veritabani/tablo.html', {
        'request': request, 'tablo_adi': tablo_adi,
        'sutunlar': [c.name for c in t.columns],
        'satirlar': satirlar, 'toplam': toplam,
        'sayfa': sayfa, 'sayfa_sayisi': max(1, (toplam + boyut - 1) // boyut),
    })
