"""Çilek Hastalık Tespit — yerel ağ web uygulaması.

Telefondan fotoğraf/video yükleyin veya IP kameradan anlık görüntü alın;
model tahmin üretsin, sonuçlar veritabanına kaydedilsin.

Çalıştırma:
    python -m app.main
    # veya
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import config
from app.database import Analiz, Kamera, SessionLocal, Tespit, get_db, init_db
from app.detector import detector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = config.BASE_DIR
app = FastAPI(title='Çilek Hastalık Tespit')
templates = Jinja2Templates(directory=str(BASE_DIR / 'app' / 'templates'))
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'app' / 'static')), name='static')
app.mount('/media', StaticFiles(directory=str(config.STORAGE_DIR)), name='media')

GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_UZANTI = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def tedavi_yukle() -> dict:
    p = BASE_DIR / 'configs' / 'tedavi_onerileri.yaml'
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding='utf-8')) or {}


TEDAVI = tedavi_yukle()


@app.on_event('startup')
def baslangic():
    init_db()
    logger.info(f'Veritabanı hazır: {config.DATABASE_URL}')
    if detector.hazir:
        logger.info(f'Model bulundu: {config.MODEL_PATH}')
    else:
        logger.warning(f'⚠️ Model YOK: {config.MODEL_PATH} — analiz yapılamaz. '
                       'Eğitilmiş best.pt dosyasını models/ klasörüne koyun.')


def _yerel(dt: datetime) -> str:
    """UTC → yerel saat (görüntüleme için)."""
    if dt is None:
        return ''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime('%d.%m.%Y %H:%M')


templates.env.filters['yerel'] = _yerel


def _kaydet(sonuc, db: Session, kaynak_tip: str, kaynak_ad: str,
            dosya_yolu: str, kamera_id: Optional[int] = None) -> Analiz:
    """Detector sonucunu veritabanına yazar."""
    a = Analiz(
        kaynak_tip=kaynak_tip,
        kaynak_ad=kaynak_ad,
        kamera_id=kamera_id,
        dosya_yolu=dosya_yolu,
        sonuc_yolu=str(Path(sonuc.sonuc_yolu).relative_to(config.STORAGE_DIR)) if sonuc.sonuc_yolu else '',
        tespit_sayisi=len(sonuc.kutular),
        min_guven=sonuc.min_guven,
        ort_guven=sonuc.ort_guven,
        islenen_kare=sonuc.islenen_kare,
        sure_ms=sonuc.sure_ms,
        inceleme_gerekli=sonuc.inceleme_gerekli,
    )
    db.add(a)
    db.flush()
    for k in sonuc.kutular:
        db.add(Tespit(analiz_id=a.id, sinif_id=k.sinif_id, sinif_adi=k.sinif_adi,
                      guven=k.guven, x=k.x, y=k.y, w=k.w, h=k.h, kare=k.kare))
    db.commit()
    db.refresh(a)
    return a


# ────────────────────────────────────────────────────────────────── sayfalar
@app.get('/', response_class=HTMLResponse)
def anasayfa(request: Request, db: Session = Depends(get_db)):
    kameralar = db.query(Kamera).filter(Kamera.aktif == True).all()  # noqa: E712
    son = db.query(Analiz).order_by(Analiz.zaman.desc()).limit(6).all()
    return templates.TemplateResponse(request, 'index.html', {
        'request': request, 'kameralar': kameralar, 'son': son,
        'model_hazir': detector.hazir, 'model_yolu': config.MODEL_PATH,
    })


@app.post('/analiz/dosya')
async def analiz_dosya(request: Request, dosyalar: List[UploadFile] = File(...),
                       db: Session = Depends(get_db)):
    """Telefondan/bilgisayardan yüklenen fotoğraf ve videoları işler."""
    if not detector.hazir:
        raise HTTPException(400, f'Model bulunamadı: {config.MODEL_PATH}')

    kayitlar = []
    for up in dosyalar:
        if not up.filename:
            continue
        uzanti = Path(up.filename).suffix.lower()
        tekil = f'{uuid.uuid4().hex[:12]}{uzanti}'
        hedef = config.UPLOAD_DIR / tekil
        with open(hedef, 'wb') as f:
            shutil.copyfileobj(up.file, f)

        cikti = config.RESULT_DIR / f'{Path(tekil).stem}.jpg'
        try:
            if uzanti in VIDEO_UZANTI:
                sonuc = detector.video(str(hedef), str(cikti))
                tip = 'video'
            elif uzanti in GORUNTU_UZANTI:
                sonuc = detector.goruntu(str(hedef), str(cikti))
                tip = 'foto'
            else:
                logger.warning(f'Desteklenmeyen dosya atlandı: {up.filename}')
                continue
        except Exception as e:
            logger.exception('Analiz hatası')
            raise HTTPException(500, f'{up.filename}: {e}')

        kayitlar.append(_kaydet(sonuc, db, tip, up.filename,
                                str(hedef.relative_to(config.STORAGE_DIR))))

    if not kayitlar:
        raise HTTPException(400, 'İşlenebilir dosya bulunamadı.')
    if len(kayitlar) == 1:
        return RedirectResponse(f'/kayit/{kayitlar[0].id}', status_code=303)
    return RedirectResponse('/gecmis', status_code=303)


@app.post('/analiz/kamera')
def analiz_kamera(kamera_id: Optional[int] = Form(None), url: str = Form(''),
                  db: Session = Depends(get_db)):
    """IP kameradan anlık görüntü alıp analiz eder."""
    if not detector.hazir:
        raise HTTPException(400, f'Model bulunamadı: {config.MODEL_PATH}')

    kam = db.get(Kamera, kamera_id) if kamera_id else None
    hedef_url = (kam.url if kam else url).strip()
    if not hedef_url:
        raise HTTPException(400, 'Kamera seçin veya URL girin.')

    tekil = uuid.uuid4().hex[:12]
    kaynak = config.UPLOAD_DIR / f'{tekil}.jpg'
    cikti = config.RESULT_DIR / f'{tekil}_sonuc.jpg'
    try:
        sonuc = detector.kamera(hedef_url, str(cikti), kaynak_kaydet=str(kaynak))
    except Exception as e:
        raise HTTPException(502, str(e))

    a = _kaydet(sonuc, db, 'kamera', kam.ad if kam else hedef_url,
                str(kaynak.relative_to(config.STORAGE_DIR)),
                kamera_id=kam.id if kam else None)
    return RedirectResponse(f'/kayit/{a.id}', status_code=303)


@app.get('/kayit/{analiz_id}', response_class=HTMLResponse)
def kayit(analiz_id: int, request: Request, db: Session = Depends(get_db)):
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')

    # Sınıf bazlı özet + tedavi önerisi
    gruplar = {}
    for t in a.tespitler:
        g = gruplar.setdefault(t.sinif_adi, {'adet': 0, 'max_guven': 0.0})
        g['adet'] += 1
        g['max_guven'] = max(g['max_guven'], t.guven)
    for ad, g in gruplar.items():
        g['tedavi'] = TEDAVI.get(ad, {})

    return templates.TemplateResponse(request, 'kayit.html', {
        'request': request, 'a': a,
        'gruplar': sorted(gruplar.items(), key=lambda x: -x[1]['adet']),
    })


@app.get('/gecmis', response_class=HTMLResponse)
def gecmis(request: Request, sinif: str = '', tip: str = '', gun: int = 0,
           db: Session = Depends(get_db)):
    q = db.query(Analiz)
    if sinif:
        q = q.join(Tespit).filter(Tespit.sinif_adi == sinif)
    if tip:
        q = q.filter(Analiz.kaynak_tip == tip)
    if gun:
        q = q.filter(Analiz.zaman >= datetime.now(timezone.utc) - timedelta(days=gun))
    kayitlar = q.order_by(Analiz.zaman.desc()).limit(200).all()

    siniflar = [r[0] for r in db.query(Tespit.sinif_adi).distinct().all()]
    return templates.TemplateResponse(request, 'gecmis.html', {
        'request': request, 'kayitlar': kayitlar, 'siniflar': sorted(siniflar),
        'secili': {'sinif': sinif, 'tip': tip, 'gun': gun},
    })


@app.get('/panel', response_class=HTMLResponse)
def panel(request: Request, db: Session = Depends(get_db)):
    toplam = db.query(func.count(Analiz.id)).scalar() or 0
    bekleyen = db.query(func.count(Analiz.id)).filter(
        Analiz.inceleme_gerekli == True, Analiz.incelendi == False).scalar() or 0  # noqa: E712

    sinif_dagilim = (db.query(Tespit.sinif_adi, func.count(Tespit.id))
                     .group_by(Tespit.sinif_adi)
                     .order_by(func.count(Tespit.id).desc()).all())

    sinir = datetime.now(timezone.utc) - timedelta(days=30)
    gunluk = (db.query(func.date(Analiz.zaman), func.count(Analiz.id))
              .filter(Analiz.zaman >= sinir)
              .group_by(func.date(Analiz.zaman))
              .order_by(func.date(Analiz.zaman)).all())

    return templates.TemplateResponse(request, 'panel.html', {
        'request': request, 'toplam': toplam, 'bekleyen': bekleyen,
        'sinif_dagilim': sinif_dagilim, 'gunluk': gunluk,
        'en_yuksek': max((s for _, s in gunluk), default=1),
    })


# ─────────────────────────────────────────────────── inceleme (aktif öğrenme)
@app.get('/inceleme', response_class=HTMLResponse)
def inceleme(request: Request, db: Session = Depends(get_db)):
    kayitlar = (db.query(Analiz)
                .filter(Analiz.inceleme_gerekli == True, Analiz.incelendi == False)  # noqa: E712
                .order_by(Analiz.min_guven.asc(), Analiz.zaman.desc()).limit(100).all())
    return templates.TemplateResponse(request, 'inceleme.html', {
        'request': request, 'kayitlar': kayitlar,
        'esik': config.REVIEW_THRESHOLD,
    })


@app.post('/inceleme/{analiz_id}/tamam')
def inceleme_tamam(analiz_id: int, db: Session = Depends(get_db)):
    a = db.get(Analiz, analiz_id)
    if a:
        a.incelendi = True
        db.commit()
    return RedirectResponse('/inceleme', status_code=303)


@app.post('/inceleme/disa-aktar')
def inceleme_disa_aktar(db: Session = Depends(get_db)):
    """Bekleyen kayıtları ön-etiketleriyle dışa aktarır.

    Çıktı doğrudan Roboflow'a yüklenebilir: uzman sıfırdan çizmez, modelin
    tahminlerini düzeltir. Düzeltilmiş veri merge_datasets.py ile ana
    dataset'e katılır (bkz. README — Sürekli İyileştirme).
    """
    kayitlar = (db.query(Analiz)
                .filter(Analiz.inceleme_gerekli == True, Analiz.incelendi == False)  # noqa: E712
                .all())
    if not kayitlar:
        raise HTTPException(400, 'Dışa aktarılacak kayıt yok.')

    damga = datetime.now().strftime('%Y%m%d_%H%M')
    hedef = config.EXPORT_DIR / f'inceleme_{damga}'
    (hedef / 'images').mkdir(parents=True, exist_ok=True)
    (hedef / 'labels').mkdir(parents=True, exist_ok=True)

    n = 0
    for a in kayitlar:
        kaynak = config.STORAGE_DIR / a.dosya_yolu
        if not kaynak.exists():
            continue
        ad = f'{a.id}_{kaynak.name}'
        shutil.copy2(kaynak, hedef / 'images' / ad)
        with open(hedef / 'labels' / f'{Path(ad).stem}.txt', 'w', encoding='utf-8') as f:
            for t in a.tespitler:
                f.write(f'{t.sinif_id} {t.x:.6f} {t.y:.6f} {t.w:.6f} {t.h:.6f}\n')
        n += 1

    logger.info(f'{n} kayıt dışa aktarıldı: {hedef}')
    return RedirectResponse(f'/inceleme?aktarildi={n}', status_code=303)


# ──────────────────────────────────────────────────────────────────── kameralar
@app.get('/kameralar', response_class=HTMLResponse)
def kameralar(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(request, 'kameralar.html', {
        'request': request, 'kameralar': db.query(Kamera).all(),
    })


@app.post('/kameralar/ekle')
def kamera_ekle(ad: str = Form(...), url: str = Form(...), konum: str = Form(''),
                db: Session = Depends(get_db)):
    db.add(Kamera(ad=ad.strip(), url=url.strip(), konum=konum.strip()))
    db.commit()
    return RedirectResponse('/kameralar', status_code=303)


@app.post('/kameralar/{kamera_id}/sil')
def kamera_sil(kamera_id: int, db: Session = Depends(get_db)):
    kam = db.get(Kamera, kamera_id)
    if kam:
        kam.aktif = False        # kayıtlar bozulmasın diye pasife alınır
        db.commit()
    return RedirectResponse('/kameralar', status_code=303)


if __name__ == '__main__':
    import uvicorn
    print(f'\n🍓 Arayüz: http://localhost:{config.PORT}')
    print(f'📱 Telefondan: http://<bilgisayarınızın-IP-adresi>:{config.PORT}\n')
    uvicorn.run('app.main:app', host=config.HOST, port=config.PORT, reload=False)
