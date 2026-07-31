"""Çilek Hastalık Tespit — yerel ağ web uygulaması.

Telefondan fotoğraf/video yükleyin veya IP kameradan anlık görüntü alın;
model tahmin üretsin, sonuçlar veritabanına kaydedilsin.

Çalıştırma:
    python -m app.main
    # veya
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import hashlib
import logging
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import yaml
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import config
from app.database import (Analiz, Kamera, Sera, SessionLocal, Tespit, Uretici,
                          engine, get_db, init_db)
from app.detector import detector
from app import dil, moduller, yetki

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = config.BASE_DIR
app = FastAPI(title='Çilek Hastalık Tespit')
templates = Jinja2Templates(directory=str(BASE_DIR / 'app' / 'templates'))
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'app' / 'static')), name='static')
app.mount('/media', StaticFiles(directory=str(config.STORAGE_DIR)), name='media')

@app.middleware('http')
async def dil_ara_katmani(request: Request, call_next):
    """Her istekte seçili dili bağlama koyar (şablon süzgeçleri buradan okur)."""
    dil.ayarla(dil.istekten_oku(request))
    return await call_next(request)


GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_UZANTI = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def tedavi_yukle(urun=None) -> dict:
    from app import urunler
    p = urunler.yapilandirma(urun, 'tedavi_onerileri.yaml')
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding='utf-8')) or {}


def siniflari_yukle() -> dict:
    """Sınıf listesi configs/strawberry_data.yaml'dan okunur.

    Elle etiketleme yaparken kullanılan ID'ler EĞİTİMDEKİYLE aynı olmalı;
    aksi halde düzeltilen veri modele yanlış sınıfla döner.
    """
    from app import siniflar
    return siniflar.id_haritasi()


SINIFLAR = siniflari_yukle()

TEDAVI = tedavi_yukle()

# Docker içinde MODEL_PATH konteyner yoludur (/app/models/best.pt); kullanıcı
# dosyayı kendi bilgisayarındaki proje klasörüne koyar. Karışmasın diye ayırt edilir.
DOCKER_ICINDE = Path('/.dockerenv').exists()


MODULLER = moduller.kaydet(app, engine)


@app.on_event('startup')
def baslangic():
    init_db()
    logger.info(f'Veritabanı hazır: {config.DATABASE_URL}')
    # Analiz iki yoldan yapılabilir: hiyerarşik boru hattı (organ → ROI →
    # uzman model) veya eski tek model. `detector.hazir` YALNIZCA tek modeli
    # bilir; hiyerarşiye geçtikten sonra best.pt kaldırıldığında "analiz
    # yapılamaz" diye yanıltıcı uyarı basıyordu. Karar ikisine birden bakmalı.
    from app import modeller
    hiyerarsik = modeller.hiyerarsik_hazir()
    if hiyerarsik:
        eksik = modeller.eksikler()
        logger.info('Hiyerarşik boru hattı AKTİF: '
                    + ', '.join(t['ad'] for t in modeller.durum() if t['var']))
        if eksik:
            logger.info(f'   Henüz eğitilmemiş uzman modeller: {", ".join(eksik)}')
    elif detector.hazir:
        logger.info(f'Tek model (miras) kullanılıyor: {config.MODEL_PATH}')
    else:
        logger.warning('⚠️ Hiçbir model yok — analiz yapılamaz. En az organ modeli '
                       '(models/<urun>/organ.pt) veya eski best.pt gerekir. '
                       'Kurulum: python scripts/model_kur.py --listele')


def icerik_hash(yol: Path) -> str:
    """Görüntü içeriğinin kısa hash'i — aynı dosya = aynı hash."""
    try:
        return hashlib.sha256(yol.read_bytes()).hexdigest()[:12]
    except OSError:
        return ''


def _yerel(dt: datetime) -> str:
    """UTC → yerel saat (görüntüleme için)."""
    if dt is None:
        return ''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime('%d.%m.%Y %H:%M')


templates.env.filters['yerel'] = _yerel
def _menu_moduller(grup: str):
    return [m for m in MODULLER if m.menude and m.grup == grup]


def bekleyen_sayisi() -> int:
    """İnceleme bekleyen kayıt sayısı — menüde rozet olarak gösterilir.

    Kullanıcı 'yapılacak işi' menüde görsün diye burada hesaplanır; sayfa
    başına tek küçük sorgudur.
    """
    try:
        with SessionLocal() as db:
            return (db.query(Analiz)
                    .filter(Analiz.inceleme_gerekli == True,      # noqa: E712
                            Analiz.incelendi == False).count())   # noqa: E712
    except Exception:
        return 0


# Sınıf adları ekranda seçili dile göre yazılır; veritabanı/dışa aktarım
# İngilizce adı korur (eğitimle tutarlılık için — bkz. app/dil.py)
templates.env.filters['sinif'] = dil.sinif_adi
templates.env.globals['diller'] = dil.DILLER
templates.env.globals['aktif_dil'] = dil.aktif
templates.env.globals['menu_moduller'] = _menu_moduller
templates.env.globals['bekleyen_sayisi'] = bekleyen_sayisi


def _urun_kapsami(db: Session, sera_id: Optional[int]) -> str:
    """Bu analiz hangi bitkinin model setine ait?

    Kaynak sıradüzeni: seranın ürünü → varsayılan. Bitki türünü GÖRÜNTÜDEN
    tespit etmek birincil yol değildir; kare tamamen yaprakla dolduğunda
    güvenilmez, oysa "hangi sera" bilgisi kesindir.
    """
    from app import urunler
    if sera_id:
        try:
            return urunler.seradan(db.get(Sera, sera_id))
        except Exception:
            pass
    return urunler.VARSAYILAN


def _kaydet(sonuc, db: Session, kaynak_tip: str, kaynak_ad: str,
            dosya_yolu: str, kamera_id: Optional[int] = None,
            sera_id: Optional[int] = None) -> Analiz:
    """Detector sonucunu veritabanına yazar."""
    a = Analiz(
        urun=_urun_kapsami(db, sera_id),
        kaynak_tip=kaynak_tip,
        kaynak_ad=kaynak_ad,
        kamera_id=kamera_id,
        sera_id=sera_id,
        dosya_yolu=dosya_yolu,
        dosya_hash=icerik_hash(config.STORAGE_DIR / dosya_yolu),
        # URL'de kullanildigi icin daima ileri bolu: Windows'ta '\' URL ayraci degildir
        sonuc_yolu=Path(sonuc.sonuc_yolu).relative_to(config.STORAGE_DIR).as_posix() if sonuc.sonuc_yolu else '',
        tespit_sayisi=len(sonuc.kutular),
        min_guven=sonuc.min_guven,
        ort_guven=sonuc.ort_guven,
        islenen_kare=sonuc.islenen_kare,
        sure_ms=sonuc.sure_ms,
        keskinlik=getattr(sonuc, 'keskinlik', 0.0),
        bulanik_kare=getattr(sonuc, 'bulanik_kare', 0),
        kalite_notu=getattr(sonuc, 'kalite_notu', ''),
        inceleme_gerekli=sonuc.inceleme_gerekli,
    )
    db.add(a)
    db.flush()
    for k in sonuc.kutular:
        db.add(Tespit(analiz_id=a.id, sinif_id=k.sinif_id, sinif_adi=k.sinif_adi,
                      guven=k.guven, x=k.x, y=k.y, w=k.w, h=k.h, kare=k.kare))
    db.commit()
    db.refresh(a)

    # Konum modülü etkinse konumu belirlemeye çalış (EXIF GPS / kamera konumu)
    try:
        from app.moduller.konum import konum_ata
        konum_ata(db, a, config.STORAGE_DIR / dosya_yolu,
                  db.get(Kamera, kamera_id) if kamera_id else None)
    except ImportError:
        pass                     # modül kapalı — konum atlanır
    except Exception as e:
        logger.warning(f'Konum atanamadı: {e}')

    return a


# ────────────────────────────────────────────────────────────────── sayfalar
@app.get('/', response_class=HTMLResponse)
def anasayfa(request: Request, db: Session = Depends(get_db)):
    # Veriye doğrudan değil yetki katmanı üzerinden erişilir: giriş sistemi
    # eklendiğinde izolasyon otomatik uygulanır (bkz. app/yetki.py)
    kullanici = yetki.aktif_kullanici(db)
    kameralar = yetki.gorunur_kameralar(db, kullanici)
    seralar = yetki.gorunur_seralar(db, kullanici)
    son = yetki.analiz_sorgusu(db, kullanici).order_by(Analiz.zaman.desc()).limit(6).all()
    return templates.TemplateResponse(request, 'index.html', {
        'request': request, 'kameralar': kameralar, 'seralar': seralar, 'son': son,
        'model_hazir': detector.hazir, 'model_yolu': config.MODEL_PATH,
        'docker_icinde': DOCKER_ICINDE,
    })


@app.post('/analiz/dosya')
async def analiz_dosya(request: Request, dosyalar: List[UploadFile] = File(...),
                       sera_id: Optional[int] = Form(None),
                       detayli: Optional[str] = Form(None),
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
                # Ayrıntılı mod: çok ölçekli + dilimli tarama (yavaş ama kararlı)
                sonuc = (detector.goruntu_detayli(str(hedef), str(cikti)) if detayli
                         else detector.goruntu(str(hedef), str(cikti)))
                tip = 'foto'
            else:
                logger.warning(f'Desteklenmeyen dosya atlandı: {up.filename}')
                continue
        except Exception as e:
            logger.exception('Analiz hatası')
            raise HTTPException(500, f'{up.filename}: {e}')

        kayitlar.append(_kaydet(sonuc, db, tip, up.filename,
                                hedef.relative_to(config.STORAGE_DIR).as_posix(),
                                sera_id=sera_id))

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

    a = _kaydet(sonuc, db, 'kamera', kam.tam_ad if kam else hedef_url,
                kaynak.relative_to(config.STORAGE_DIR).as_posix(),
                kamera_id=kam.id if kam else None,
                sera_id=kam.sera_id if kam else None)
    return RedirectResponse(f'/kayit/{a.id}', status_code=303)


@app.get('/kayit/{analiz_id}', response_class=HTMLResponse)
def kayit(analiz_id: int, request: Request, db: Session = Depends(get_db)):
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')
    if not yetki.erisebilir_mi(db, yetki.aktif_kullanici(db), a):
        raise HTTPException(403, 'Bu kayda erişim yetkiniz yok')

    # Sınıf bazlı özet + tedavi önerisi
    gruplar = {}
    for t in a.tespitler:
        g = gruplar.setdefault(t.sinif_adi, {'adet': 0, 'max_guven': 0.0})
        g['adet'] += 1
        g['max_guven'] = max(g['max_guven'], t.guven)
    for ad, g in gruplar.items():
        g['tedavi'] = TEDAVI.get(ad, {})

    # Aynı görüntü daha önce analiz edilmiş mi? (tekrar yükleme uyarısı)
    ayni = []
    if a.dosya_hash:
        ayni = (db.query(Analiz)
                .filter(Analiz.dosya_hash == a.dosya_hash, Analiz.id != a.id)
                .order_by(Analiz.id.desc()).all())

    return templates.TemplateResponse(request, 'kayit.html', {
        'request': request, 'a': a, 'ayni': ayni,
        'gruplar': sorted(gruplar.items(), key=lambda x: -x[1]['adet']),
    })


@app.get('/gecmis', response_class=HTMLResponse)
def gecmis(request: Request, sinif: str = '', tip: str = '', gun: int = 0,
           sera_id: int = 0, uretici_id: int = 0,
           db: Session = Depends(get_db)):
    kullanici = yetki.aktif_kullanici(db)
    q = yetki.analiz_sorgusu(db, kullanici)
    if sinif:
        q = q.join(Tespit).filter(Tespit.sinif_adi == sinif)
    if tip:
        q = q.filter(Analiz.kaynak_tip == tip)
    if gun:
        q = q.filter(Analiz.zaman >= datetime.now(timezone.utc) - timedelta(days=gun))
    if sera_id:
        q = q.filter(Analiz.sera_id == sera_id)
    if uretici_id:
        # Üreticinin tüm seralarındaki kayıtlar
        sera_idler = [s.id for s in db.query(Sera).filter(Sera.uretici_id == uretici_id)]
        q = q.filter(Analiz.sera_id.in_(sera_idler or [0]))
    kayitlar = q.order_by(Analiz.zaman.desc()).limit(200).all()

    siniflar = [r[0] for r in db.query(Tespit.sinif_adi).distinct().all()]
    return templates.TemplateResponse(request, 'gecmis.html', {
        'request': request, 'kayitlar': kayitlar, 'siniflar': sorted(siniflar),
        'seralar': yetki.gorunur_seralar(db, kullanici),
        'ureticiler': yetki.gorunur_ureticiler(db, kullanici),
        'secili': {'sinif': sinif, 'tip': tip, 'gun': gun,
                   'sera_id': sera_id, 'uretici_id': uretici_id},
    })


@app.get('/panel', response_class=HTMLResponse)
def panel(request: Request, db: Session = Depends(get_db)):
    """Genel durum.

    ÖNEMLİ İKİ KURAL:
    - Aynı görüntünün birden çok kaydı varsa yalnızca EN SON kayıt sayılır.
      Aksi halde tek bir fotoğraf, tekrar yüklendiği için istatistiği şişirir.
    - Model tespitleri ile elle düzeltilmiş etiketler AYRI raporlanır: biri
      modelin ne bulduğunu, diğeri gerçekte ne olduğunu gösterir.
    """
    kullanici = yetki.aktif_kullanici(db)
    kayitlar = yetki.analiz_sorgusu(db, kullanici).order_by(Analiz.id.asc()).all()

    # Aynı görüntünün son kaydı geçerli (etiket düzeltilmişse o sayılsın)
    benzersiz = {}
    for a in kayitlar:
        benzersiz[a.dosya_hash or f'id{a.id}'] = a
    sayilan = list(benzersiz.values())

    model_dag, elle_dag = {}, {}
    for a in sayilan:
        hedef = elle_dag if a.elle_etiketlendi else model_dag
        for t in a.tespitler:
            hedef[t.sinif_adi] = hedef.get(t.sinif_adi, 0) + 1

    bekleyen = sum(1 for a in sayilan if a.inceleme_gerekli and not a.incelendi)
    etiketli = sum(1 for a in sayilan if a.elle_etiketlendi)

    sinir = datetime.now(timezone.utc) - timedelta(days=30)
    gunluk = (yetki.analiz_sorgusu(db, kullanici)
              .filter(Analiz.zaman >= sinir)
              .with_entities(func.date(Analiz.zaman), func.count(Analiz.id))
              .group_by(func.date(Analiz.zaman))
              .order_by(func.date(Analiz.zaman)).all())

    # Sera bazlı özet — aynı tekilleştirme kuralıyla
    sera_ozet = []
    for sera in yetki.gorunur_seralar(db, kullanici):
        ait = [a for a in sayilan if a.sera_id == sera.id]
        sayac = {}
        for a in ait:
            for t in a.tespitler:
                sayac[t.sinif_adi] = sayac.get(t.sinif_adi, 0) + 1
        sera_ozet.append({
            'sera': sera, 'analiz': len(ait),
            'tespit': sum(sayac.values()),
            'bekleyen': sum(1 for a in ait if a.inceleme_gerekli and not a.incelendi),
            'en_sik': max(sayac, key=sayac.get) if sayac else '—',
        })

    return templates.TemplateResponse(request, 'panel.html', {
        'request': request,
        'toplam_analiz': len(kayitlar),
        'benzersiz_goruntu': len(sayilan),
        'tekrar': len(kayitlar) - len(sayilan),
        'bekleyen': bekleyen, 'etiketli': etiketli,
        'model_dag': sorted(model_dag.items(), key=lambda x: -x[1]),
        'elle_dag': sorted(elle_dag.items(), key=lambda x: -x[1]),
        'model_toplam': sum(model_dag.values()),
        'elle_toplam': sum(elle_dag.values()),
        'gunluk': gunluk,
        'sera_ozet': sorted(sera_ozet, key=lambda x: -x['analiz']),
        'en_yuksek': max((n for _, n in gunluk), default=1),
    })


# ─────────────────────────────────────────────────── inceleme (aktif öğrenme)
@app.get('/inceleme', response_class=HTMLResponse)
def inceleme(request: Request, db: Session = Depends(get_db)):
    kullanici = yetki.aktif_kullanici(db)
    kayitlar = (yetki.analiz_sorgusu(db, kullanici)
                .filter(Analiz.inceleme_gerekli == True, Analiz.incelendi == False)  # noqa: E712
                .order_by(Analiz.min_guven.asc(), Analiz.zaman.desc()).limit(100).all())
    etiketli = (yetki.analiz_sorgusu(db, kullanici)
                .filter(Analiz.elle_etiketlendi == True).count())  # noqa: E712
    bekleyen_aktarim = (yetki.analiz_sorgusu(db, kullanici)
                        .filter(Analiz.elle_etiketlendi == True,
                                Analiz.disa_aktarildi == False).count())  # noqa: E712
    return templates.TemplateResponse(request, 'inceleme.html', {
        'request': request, 'kayitlar': kayitlar,
        'esik': config.REVIEW_THRESHOLD,
        'etiketli': etiketli, 'bekleyen_aktarim': bekleyen_aktarim,
        'havuz_yolu': str(config.EGITIM_DIR),
        'paket_yolu': str(config.INCELEME_DIR),
        'havuz_adet': len(list((config.EGITIM_DIR / 'images').glob('*')))
                      if (config.EGITIM_DIR / 'images').exists() else 0,
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
    """Bekleyen kayıtları ön-etiketleriyle DIŞ ARAÇ paketine yazar.

    Tek klasör kullanılır ve her aktarımda TEMİZLENİP yeniden yazılır. Tarihli
    anlık görüntü biriktirilseydi, kayıt sonradan etiketlendiğinde klasördeki
    etiket dosyası veritabanıyla çelişirdi (eski tahmin vs düzeltilmiş etiket).
    """
    kayitlar = (yetki.analiz_sorgusu(db, yetki.aktif_kullanici(db))
                .filter(Analiz.inceleme_gerekli == True, Analiz.incelendi == False)  # noqa: E712
                .all())
    if not kayitlar:
        raise HTTPException(400, 'Dışa aktarılacak bekleyen kayıt yok.')

    hedef = config.INCELEME_DIR
    if hedef.exists():
        shutil.rmtree(hedef)            # eskimiş içerik kalmasın
    (hedef / 'images').mkdir(parents=True, exist_ok=True)
    (hedef / 'labels').mkdir(parents=True, exist_ok=True)

    n = 0
    for a in kayitlar:
        kaynak = config.STORAGE_DIR / a.dosya_yolu
        if not kaynak.exists():
            continue
        # Adlandırma eğitim havuzuyla aynı: <sera>_<icerik-hash>
        sera = (a.sera.ad if a.sera else 'atanmamis').replace(' ', '_')
        damga = a.dosya_hash or f'id{a.id}'
        ad = f'{sera}_{damga}{kaynak.suffix.lower()}'
        shutil.copy2(kaynak, hedef / 'images' / ad)
        satirlar = [f'{t.sinif_id} {t.x:.6f} {t.y:.6f} {t.w:.6f} {t.h:.6f}'
                    for t in a.tespitler]
        (hedef / 'labels' / f'{Path(ad).stem}.txt').write_text(
            chr(10).join(satirlar) + (chr(10) if satirlar else ''), encoding='utf-8')
        n += 1

    with open(hedef / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump({'train': 'images', 'val': 'images',
                   'nc': len(SINIFLAR),
                   'names': {int(k): v for k, v in SINIFLAR.items()}},
                  f, allow_unicode=True, sort_keys=False)

    logger.info(f'{n} kayıt inceleme paketine yazıldı (yeniden üretildi): {hedef}')
    return RedirectResponse(f'/inceleme?aktarildi={n}', status_code=303)


# ──────────────────────────────────────────────── elle etiketleme (düzeltme)
@app.get('/kayit/{analiz_id}/etiketle', response_class=HTMLResponse)
def etiketle(analiz_id: int, request: Request, db: Session = Depends(get_db)):
    """Tahmin kutularının tarayıcı üzerinde düzeltildiği sayfa.

    Model tahminleri ÖN-ETİKET olarak gelir; uzman düzeltir. Düzeltilen kayıt
    doğrudan eğitim verisi olur (bkz. /inceleme → eğitime hazırla).
    """
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')
    if not yetki.erisebilir_mi(db, yetki.aktif_kullanici(db), a):
        raise HTTPException(403, 'Bu kayda erişim yetkiniz yok')

    kutular = [{'sinif_id': t.sinif_id, 'sinif_adi': dil.sinif_adi(t.sinif_adi),
                'guven': t.guven,
                'x': t.x, 'y': t.y, 'w': t.w, 'h': t.h} for t in a.tespitler]
    return templates.TemplateResponse(request, 'etiketle.html', {
        'request': request, 'a': a, 'kutular': kutular, 'siniflar': SINIFLAR,
    })


@app.post('/api/kayit/{analiz_id}/etiketler')
def etiketleri_kaydet(analiz_id: int, veri: dict = Body(...),
                      db: Session = Depends(get_db)):
    """Düzeltilmiş kutuları kaydeder (mevcut tespitlerin yerine geçer)."""
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')
    if not yetki.erisebilir_mi(db, yetki.aktif_kullanici(db), a):
        raise HTTPException(403, 'Bu kayda erişim yetkiniz yok')

    kutular = veri.get('kutular', [])
    for k in kutular:
        if int(k['sinif_id']) not in SINIFLAR:
            raise HTTPException(400, f"Geçersiz sınıf: {k['sinif_id']}")
        for alan in ('x', 'y', 'w', 'h'):
            if not (0.0 <= float(k[alan]) <= 1.0):
                raise HTTPException(400, f'Koordinat aralık dışı: {alan}={k[alan]}')

    # Eski tespitler silinir; düzeltilmiş küme yazılır
    db.query(Tespit).filter(Tespit.analiz_id == a.id).delete()
    for k in kutular:
        cid = int(k['sinif_id'])
        db.add(Tespit(analiz_id=a.id, sinif_id=cid, sinif_adi=SINIFLAR[cid],
                      guven=1.0,          # elle çizilen kutu kesin kabul edilir
                      x=float(k['x']), y=float(k['y']),
                      w=float(k['w']), h=float(k['h'])))

    a.tespit_sayisi = len(kutular)
    a.min_guven = 1.0 if kutular else 0.0
    a.ort_guven = 1.0 if kutular else 0.0
    a.elle_etiketlendi = True
    a.incelendi = True
    a.disa_aktarildi = False        # düzeltildi → yeniden dışa aktarılmalı
    db.commit()
    return {'durum': 'ok', 'kutu': len(kutular)}


@app.post('/kayit/{analiz_id}/yanlis-tespit')
def yanlis_tespit(analiz_id: int, db: Session = Depends(get_db)):
    """Kaydı tek tıkla NEGATİF ÖRNEK yapar (tüm kutular silinir).

    NEDEN GEREKLİ?
        Modelin en can sıkıcı hatası yanlış pozitiftir: sarı yapışkan tuzağı
        olgunlaşmamış çilek, saksı kenarını lezyon sanmak gibi. Bu hatayı eşik
        yükselterek "gizlemek" gerçek tespitleri de kaybettirir.

        Kalıcı çözüm, o görüntüyü BOŞ ETİKETLE eğitime katmaktır. YOLO'da
        etiketi boş olan görüntü "background" örneğidir: model o görünümde
        hiçbir sınıfın olmadığını öğrenir. Yanlış pozitifleri azaltmanın
        standart ve en etkili yolu budur.

        Etiketleme ekranından kutuları tek tek silmek de aynı sonucu verir;
        bu düğme sık yapılan bu işi tek adıma indirir.
    """
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')
    if not yetki.erisebilir_mi(db, yetki.aktif_kullanici(db), a):
        raise HTTPException(403, 'Bu kayda erişim yetkiniz yok')

    db.query(Tespit).filter(Tespit.analiz_id == a.id).delete()
    a.tespit_sayisi = 0
    a.min_guven = a.ort_guven = 0.0
    a.elle_etiketlendi = True       # elle onaylanmış negatif → eğitime gider
    a.incelendi = True
    a.disa_aktarildi = False        # havuza yeniden yazılmalı
    db.commit()
    logger.info(f'Kayıt #{a.id} negatif örnek olarak işaretlendi')
    return RedirectResponse(f'/kayit/{a.id}', status_code=303)


@app.post('/inceleme/egitime-hazirla')
def egitime_hazirla(yeniden: int = Form(0), db: Session = Depends(get_db)):
    """Elle etiketlenmiş kayıtları TEK birikimli eğitim klasörüne yazar.

    NEDEN TEK KLASÖR: Her aktarımda tarihli yeni klasör açmak, eğitim öncesinde
    onlarca klasörü elle toplamayı gerektirirdi. Kayıtlar storage/egitim_verisi/
    altında birikir; merge_datasets.py'ye her zaman aynı tek yol verilir.

    KOPYA ÖNLEME: Dosya adı görüntünün İÇERİK HASH'ine göre verilir. Aynı
    fotoğraf iki kez yüklenip iki kez etiketlenmişse havuzda tek dosya olur ve
    EN SON etiketlenen sürüm geçerli sayılır — çelişen etiketlerle eğitim yapılmaz.
    """
    q = db.query(Analiz).filter(Analiz.elle_etiketlendi == True)  # noqa: E712
    kayitlar = q.order_by(Analiz.id.asc()).all()
    if not kayitlar:
        raise HTTPException(400, 'Aktarılacak etiketli kayıt yok. Önce inceleme '
                                 'kuyruğundaki kayıtları etiketleyin.')

    # Aynı görüntünün birden çok kaydı varsa en son etiketleneni kalsın
    benzersiz = {}
    for a in kayitlar:
        anahtar = a.dosya_hash or f'id{a.id}'
        benzersiz[anahtar] = a          # sıralı gidildiği için en büyük id kalır
    secilenler = list(benzersiz.values())

    if not yeniden:
        yazilacak = [a for a in secilenler if not a.disa_aktarildi]
        if not yazilacak:
            raise HTTPException(400, 'Aktarılacak yeni etiketli kayıt yok.')
    else:
        yazilacak = secilenler

    hedef = config.EGITIM_DIR
    (hedef / 'images').mkdir(parents=True, exist_ok=True)
    (hedef / 'labels').mkdir(parents=True, exist_ok=True)

    n = 0
    for a in yazilacak:
        kaynak = config.STORAGE_DIR / a.dosya_yolu
        if not kaynak.exists():
            continue
        # Ad: <sera>_<icerik-hash> — sera bilgisi split_dataset.py'nin grup ayrımı
        # için, hash ise aynı görüntünün tek dosyaya yazılması için
        sera = (a.sera.ad if a.sera else 'atanmamis').replace(' ', '_')
        damga = a.dosya_hash or f'id{a.id}'
        ad = f'{sera}_{damga}{kaynak.suffix.lower()}'
        shutil.copy2(kaynak, hedef / 'images' / ad)
        satirlar = [f'{t.sinif_id} {t.x:.6f} {t.y:.6f} {t.w:.6f} {t.h:.6f}'
                    for t in a.tespitler]
        (hedef / 'labels' / f'{Path(ad).stem}.txt').write_text(
            chr(10).join(satirlar) + (chr(10) if satirlar else ''), encoding='utf-8')
        a.disa_aktarildi = True
        n += 1

    # Aynı görüntünün eski (id tabanlı) kopyaları varsa temizle
    gecerli = {f'{(a.sera.ad if a.sera else "atanmamis").replace(" ", "_")}_'
               f'{a.dosya_hash or f"id{a.id}"}' for a in secilenler}
    for f in (hedef / 'images').glob('*'):
        if f.stem not in gecerli:
            f.unlink(missing_ok=True)
            (hedef / 'labels' / f'{f.stem}.txt').unlink(missing_ok=True)
            logger.info(f'Havuzdan kaldırıldı (kopya/eskimiş): {f.name}')

    # merge_datasets.py her kaynakta data.yaml arar
    with open(hedef / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump({'train': 'images', 'val': 'images',
                   'nc': len(SINIFLAR),
                   'names': {int(k): v for k, v in SINIFLAR.items()}},
                  f, allow_unicode=True, sort_keys=False)

    db.commit()
    logger.info(f'{n} etiketli kayıt eğitim havuzuna yazıldı: {hedef}')
    return RedirectResponse(f'/inceleme?eklendi={n}', status_code=303)


# ─────────────────────────────────── etiketlenmiş kayıtlar (görüntüleme)
# Etiketleme arayüzüyle aynı renk düzeni (BGR — OpenCV)
ETIKET_RENKLERI = [(53, 57, 229), (162, 36, 142), (75, 73, 57), (139, 136, 0),
                   (30, 81, 244), (65, 76, 109), (51, 202, 192), (193, 172, 0),
                   (0, 140, 251), (177, 53, 94)]


@app.get('/kayit/{analiz_id}/etiket-onizleme.jpg')
def etiket_onizleme(analiz_id: int, db: Session = Depends(get_db)):
    """Kutuları görüntü üzerine ANLIK çizip döner.

    Önizleme dosya olarak saklanmaz: etiket düzeltilince eskimiş görsel
    kalmasın diye her istekte veritabanının o anki hâlinden üretilir.
    """
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')
    if not yetki.erisebilir_mi(db, yetki.aktif_kullanici(db), a):
        raise HTTPException(403, 'Bu kayda erişim yetkiniz yok')

    kaynak = config.STORAGE_DIR / a.dosya_yolu
    if not kaynak.exists():
        raise HTTPException(404, 'Görüntü dosyası yok')

    img = cv2.imread(str(kaynak))
    if img is None:
        raise HTTPException(500, 'Görüntü okunamadı')
    h, w = img.shape[:2]
    kalinlik = max(2, int(min(h, w) / 350))

    for t in a.tespitler:
        x1 = int((t.x - t.w / 2) * w); y1 = int((t.y - t.h / 2) * h)
        x2 = int((t.x + t.w / 2) * w); y2 = int((t.y + t.h / 2) * h)
        renk = ETIKET_RENKLERI[t.sinif_id % len(ETIKET_RENKLERI)]
        cv2.rectangle(img, (x1, y1), (x2, y2), renk, kalinlik)
        cv2.putText(img, f'{t.sinif_id} {t.sinif_adi}', (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, min(h, w) / 1600),
                    renk, kalinlik)

    ok, tampon = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, 'Önizleme üretilemedi')
    return Response(content=tampon.tobytes(), media_type='image/jpeg',
                    headers={'Cache-Control': 'no-store'})   # eskimiş görsel gösterme


@app.get('/etiketlenenler', response_class=HTMLResponse)
def etiketlenenler(request: Request, db: Session = Depends(get_db)):
    """Elle etiketlenmiş kayıtlar — ne etiketlediğinizi görün ve düzeltin."""
    kullanici = yetki.aktif_kullanici(db)
    kayitlar = (yetki.analiz_sorgusu(db, kullanici)
                .filter(Analiz.elle_etiketlendi == True)  # noqa: E712
                .order_by(Analiz.id.desc()).all())

    # Sınıf dağılımı: etiketlediğiniz kutular hangi sınıflarda?
    dagilim = {}
    for a in kayitlar:
        for t in a.tespitler:
            dagilim[t.sinif_adi] = dagilim.get(t.sinif_adi, 0) + 1

    return templates.TemplateResponse(request, 'etiketlenenler.html', {
        'request': request, 'kayitlar': kayitlar,
        'dagilim': sorted(dagilim.items(), key=lambda x: -x[1]),
        'toplam_kutu': sum(dagilim.values()),
        'bekleyen_aktarim': sum(1 for a in kayitlar if not a.disa_aktarildi),
        'havuz_yolu': str(config.EGITIM_DIR),
    })



# ───────────────────────────────────────────────────── kalıcı silme
def _havuz_anahtari(a: Analiz) -> str:
    """Eğitim havuzu / paket dosyalarının ad kökü: <sera>_<icerik-hash>."""
    sera = (a.sera.ad if a.sera else 'atanmamis').replace(' ', '_')
    return f'{sera}_{a.dosya_hash or f"id{a.id}"}'


@app.post('/kayit/{analiz_id}/sil')
def kayit_sil(analiz_id: int, db: Session = Depends(get_db)):
    """Kaydı ve ona ait TÜM dosyaları kalıcı olarak siler.

    Silinenler: veritabanı satırı ve tespitleri, yüklenen orijinal görüntü,
    kutulanmış sonuç görseli, eğitim havuzundaki ve dış araç paketindeki
    kopyaları. Geri alınamaz.

    DİKKAT: Aynı görüntünün etiketlenmiş başka bir kaydı varsa havuz dosyası
    SİLİNMEZ — o kayda ait olduğu için kalması gerekir.
    """
    a = db.get(Analiz, analiz_id)
    if not a:
        raise HTTPException(404, 'Kayıt bulunamadı')
    if not yetki.erisebilir_mi(db, yetki.aktif_kullanici(db), a):
        raise HTTPException(403, 'Bu kayda erişim yetkiniz yok')

    anahtar = _havuz_anahtari(a)
    silinecek = []

    # Yüklenen orijinal ve kutulanmış sonuç
    for goreli in (a.dosya_yolu, a.sonuc_yolu):
        if goreli:
            silinecek.append(config.STORAGE_DIR / goreli)

    # Havuz/paket kopyaları — aynı anahtarı kullanan başka kayıt kalmıyorsa
    baska = [b for b in db.query(Analiz).filter(Analiz.id != a.id).all()
             if b.elle_etiketlendi and _havuz_anahtari(b) == anahtar]
    if not baska:
        for kok in (config.EGITIM_DIR, config.INCELEME_DIR):
            for f in (kok / 'images').glob(f'{anahtar}.*'):
                silinecek.append(f)
            silinecek.append(kok / 'labels' / f'{anahtar}.txt')

    silinen = 0
    for yol in silinecek:
        try:
            if yol.exists():
                yol.unlink()
                silinen += 1
        except OSError as e:
            logger.warning(f'Dosya silinemedi: {yol} ({e})')

    db.delete(a)          # tespitler cascade ile gider
    db.commit()
    logger.info(f'Kayıt #{analiz_id} kalıcı silindi ({silinen} dosya).')
    return RedirectResponse('/etiketlenenler?silindi=1', status_code=303)



# ──────────────────────────────────────────────────────────────────── kameralar
@app.get('/kameralar', response_class=HTMLResponse)
def kameralar(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(request, 'kameralar.html', {
        'request': request,
        'kameralar': yetki.gorunur_kameralar(db, yetki.aktif_kullanici(db), yalniz_aktif=False),
        'seralar': yetki.gorunur_seralar(db, yetki.aktif_kullanici(db)),
    })


@app.post('/kameralar/ekle')
def kamera_ekle(ad: str = Form(...), url: str = Form(...), konum: str = Form(''),
                sera_id: Optional[int] = Form(None),
                blok: str = Form(''), sira: str = Form(''),
                enlem: str = Form(''), boylam: str = Form(''),
                db: Session = Depends(get_db)):
    def sayi(x):
        try:
            return float(x) if x else None
        except ValueError:
            return None
    db.add(Kamera(ad=ad.strip(), url=url.strip(), konum=konum.strip(), sera_id=sera_id,
                  blok=blok.strip(), sira=sira.strip(),
                  enlem=sayi(enlem), boylam=sayi(boylam)))
    db.commit()
    return RedirectResponse('/kameralar', status_code=303)


@app.post('/kameralar/{kamera_id}/sil')
def kamera_sil(kamera_id: int, db: Session = Depends(get_db)):
    kam = db.get(Kamera, kamera_id)
    if kam:
        kam.aktif = False        # kayıtlar bozulmasın diye pasife alınır
        db.commit()
    return RedirectResponse('/kameralar', status_code=303)


# ──────────────────────────────────────────────── üretici ve sera yönetimi
@app.get('/isletmeler', response_class=HTMLResponse)
def isletmeler(request: Request, db: Session = Depends(get_db)):
    """Üretici → Sera → Kamera hiyerarşisini tek sayfada yönetir."""
    return templates.TemplateResponse(request, 'isletmeler.html', {
        'request': request,
        'ureticiler': yetki.gorunur_ureticiler(db, yetki.aktif_kullanici(db)),
    })


@app.post('/ureticiler/ekle')
def uretici_ekle(ad: str = Form(...), telefon: str = Form(''), eposta: str = Form(''),
                 notlar: str = Form(''), db: Session = Depends(get_db)):
    db.add(Uretici(ad=ad.strip(), telefon=telefon.strip(),
                   eposta=eposta.strip(), notlar=notlar.strip()))
    db.commit()
    return RedirectResponse('/isletmeler', status_code=303)


@app.post('/ureticiler/{uretici_id}/sil')
def uretici_sil(uretici_id: int, db: Session = Depends(get_db)):
    u = db.get(Uretici, uretici_id)
    if u:
        u.aktif = False          # geçmiş kayıtlar sahipsiz kalmasın diye pasife alınır
        db.commit()
    return RedirectResponse('/isletmeler', status_code=303)


@app.post('/seralar/ekle')
def sera_ekle(uretici_id: int = Form(...), ad: str = Form(...), konum: str = Form(''),
              urun: str = Form('Çilek'), db: Session = Depends(get_db)):
    db.add(Sera(uretici_id=uretici_id, ad=ad.strip(),
                konum=konum.strip(), urun=urun.strip() or 'Çilek'))
    db.commit()
    return RedirectResponse('/isletmeler', status_code=303)


@app.post('/seralar/{sera_id}/sil')
def sera_sil(sera_id: int, db: Session = Depends(get_db)):
    s = db.get(Sera, sera_id)
    if s:
        s.aktif = False
        db.commit()
    return RedirectResponse('/isletmeler', status_code=303)


@app.get('/dil/{kod}')
def dil_sec(kod: str, request: Request):
    """Arayüz dilini değiştirir ve gelinen sayfaya döner."""
    kod = kod if kod in dil.DILLER else dil.VARSAYILAN
    geri = request.headers.get('referer') or '/'
    yanit = RedirectResponse(geri, status_code=303)
    # 1 yıl: kullanıcı her açılışta tekrar seçmesin
    yanit.set_cookie(dil.CEREZ, kod, max_age=31536000, samesite='lax')
    return yanit


if __name__ == '__main__':
    import threading

    import uvicorn

    # HTTP her zaman açık kalır (:8000). Sertifika varsa AYRICA https açılır
    # (:8443).
    #
    # NEDEN İKİSİ BİRDEN: canlı kamera (getUserMedia) yalnızca güvenli bağlamda
    # çalışır, yani https şart. Ama sunucuyu sadece https yapmak eski
    # http://...:8000 adresini kırar ve "proje çalışmıyor" gibi görünür.
    # İki dinleyici aynı uygulama nesnesini paylaşır — tek süreç, tek
    # veritabanı bağlantısı, kilitlenme riski yok.
    def _sunucu(**kw):
        # Server.run() kendi olay döngüsünü kurar; ana thread dışında uvicorn
        # sinyal işleyicilerini kendiliğinden atlar.
        uvicorn.Server(uvicorn.Config(app, log_level='info', **kw)).run()

    guvenli = bool(config.SSL_CERT and config.SSL_KEY)

    print(f'\n🍓 Arayüz: http://localhost:{config.PORT}')
    if guvenli:
        print(f'🔒 Güvenli (canlı kamera için): https://localhost:{config.HTTPS_PORT}')
        print(f'📱 Telefondan: https://<bilgisayarınızın-IP-adresi>:{config.HTTPS_PORT}')
    else:
        print(f'📱 Telefondan: http://<bilgisayarınızın-IP-adresi>:{config.PORT}')
        print('   ⚠️ Canlı kamera için sertifika gerekir: python scripts/https_sertifika.py')
    print()

    if guvenli:
        threading.Thread(target=_sunucu, daemon=True, kwargs={
            'host': config.HOST, 'port': config.HTTPS_PORT,
            'ssl_certfile': config.SSL_CERT, 'ssl_keyfile': config.SSL_KEY,
        }).start()

    _sunucu(host=config.HOST, port=config.PORT)
