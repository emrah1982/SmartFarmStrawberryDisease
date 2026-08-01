"""Böcek teşhis modülünün sayfaları.

SEKME YAPISI
    /bocek         → Teşhis (fotoğraf yükle, sonuç gör)
    /bocek/gecmis  → Geçmiş (kayıtlar, isabet özeti)

    Sekmeler AYRI ROTALARDIR, tek sayfada JS ile gizlenen paneller değil.
    Telefonda tek sayfaya her şeyi yığmak yükleme süresini uzatır; ayrıca
    geçmiş sayfası büyüdükçe teşhis akışını yavaşlatırdı. Üstteki sekme
    çubuğu ikisini görsel olarak birleştirir.
"""

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app import config
from app.database import get_db
from app.moduller.bocek import servis
from app.moduller.bocek.modeller import DOGRULAMA, BocekKaydi, isabet

router = APIRouter(prefix='/bocek', tags=['bocek'])

templates = Jinja2Templates(directory=[
    str(Path(__file__).parent / 'templates'),
    str(config.BASE_DIR / 'app' / 'templates'),
])

KAYIT_DIZINI = 'bocek'


def _ortak_ayarlar():
    from app.moduller import sablon_ayarla
    sablon_ayarla(templates)


def _baglam(request: Request, sekme: str, **ek):
    from app import urunler
    urun = urunler.VARSAYILAN
    return {
        'request': request,
        'sekme': sekme,
        'hazir': servis.hazir(urun),
        'turler': servis.taniyabildikleri(urun),
        **ek,
    }


# ─────────────────────────────────────────────────────────── teşhis sekmesi
@router.get('', response_class=HTMLResponse)
def sayfa(request: Request):
    _ortak_ayarlar()
    return templates.TemplateResponse(request, 'bocek.html',
                                      _baglam(request, 'teshis'))


@router.post('/tani', response_class=HTMLResponse)
async def tani(request: Request, dosya: UploadFile = File(...),
               db: Session = Depends(get_db)):
    _ortak_ayarlar()
    from app import urunler
    urun = urunler.VARSAYILAN

    ham = await dosya.read()
    if not ham:
        return templates.TemplateResponse(
            request, 'bocek.html',
            _baglam(request, 'teshis', hata='Dosya boş görünüyor.'))

    # cv2.imread Türkçe karakterli yolda çalışmaz; bellekten çözüyoruz.
    frame = cv2.imdecode(np.frombuffer(ham, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return templates.TemplateResponse(
            request, 'bocek.html',
            _baglam(request, 'teshis',
                    hata='Görüntü okunamadı — dosya bozuk olabilir.'))

    sonuc = servis.tani(frame, urun)

    # Kutu görsele çizilir: kullanıcı modelin karenin NERESİNE baktığını
    # görmeli. Karede birden çok canlı varsa "hangisini tanıdı" sorusu
    # ancak kutuyla cevaplanır; yanlış yere bakmışsa da anlaşılır.
    from app import dil
    kls = config.STORAGE_DIR / KAYIT_DIZINI
    kls.mkdir(parents=True, exist_ok=True)
    ad = f'{uuid.uuid4().hex[:12]}.jpg'
    cv2.imwrite(str(kls / ad), servis.adaylari_ciz(frame, sonuc, dil.sinif_adi))
    goreli = f'{KAYIT_DIZINI}/{ad}'

    # Kayıt: böcek bulunamasa bile tutulur. "Model bu fotoğrafta bir şey
    # göremedi" bilgisi, modeli değerlendirmek için tespit kadar değerlidir.
    kayit = BocekKaydi(
        gorsel=goreli,
        tur=sonuc.en_iyi.ad if sonuc.en_iyi else '',
        guven=sonuc.en_iyi.guven if sonuc.en_iyi else 0.0,
        kararsiz=sonuc.kararsiz,
        adaylar_json=json.dumps([{'ad': a.ad, 'guven': a.guven}
                                 for a in sonuc.adaylar], ensure_ascii=False),
    )
    db.add(kayit)
    db.commit()
    db.refresh(kayit)

    onermeler = {}
    for a in sonuc.adaylar:
        o = servis.oneri(a.ad, urun)
        if o:
            onermeler[a.ad] = o

    return templates.TemplateResponse(request, 'bocek.html', _baglam(
        request, 'teshis', sonuc=sonuc, onermeler=onermeler,
        kayit=kayit, gorsel=f'/media/{goreli}'))


# ─────────────────────────────────────────────────────────── geçmiş sekmesi
@router.get('/gecmis', response_class=HTMLResponse)
def gecmis(request: Request, tur: str = '', dogrulama: str = '',
           db: Session = Depends(get_db)):
    _ortak_ayarlar()
    q = db.query(BocekKaydi)
    if tur:
        q = q.filter(BocekKaydi.tur == tur)
    if dogrulama:
        q = q.filter(BocekKaydi.dogrulama == dogrulama)
    kayitlar = q.order_by(BocekKaydi.id.desc()).limit(200).all()

    # İsabet TÜM kayıtlardan hesaplanır, süzülmüş listeden değil; yoksa
    # "yanlış" süzgecinde isabet %0 görünür ve yanıltır.
    ozet = isabet(db.query(BocekKaydi).all())

    return templates.TemplateResponse(request, 'bocek_gecmis.html', _baglam(
        request, 'gecmis', kayitlar=kayitlar, ozet=ozet,
        secili={'tur': tur, 'dogrulama': dogrulama},
        dogrulama_secenekleri=DOGRULAMA))


@router.post('/kayit/{kayit_id}/dogrula')
def dogrula(kayit_id: int, dogrulama: str = Form(''), dogru_tur: str = Form(''),
            notlar: str = Form(''), nereye: str = Form('gecmis'),
            db: Session = Depends(get_db)):
    """Kullanıcının 'doğru / yanlış / listede yok' işareti.

    Modelin ilk cevabı (`tur`) DEĞİŞTİRİLMEZ — isabet ancak modelin ne
    dediği ile gerçeğin karşılaştırılmasıyla ölçülebilir.
    """
    kayit = db.get(BocekKaydi, kayit_id)
    if kayit is None:
        return RedirectResponse('/bocek/gecmis', status_code=303)

    kayit.dogrulama = dogrulama if dogrulama in DOGRULAMA else ''
    kayit.dogru_tur = dogru_tur.strip() if kayit.dogrulama == 'yanlis' else ''
    kayit.notlar = notlar.strip()
    db.commit()
    return RedirectResponse('/bocek/gecmis' if nereye == 'gecmis' else '/bocek',
                            status_code=303)


@router.post('/kayit/{kayit_id}/sil')
def sil(kayit_id: int, db: Session = Depends(get_db)):
    kayit = db.get(BocekKaydi, kayit_id)
    if kayit:
        if kayit.gorsel:
            try:
                (config.STORAGE_DIR / kayit.gorsel).unlink(missing_ok=True)
            except OSError:
                pass          # dosya gitmişse kayıt yine de silinmeli
        db.delete(kayit)
        db.commit()
    return RedirectResponse('/bocek/gecmis', status_code=303)
