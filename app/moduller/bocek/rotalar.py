"""Böcek teşhis modülünün sayfaları."""

import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app import config
from app.moduller.bocek import servis

router = APIRouter(prefix='/bocek', tags=['bocek'])

templates = Jinja2Templates(directory=[
    str(Path(__file__).parent / 'templates'),
    str(config.BASE_DIR / 'app' / 'templates'),
])

# Yüklenen kare burada tutulur; kullanıcı sonucu görürken fotoğrafını da görsün.
KAYIT_DIZINI = 'bocek'


def _ortak_ayarlar():
    from app.moduller import sablon_ayarla
    sablon_ayarla(templates)


def _baglam(request: Request, **ek):
    """Sayfanın her hâlinde gereken ortak veriler."""
    from app import urunler
    urun = urunler.VARSAYILAN
    return {
        'request': request,
        'hazir': servis.hazir(urun),
        'turler': servis.taniyabildikleri(urun),
        **ek,
    }


@router.get('', response_class=HTMLResponse)
def sayfa(request: Request):
    _ortak_ayarlar()
    return templates.TemplateResponse(request, 'bocek.html', _baglam(request))


@router.post('/tani', response_class=HTMLResponse)
async def tani(request: Request, dosya: UploadFile = File(...)):
    """Makro fotoğraftan tür teşhisi.

    KAYIT YAPILMIYOR (bilerek): bu akış bitki analizi değil, tür sorgusudur.
    Analiz tablosuna yazılsaydı hastalık istatistiklerine ve yaygınlık
    haritasına karışır, "şu serada 12 tespit" sayısı anlamını yitirirdi.
    Kayıt istenirse ayrı tablo açılmalı.
    """
    _ortak_ayarlar()
    from app import urunler
    urun = urunler.VARSAYILAN

    ham = await dosya.read()
    if not ham:
        return templates.TemplateResponse(
            request, 'bocek.html',
            _baglam(request, hata='Dosya boş görünüyor.'))

    # cv2.imread Türkçe karakterli yolda çalışmaz; bellekten çözüyoruz.
    frame = cv2.imdecode(np.frombuffer(ham, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return templates.TemplateResponse(
            request, 'bocek.html',
            _baglam(request, hata='Görüntü okunamadı — dosya bozuk olabilir.'))

    sonuc = servis.tani(frame, urun)

    # Kullanıcı sonucu kendi fotoğrafıyla birlikte görsün
    kls = config.STORAGE_DIR / KAYIT_DIZINI
    kls.mkdir(parents=True, exist_ok=True)
    ad = f'{uuid.uuid4().hex[:12]}.jpg'
    cv2.imwrite(str(kls / ad), frame)

    onermeler = {}
    for a in sonuc.adaylar:
        o = servis.oneri(a.ad, urun)
        if o:
            onermeler[a.ad] = o

    return templates.TemplateResponse(request, 'bocek.html', _baglam(
        request, sonuc=sonuc, onermeler=onermeler,
        gorsel=f'/media/{KAYIT_DIZINI}/{ad}'))
