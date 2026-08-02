"""Aday etiket inceleme — HTTP katmanı.

İş mantığı servis.py'de. Burada yalnızca istek çözme, yanıt kurma ve
hata mesajı üretme var.
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app import config, urunler
from app.moduller.etiket import servis

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/etiket', tags=['etiket'])
# İKİ dizin: modülün kendi şablonları + çekirdeğin base.html'i.
# Yalnızca modül dizini verilirse {% extends "base.html" %} çözülemez.
templates = Jinja2Templates(directory=[
    str(Path(__file__).parent / 'templates'),
    str(config.BASE_DIR / 'app' / 'templates'),
])


def _ortak_ayarlar():
    """Çekirdeğin filtre ve global'lerini (menü, sinif, tarih) taşır."""
    from app.moduller import sablon_ayarla
    sablon_ayarla(templates)

DATASET_KOK = config.BASE_DIR / 'datasets'


def _paket_ya_da_404(urun: str, ad: str) -> servis.Paket:
    p = servis.paket_bul(DATASET_KOK, urun or '', ad)
    if p is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paket bulunamadı: {urun or '-'}/{ad}. "
                   'İnceleme için paket klasöründe images/ ve '
                   'labels_aday/ (veya labels_organ/) bulunmalıdır.')
    return p


@router.get('')
def liste(request: Request):
    _ortak_ayarlar()
    paketler = servis.paketleri_bul(DATASET_KOK)
    return templates.TemplateResponse(request, 'etiket_liste.html', {
        'paketler': paketler,
        'dataset_kok': str(DATASET_KOK),
    })


@router.get('/{urun}/{ad}')
def paket(request: Request, urun: str, ad: str, sirala: str = 'guven'):
    _ortak_ayarlar()
    p = _paket_ya_da_404(urun, ad)
    return templates.TemplateResponse(request, 'etiket_paket.html', {
        'paket': p,
        'kareler': servis.kareler(p, sirala),
        'sirala': sirala,
        'uyarilar': servis.kalite_denetimi(p),
    })


@router.get('/{urun}/{ad}/kare/{kare}')
def kare(request: Request, urun: str, ad: str, kare: str,
         sirala: str = 'guven'):
    _ortak_ayarlar()
    p = _paket_ya_da_404(urun, ad)
    if servis.kare_yolu(p, kare) is None:
        raise HTTPException(status_code=404, detail=f'Kare yok: {kare}')
    liste_ = servis.kareler(p, sirala)
    adlar = [k.ad for k in liste_]
    i = adlar.index(kare) if kare in adlar else 0
    taban = f'/etiket/{urun}/{ad}'
    kutular = [k.__dict__ for k in servis.kare_kutulari(p, kare)]
    return templates.TemplateResponse(request, 'etiket_duzenle.html', {
        'paket': p, 'kare': kare,
        'kutular': kutular,
        # Tuval editorunun ihtiyaci olan her sey TEK sozlukte; sablon
        # icinde sozluk kurmak Jinja'da kirilgan (tojson patliyor).
        'ayar': {
            'kutular': kutular,
            'siniflar': p.siniflar,
            'en_kucuk_kenar': servis.EN_KUCUK_KENAR_PX,
            'goruntu_yolu': f'{taban}/goruntu/{kare}',
            'kaydet_yolu': f'{taban}/kare/{kare}/kaydet',
            'kare_yolu_kalibi': f'{taban}/kare/__KARE__?sirala={sirala}',
        },
        'hastalik_kutulari': [k.__dict__ for k in
                              servis.kare_kutulari(p, kare, 'hastalik')],
        'onayli': bool(liste_[i].onayli) if liste_ else False,
        'sira': i + 1, 'toplam': len(liste_),
        'onceki': adlar[i - 1] if i > 0 else None,
        'sonraki': adlar[i + 1] if i + 1 < len(adlar) else None,
        'sirala': sirala,
        'en_kucuk_kenar': servis.EN_KUCUK_KENAR_PX,
    })


@router.get('/{urun}/{ad}/goruntu/{kare}')
def goruntu(urun: str, ad: str, kare: str):
    """Kare dosyasını servis eder — yol denetimi servis katmanında."""
    p = _paket_ya_da_404(urun, ad)
    yol = servis.kare_yolu(p, kare)
    if yol is None:
        raise HTTPException(status_code=404, detail=f'Kare yok: {kare}')
    return FileResponse(yol)


@router.post('/{urun}/{ad}/kare/{kare}/kaydet')
def kaydet(urun: str, ad: str, kare: str, veri: dict = Body(...)):
    p = _paket_ya_da_404(urun, ad)
    ham = veri.get('kutular')
    if not isinstance(ham, list):
        raise HTTPException(status_code=400,
                            detail="Gövdede 'kutular' listesi bekleniyor.")
    kutular = []
    for k in ham:
        try:
            kutular.append(servis.Kutu(
                sinif=int(k['sinif']), cx=float(k['cx']), cy=float(k['cy']),
                w=float(k['w']), h=float(k['h'])))
        except (KeyError, TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f'Bozuk kutu: {k!r}. Beklenen alanlar: '
                       'sinif, cx, cy, w, h')
    katman = 'hastalik' if veri.get('katman') == 'hastalik' else 'organ'
    ok, mesaj = servis.kare_kaydet(p, kare, kutular, katman)
    if not ok:
        raise HTTPException(status_code=400, detail=mesaj)
    if veri.get('onayla'):
        servis.onay_ver(p, kare, True)
    return {'ok': True, 'mesaj': mesaj,
            'sonraki': servis.sonraki_kare(p, kare)}


@router.post('/{urun}/{ad}/kare/{kare}/onay')
def onay(urun: str, ad: str, kare: str, veri: dict = Body(default={})):
    p = _paket_ya_da_404(urun, ad)
    onayli = veri.get('onayli', True) is not False
    if not servis.onay_ver(p, kare, onayli):
        raise HTTPException(status_code=404, detail=f'Kare yok: {kare}')
    return {'ok': True, 'onayli': onayli,
            'sonraki': servis.sonraki_kare(p, kare)}


@router.post('/{urun}/{ad}/aktar')
def aktar(urun: str, ad: str, veri: dict = Body(default={})):
    p = _paket_ya_da_404(urun, ad)
    yalniz = veri.get('yalniz_onayli', True) is not False
    sonuc = servis.disa_aktar(p, yalniz)
    if not sonuc['yazilan']:
        return JSONResponse(status_code=400, content={
            'ok': False,
            'mesaj': 'Hiç kare yazılmadı — henüz onaylanmış kare yok. '
                     'Kareleri onaylayın ya da "hepsini aktar" seçin.'})
    return {'ok': True, **sonuc,
            'mesaj': f"{sonuc['yazilan']} kare labels/ altına yazıldı "
                     f"({sonuc['bos']} tanesi kutusuz — bu geçerli bir "
                     'negatif örnektir).'}
