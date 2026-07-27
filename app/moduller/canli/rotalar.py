"""Canlı akış uç noktaları: sayfa, WebSocket ve REST yedeği.

AKIŞ (geri basınçlı)
    tarayıcı ──kare──▶ sunucu ──model──▶ kutular ──JSON──▶ tarayıcı ──çizim
        ▲                                                        │
        └──────────── bir sonraki kare ancak sonuç gelince ◀──────┘

    Tarayıcı sabit FPS ile göndermez; önceki karenin sonucu gelmeden yenisini
    yollamaz. Böylece sunucu ne kadar hızlıysa akış o hıza kendiliğinden
    uyar — yavaş sunucuda kuyruk birikmez, hızlı sunucuda akış akıcı olur.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app import config
from app.moduller.canli import ayarlar, depo, servis

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/canli', tags=['canli'])

templates = Jinja2Templates(directory=[
    str(Path(__file__).parent / 'templates'),
    str(config.BASE_DIR / 'app' / 'templates'),
])


def _ortak_ayarlar():
    from app.moduller import sablon_ayarla
    sablon_ayarla(templates)


# ─────────────────────────────────────────────────────────────────── sayfa
@router.get('', response_class=HTMLResponse)
def izle(request: Request):
    _ortak_ayarlar()
    from app import main as cekirdek
    return templates.TemplateResponse(request, 'canli/izle.html', {
        'request': request,
        'model_hazir': cekirdek.detector.hazir,
        'seralar': depo.seralar(),
        # Sayfa http'den açıldıysa kullanıcıyı güvenli adrese yönlendirebilmek için
        'https_port': config.HTTPS_PORT,
        'guvenli_hazir': bool(config.SSL_CERT and config.SSL_KEY),
        'ayar': {
            'genislik': ayarlar.GONDERIM_GENISLIK,
            'kalite': ayarlar.GONDERIM_KALITE,
            'en_az_aralik': ayarlar.EN_AZ_ARALIK_MS,
            'otomatik': ayarlar.OTOMATIK_KAYIT,
            'kararlilik': ayarlar.KARARLILIK_KARE,
            'kayit_guven': ayarlar.KAYIT_GUVEN,
        },
    })


# ───────────────────────────────────────────────────────── ortak kare işleme
def _isle(veri: bytes, karar: Optional[servis.KayitKarari],
          sera_id: Optional[int], kaydet: bool) -> dict:
    """Bir kareyi işler ve tarayıcıya dönecek sözlüğü üretir.

    WebSocket ve REST aynı fonksiyonu kullanır: davranış tek yerde tanımlı
    kalır, iki uçtan biri düzeltilip diğeri unutulmaz.
    """
    frame = servis.kare_coz(veri)
    if frame is None:
        return {'tip': 'hata', 'mesaj': 'Kare çözülemedi'}

    sonuc = servis.tespit(frame)
    yanit = {
        'tip': 'sonuc',
        'kutular': [servis.kutu_sozlugu(k) for k in sonuc.kutular],
        'ms': sonuc.sure_ms,
        'bulanik': sonuc.kalite_notu == 'bulanik',
        'keskinlik': round(sonuc.keskinlik, 1),
        'kayit_id': None,
    }

    hedef = None
    if kaydet and sonuc.kutular is not None:
        hedef = 'elle'
    elif karar is not None and ayarlar.OTOMATIK_KAYIT:
        secilen = karar.degerlendir(sonuc.kutular, time.monotonic())
        if secilen is not None:
            hedef = 'otomatik'

    if hedef:
        kayit_id = depo.kare_kaydet(frame, sonuc.kutular, sera_id=sera_id,
                                    kaynak_ad=f'canli-{hedef}')
        if kayit_id:
            yanit['kayit_id'] = kayit_id
            yanit['kayit_tipi'] = hedef
    return yanit


# ─────────────────────────────────────────────────────────────── WebSocket
@router.websocket('/ws')
async def akis(websocket: WebSocket):
    await websocket.accept()
    karar = servis.KayitKarari()
    sera_id: Optional[int] = None
    kaydet_istegi = False

    try:
        while True:
            mesaj = await websocket.receive()
            if mesaj.get('type') == 'websocket.disconnect':
                break

            # Metin mesajları = kontrol (ayar / kaydet isteği)
            if mesaj.get('text') is not None:
                try:
                    veri = json.loads(mesaj['text'])
                except json.JSONDecodeError:
                    continue
                if veri.get('sera_id'):
                    sera_id = int(veri['sera_id'])
                if veri.get('tip') == 'kaydet':
                    kaydet_istegi = True        # sıradaki kare kaydedilsin
                continue

            kare = mesaj.get('bytes')
            if not kare:
                continue

            # Model çağrısı bloklayıcıdır → olay döngüsünü kilitlemesin
            yanit = await run_in_threadpool(_isle, kare, karar, sera_id, kaydet_istegi)
            kaydet_istegi = False
            await websocket.send_text(json.dumps(yanit))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f'Canlı akış hatası: {e}')
        try:
            await websocket.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────── REST yedeği
# Bazı ağlarda/vekil sunucularda WebSocket engellidir. O durumda tarayıcı
# aynı kareyi buraya POST eder; protokol dışında hiçbir şey değişmez.
_oturumlar: Dict[str, servis.KayitKarari] = {}


def _oturum(anahtar: str) -> servis.KayitKarari:
    if anahtar not in _oturumlar:
        if len(_oturumlar) > 20:            # sızıntı olmasın
            _oturumlar.clear()
        _oturumlar[anahtar] = servis.KayitKarari()
    return _oturumlar[anahtar]


@router.post('/kare')
async def kare(kare: UploadFile, oturum: str = Form('varsayilan'),
               sera_id: Optional[str] = Form(None), kaydet: Optional[str] = Form(None)):
    veri = await kare.read()
    yanit = await run_in_threadpool(
        _isle, veri, _oturum(oturum),
        int(sera_id) if sera_id else None, bool(kaydet))
    return JSONResponse(yanit)
