"""İki modeli AYNI test setinde karşılaştırır ve sınıf bazında rapor verir.

NEDEN GEREKLİ?
    "Yeni model daha iyi" demek için eski modelin eğitim sonundaki mAP'ı ile
    yeni modelinkini kıyaslamak YANLIŞTIR: veri seti değiştiyse (yeni kaynak
    eklendi, etiketler düzeltildi) iki sayı farklı ölçütlerden gelir.

    Tek geçerli kıyas, İKİ MODELİ DE AYNI test setinde çalıştırmaktır.

NEDEN SINIF BAZINDA?
    Toplam mAP artarken tek tek sınıflar gerileyebilir. Özellikle ince ayarda
    yeni veri belirli sınıflara yoğunlaşırsa diğerleri unutulur (catastrophic
    forgetting). Ortalama bunu gizler; sınıf bazlı tablo gizlemez.

KULLANIM
    python scripts/model_karsilastir.py \\
        --eski models/best.pt \\
        --yeni runs/train/strawberry_ince_ayar/weights/best.pt \\
        --data configs/strawberry_data.yaml --split test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Bu farkın altındaki değişim gürültü sayılır (aynı model iki kez ölçülse bile
# veri yükleme sırası/AMP nedeniyle küçük oynamalar olur).
ONEMLI_FARK = 0.005


def olc(model_yolu: str, data_yaml: str, split: str, imgsz: int, batch: int) -> dict:
    """Modeli değerlendirir; genel ve sınıf bazlı metrikleri döner."""
    from ultralytics import YOLO

    logger.info(f'Değerlendiriliyor: {model_yolu}  (split={split})')
    model = YOLO(model_yolu)
    r = model.val(data=str(Path(data_yaml).resolve()), split=split,
                  imgsz=imgsz, batch=batch, verbose=False, plots=False)

    adlar = model.names
    sinif = {}
    # ap_class_index: metriklerin hangi sınıfa ait olduğu (veri olmayan sınıf atlanır)
    for sira, cid in enumerate(getattr(r.box, 'ap_class_index', [])):
        sinif[adlar[int(cid)]] = {
            'mAP50': float(r.box.ap50[sira]),
            'mAP50-95': float(r.box.ap[sira]),
            'precision': float(r.box.p[sira]),
            'recall': float(r.box.r[sira]),
        }
    return {
        'model': model_yolu,
        'genel': {
            'mAP50': float(r.box.map50),
            'mAP50-95': float(r.box.map),
            'precision': float(r.box.mp),
            'recall': float(r.box.mr),
        },
        'sinif': sinif,
    }


def _ok(fark: float) -> str:
    if fark > ONEMLI_FARK:
        return '▲ iyileşti'
    if fark < -ONEMLI_FARK:
        return '▼ GERİLEDİ'
    return '≈ değişmedi'


def rapor(eski: dict, yeni: dict, olcut: str = 'mAP50-95') -> dict:
    print()
    print('=' * 78)
    print(f'MODEL KARŞILAŞTIRMASI  (aynı test seti, ölçüt: {olcut})')
    print('=' * 78)
    print(f'Eski : {eski["model"]}')
    print(f'Yeni : {yeni["model"]}')
    print()

    print(f'{"GENEL":24} {"eski":>8} {"yeni":>8} {"fark":>8}')
    print('-' * 78)
    for k in ('mAP50', 'mAP50-95', 'precision', 'recall'):
        e, y = eski['genel'][k], yeni['genel'][k]
        print(f'{k:24} {e:>8.3f} {y:>8.3f} {y - e:>+8.3f}   {_ok(y - e)}')

    print()
    print(f'{"SINIF":24} {"eski":>8} {"yeni":>8} {"fark":>8}')
    print('-' * 78)
    gerileyen, iyilesen = [], []
    for ad in sorted(set(eski['sinif']) | set(yeni['sinif'])):
        e = eski['sinif'].get(ad, {}).get(olcut)
        y = yeni['sinif'].get(ad, {}).get(olcut)
        if e is None or y is None:
            durum = 'yalnızca bir modelde var'
            print(f'{ad:24} {"-" if e is None else f"{e:.3f}":>8} '
                  f'{"-" if y is None else f"{y:.3f}":>8} {"":>8}   {durum}')
            continue
        fark = y - e
        print(f'{ad:24} {e:>8.3f} {y:>8.3f} {fark:>+8.3f}   {_ok(fark)}')
        (gerileyen if fark < -ONEMLI_FARK else
         iyilesen if fark > ONEMLI_FARK else []).append((ad, fark))

    print()
    print('=' * 78)
    genel_fark = yeni['genel'][olcut] - eski['genel'][olcut]
    if gerileyen:
        print(f'⚠️  {len(gerileyen)} sınıf GERİLEDİ:')
        for ad, f in sorted(gerileyen, key=lambda x: x[1]):
            print(f'      {ad:24} {f:+.3f}')
        print('    Olası sebep: yeni veri bu sınıflarda seyrek → unutma')
        print('    (catastrophic forgetting). Çözüm: o sınıflardan örnek ekleyin')
        print('    veya daha düşük öğrenme oranıyla (lr0) tekrar deneyin.')
    if iyilesen:
        print(f'✅ {len(iyilesen)} sınıf iyileşti '
              f'(en çok: {max(iyilesen, key=lambda x: x[1])[0]})')

    print()
    if genel_fark > ONEMLI_FARK and not gerileyen:
        print(f'KARAR: Yeni model açık şekilde daha iyi ({olcut} {genel_fark:+.3f}). '
              'Dağıtıma alınabilir.')
    elif genel_fark > ONEMLI_FARK:
        print(f'KARAR: Genel iyileşme var ({genel_fark:+.3f}) ama bazı sınıflar '
              'geriledi. Geri gelen sınıflar sizin için kritikse dağıtmayın.')
    elif genel_fark < -ONEMLI_FARK:
        print(f'KARAR: Yeni model DAHA KÖTÜ ({genel_fark:+.3f}). Eski modeli koruyun.')
    else:
        print('KARAR: Anlamlı fark yok. Yeni modeli dağıtmanın bir kazancı olmaz.')
    print('=' * 78)

    return {'eski': eski, 'yeni': yeni, 'olcut': olcut,
            'genel_fark': genel_fark,
            'gerileyen': [a for a, _ in gerileyen],
            'iyilesen': [a for a, _ in iyilesen]}


def main():
    ap = argparse.ArgumentParser(description='İki modeli aynı test setinde karşılaştırır')
    ap.add_argument('--eski', required=True, help='Mevcut/üretimdeki model (.pt)')
    ap.add_argument('--yeni', required=True, help='Yeni eğitilen model (.pt)')
    ap.add_argument('--data', default='configs/strawberry_data.yaml')
    ap.add_argument('--split', default='test', choices=['val', 'test'],
                    help='test önerilir: val eğitim sırasında model seçimi için kullanıldı')
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--olcut', default='mAP50-95', choices=['mAP50', 'mAP50-95'])
    ap.add_argument('--json', default='', help='Sonucu JSON olarak kaydet')
    a = ap.parse_args()

    for yol in (a.eski, a.yeni):
        if not Path(yol).exists():
            logger.error(f'Model bulunamadı: {yol}')
            return 1

    eski = olc(a.eski, a.data, a.split, a.imgsz, a.batch)
    yeni = olc(a.yeni, a.data, a.split, a.imgsz, a.batch)
    sonuc = rapor(eski, yeni, a.olcut)

    if a.json:
        Path(a.json).write_text(json.dumps(sonuc, ensure_ascii=False, indent=2),
                                encoding='utf-8')
        logger.info(f'JSON yazıldı: {a.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
