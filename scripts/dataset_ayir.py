"""Birleşik dataset'i uzman modeller için bağımsız dataset'lere ayırır.

NEDEN?
    Hiyerarşik mimaride her model kendi problem alanında eğitilir. Tek 10
    sınıflı veriden üç dataset SINIF BAZINDA türetilebilir:

        leaf_disease    ← yaprak hastalıkları
        fruit_disease   ← meyve hastalıkları
        fruit_ripeness  ← olgunluk

    organ_detection ve pest_detection için mevcut veride etiket YOKTUR;
    onlar sıfırdan etiketlenmelidir (bkz. docs/MIMARI_GECIS_PLANI.md).

SINIRI BİLİN
    Bu türetme uzman modelleri TAM GÖRÜNTÜYLE eğitir; oysa çalışma anında
    ROI kırpıntısı görecekler. Aradaki fark doğruluğu düşürür. Organ
    etiketlemesi bittikten sonra `--roi` kipiyle kırpıntılardan yeniden
    üretmek daha iyi sonuç verir.

BACKGROUND
    İlgisiz sınıfların kutuları atılır; görüntü SİLİNMEZ, etiketi boşalır.
    "Bu görüntüde bu sınıflar yok" bilgisi yanlış pozitifleri azaltır.
    Ama tamamı alınırsa background örnekleri veriyi boğar — oran sınırlanır.

KULLANIM
    python scripts/dataset_ayir.py --kuru
    python scripts/dataset_ayir.py --kaynak dataset --hedef datasets
"""

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parent.parent

# --- Ürün kapsamı (çok bitkili kurulum) -------------------------------------
# Eğitim yapılandırması her ürünün kendi klasöründedir:
#   configs/urunler/<urun>/veri.yaml
# Eski kurulumlarda configs/strawberry_data.yaml'a düşülür.
def veri_yapilandirmasi(urun: str = None) -> Path:
    import os
    urun = urun or os.environ.get('VARSAYILAN_URUN', 'cilek')
    yeni = KOK / 'configs' / 'urunler' / urun / 'veri.yaml'
    return yeni if yeni.exists() else KOK / 'configs' / 'strawberry_data.yaml'


# Hangi uzman dataset hangi master sınıfları alır.
# Gray Mold hem yaprakta hem meyvede görülür → ikisine de girer.
AYRIM = {
    'leaf_disease': ['Angular Leafspot', 'Leaf Spot', 'Powdery Mildew Leaf', 'Gray Mold'],
    'fruit_disease': ['Anthracnose Fruit Rot', 'Powdery Mildew Fruit',
                      'Blossom Blight', 'Gray Mold'],
    'fruit_ripeness': ['strawberry_unripe', 'strawberry_semi_ripe', 'strawberry_ripe'],
}


def master_siniflar(yapilandirma: Path) -> dict:
    cfg = yaml.safe_load(yapilandirma.read_text(encoding='utf-8')) or {}
    isimler = cfg.get('names', {})
    if isinstance(isimler, list):
        return {i: a for i, a in enumerate(isimler)}
    return {int(k): v for k, v in isimler.items()}


def bolum_ciftleri(kaynak: Path):
    """[(bolum, goruntu_yolu, etiket_yolu)] — tüm kaynak klasörlerini tarar."""
    for ds in sorted(kaynak.iterdir()):
        if not ds.is_dir() or ds.name.startswith('.'):
            continue
        for bolum in ('train', 'valid', 'test', ''):
            d = (ds / bolum) if bolum else ds
            g, e = d / 'images', d / 'labels'
            if not (g.is_dir() and e.is_dir()):
                continue
            hedef_bolum = bolum or 'train'
            for f in sorted(g.iterdir()):
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                yield hedef_bolum, f, e / f'{f.stem}.txt'


def ayir(kaynak: Path, hedef_kok: Path, master: dict, arka_plan_orani: float,
         kuru: bool, tohum: int = 0) -> int:
    ters = {a: i for i, a in master.items()}
    rastgele = random.Random(tohum)

    for ds_adi, sinif_adlari in AYRIM.items():
        eski_idler = {ters[a] for a in sinif_adlari if a in ters}
        if not eski_idler:
            print(f'⚠️ {ds_adi}: master listede eşleşen sınıf yok, atlandı')
            continue
        # Yeni ID'ler 0..n-1 (dataset bağımsız olmalı)
        yeni_id = {ters[a]: i for i, a in enumerate(sinif_adlari) if a in ters}
        yeni_adlar = [a for a in sinif_adlari if a in ters]

        hedef = hedef_kok / ds_adi
        sayac = Counter()
        icerikli, arka_plan_adaylari = [], []

        for bolum, gorsel, etiket in bolum_ciftleri(kaynak):
            satirlar = []
            if etiket.exists():
                for s in etiket.read_text(encoding='utf-8').splitlines():
                    p = s.split()
                    if len(p) < 5:
                        continue
                    cid = int(float(p[0]))
                    if cid in yeni_id:
                        satirlar.append(f'{yeni_id[cid]} ' + ' '.join(p[1:5]))
            (icerikli if satirlar else arka_plan_adaylari).append(
                (bolum, gorsel, satirlar))

        # Background sayısını sınırla: veriyi boğmasın
        azami_arka = int(len(icerikli) * arka_plan_orani)
        rastgele.shuffle(arka_plan_adaylari)
        secilen = icerikli + arka_plan_adaylari[:azami_arka]

        print(f'\n■ {ds_adi}  ({len(yeni_adlar)} sınıf: {", ".join(yeni_adlar)})')
        print(f'   içerikli görüntü : {len(icerikli)}')
        print(f'   background       : {min(azami_arka, len(arka_plan_adaylari))} '
              f'(havuzda {len(arka_plan_adaylari)}, oran {arka_plan_orani:.0%})')

        if not kuru:
            for bolum in ('train', 'valid', 'test'):
                (hedef / bolum / 'images').mkdir(parents=True, exist_ok=True)
                (hedef / bolum / 'labels').mkdir(parents=True, exist_ok=True)

        for bolum, gorsel, satirlar in secilen:
            sayac[bolum] += 1
            for s in satirlar:
                sayac[f'kutu_{yeni_adlar[int(s.split()[0])]}'] += 1
            if kuru:
                continue
            # Ad çakışmasını önlemek için kaynak klasör adı öne eklenir
            kaynak_ad = gorsel.parent.parent.parent.name[:18].replace(' ', '_')
            yeni_ad = f'{kaynak_ad}_{gorsel.name}'
            shutil.copy2(gorsel, hedef / bolum / 'images' / yeni_ad)
            (hedef / bolum / 'labels' / f'{Path(yeni_ad).stem}.txt').write_text(
                '\n'.join(satirlar) + ('\n' if satirlar else ''), encoding='utf-8')

        for b in ('train', 'valid', 'test'):
            print(f'   {b:6}: {sayac[b]:>6} görüntü')
        for a in yeni_adlar:
            print(f'      {a:24} {sayac[f"kutu_{a}"]:>6} kutu')

        if not kuru:
            # 'path' BILEREK YAZILMAZ: Ultralytics onu bulamayinca yaml'in
            # kendi klasorunu kok sayar. Mutlak yol yazilsaydi klasor tasininca
            # veya Colab'de acilinca "images not found" verirdi.
            (hedef / 'data.yaml').write_text(yaml.dump({
                'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
                'nc': len(yeni_adlar),
                'names': {i: a for i, a in enumerate(yeni_adlar)},
            }, allow_unicode=True, sort_keys=False), encoding='utf-8')

    if kuru:
        print('\n[KURU] Hiçbir dosya yazılmadı. Uygulamak için --kuru olmadan çalıştırın.')
    return 0


def paketle(hedef_kok: Path, adlar) -> None:
    """Her dataset'i ayrı zip yapar — Colab'e yalnızca eğitilecek olan yüklenir.

    NEDEN AYRI ZIP: tek büyük arşiv her eğitimde baştan yüklenir/açılır. Ayrı
    paketlerde yaprak modelini eğitirken meyve verisi hiç taşınmaz; yükleme ve
    açma süresi belirgin düşer.

    compresslevel=1: JPEG zaten sıkıştırılmıştır, yüksek seviye yalnızca
    süre harcar.
    """
    import zipfile
    for ad in adlar:
        d = hedef_kok / ad
        if not d.is_dir():
            continue
        zip_yolu = hedef_kok / (ad + '.zip')
        dosyalar = [f for f in d.rglob('*') if f.is_file()]
        with zipfile.ZipFile(zip_yolu, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for f in dosyalar:
                z.write(f, f.relative_to(hedef_kok))
        mb = zip_yolu.stat().st_size / 1e6
        print(f'   [paket] {zip_yolu.name}  {mb:.0f} MB  ({len(dosyalar)} dosya)')


def main():
    ap = argparse.ArgumentParser(description='Birleşik dataset → uzman dataset\'ler')
    ap.add_argument('--kaynak', default=str(KOK / 'dataset'))
    ap.add_argument('--hedef', default=None,
                    help='Hedef kök (boşsa datasets/<urun>)')
    ap.add_argument('--urun', default=None, help='Ürün kapsamı (varsayılan: cilek)')
    ap.add_argument('--yapilandirma', default=None,
                    help='Ana sınıf listesi (boşsa ürünün veri.yaml dosyası)')
    ap.add_argument('--arka-plan-orani', type=float, default=0.15,
                    help='İçerikli görüntü sayısının kaçta kaçı kadar background alınsın')
    ap.add_argument('--kuru', action='store_true')
    ap.add_argument('--paketle', action='store_true',
                    help="Her dataset'i ayri zip yapar (Colab'e yuklemek icin)")
    a = ap.parse_args()

    import os
    urun_ad = a.urun or os.environ.get('VARSAYILAN_URUN', 'cilek')
    hedef_kok = Path(a.hedef) if a.hedef else (KOK / 'datasets' / urun_ad)
    kaynak = Path(a.kaynak)
    if not kaynak.exists():
        print(f'Kaynak yok: {kaynak}')
        return 1
    master = master_siniflar(Path(a.yapilandirma) if a.yapilandirma
                             else veri_yapilandirmasi(a.urun))
    if not master:
        print('Master sınıf listesi okunamadı')
        return 1

    print(f'Kaynak : {kaynak}')
    print(f'Hedef  : {hedef_kok}')
    print(f'Ürün   : {urun_ad}')
    print(f'Master : {len(master)} sınıf')
    sonuc = ayir(kaynak, hedef_kok, master, a.arka_plan_orani, a.kuru)
    if a.paketle and not a.kuru:
        print(chr(10) + 'Paketleniyor...')
        paketle(hedef_kok, AYRIM.keys())
    return sonuc


if __name__ == '__main__':
    sys.exit(main())
