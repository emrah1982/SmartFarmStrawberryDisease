"""DINOv2 sınıflandırıcı eğitimi (dondurulmuş omurga + doğrusal başlık).

HEDEF VERİ
    datasets/<urun>/<ad>/{train,val,test}/<sinif>/*.jpg
    (scripts/siniflandirma_paketi.py bu düzeni üretir)

NEDEN DONDURULMUŞ OMURGA?
    DINOv2 kendi kendine denetimli olarak devasa veriyle eğitilmiştir;
    öznitelikleri güçlüdür. Küçük veri setinde tüm ağı açmak aşırı
    öğrenmeye gider. Önce yalnızca son katman eğitilir; yetmezse
    --katman-ac ile son N encoder bloğu düşük öğrenme oranıyla açılır.

SINIF SIRASI
    Klasör adlarının ALFABETİK sırasıdır ve kaydedilen modele gömülür.
    diseased_cluster=0, healthy_cluster=1
    Klasör adı sonradan değişirse eski model geçersiz olur.

⚠️ SADECE ACCURACY'E BAKMAYIN
    Dengesiz veride model her şeye çoğunluk sınıfını diyerek yüksek
    accuracy üretir. Bu betik precision/recall/F1 ve karışıklık matrisini
    her zaman basar; ayrıca "hep çoğunluk de" taban çizgisini de yazar ki
    modelin ondan iyi olup olmadığı görülsün.

ÇALIŞTIRMA YERİ
    torch + transformers gerekir; bu depoda kurulu DEĞİL.
    Colab: pip install -q transformers

Kullanım:
    python scripts/dinov2_egit.py datasets/findik/cotanak_saglik \
        --cikti /content/drive/MyDrive/models/dinov2_cotanak --epok 15
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

VARSAYILAN_MODEL = 'facebook/dinov2-small'
BOLUMLER = ('train', 'val', 'test')


def siniflari_bul(kok: Path) -> list:
    """Sınıf adları = train/ altındaki klasörler, ALFABETİK.

    Sıra burada sabitlenir; HuggingFace imagefolder da aynı sırayı
    kullanır. Kaydedilen modelin id2label'ı buna bağlıdır.
    """
    t = kok / 'train'
    if not t.is_dir():
        return []
    return sorted(d.name for d in t.iterdir() if d.is_dir())


def bolum_sayimi(kok: Path, siniflar: list) -> dict:
    say = {}
    for b in BOLUMLER:
        c = Counter()
        for s in siniflar:
            d = kok / b / s
            if d.is_dir():
                c[s] = sum(1 for p in d.iterdir()
                           if p.suffix.lower() in GORUNTU_UZANTI)
        if sum(c.values()):
            say[b] = c
    return say


def taban_cizgisi(sayim: dict, bolum: str = 'test') -> dict:
    """'Hep çoğunluk sınıfını söyle' stratejisinin doğruluğu.

    Model bunu geçemiyorsa hiçbir şey öğrenmemiştir. Rapora yazılır.
    """
    c = sayim.get(bolum)
    if not c or not sum(c.values()):
        return {}
    ad, n = c.most_common(1)[0]
    return {'sinif': ad, 'dogruluk': n / sum(c.values())}


def rapor_metni(kok, siniflar, sayim, taban, olcum=None, karisiklik=None):
    s = [f'# DINOv2 sınıflandırıcı — {kok.name}', '',
         '## Sınıflar (sıra ALFABETİK, modele gömülü)', '',
         '| ID | sınıf |', '|---|---|']
    for i, ad in enumerate(siniflar):
        s.append(f'| {i} | `{ad}` |')
    s += ['', '## Bölme', '',
          '| bölüm | ' + ' | '.join(siniflar) + ' | toplam |',
          '|---|' + '---|' * (len(siniflar) + 1)]
    for b in BOLUMLER:
        if b in sayim:
            c = sayim[b]
            s.append(f'| {b} | ' + ' | '.join(str(c[k]) for k in siniflar)
                     + f' | {sum(c.values())} |')
    if taban:
        s += ['', '## Taban çizgisi', '',
              f"Hep `{taban['sinif']}` demek → doğruluk "
              f"**{taban['dogruluk']:.4f}**", '',
              'Model bu sayıyı belirgin şekilde geçmiyorsa hiçbir şey',
              'öğrenmemiştir. Yalnızca accuracy\'e bakmak bu yüzden yanıltır.']
    if olcum:
        s += ['', '## Test sonucu', '',
              '| sınıf | precision | recall | f1 | destek |',
              '|---|---|---|---|---|']
        for ad in siniflar:
            m = olcum.get(ad, {})
            s.append(f"| `{ad}` | {m.get('precision', 0):.4f} | "
                     f"{m.get('recall', 0):.4f} | {m.get('f1-score', 0):.4f} | "
                     f"{int(m.get('support', 0))} |")
        if 'accuracy' in olcum:
            s.append(f"\nDoğruluk: **{olcum['accuracy']:.4f}**")
            if taban:
                fark = olcum['accuracy'] - taban['dogruluk']
                s.append(f"Taban çizgisine göre: **{fark:+.4f}**")
    if karisiklik is not None:
        s += ['', '## Karışıklık matrisi', '',
              '| gerçek \\ tahmin | ' + ' | '.join(siniflar) + ' |',
              '|---|' + '---|' * len(siniflar)]
        for ad, satir in zip(siniflar, karisiklik):
            s.append(f'| **{ad}** | ' + ' | '.join(str(int(x)) for x in satir)
                     + ' |')
    s += ['', '## ⚠️ Sızıntı notu', '',
          'Aynı ağacın farklı açıdan çekilmiş kareleri elenemedi (EXIF ve',
          'kaynak kimliği yok). Test doğruluğunu **üst sınır** olarak okuyun.',
          'Bkz. `OKUBENI.md`.', '']
    return '\n'.join(s)


def egit(kok: Path, cikti: Path, model_adi: str, epok: int, yigin: int,
         ogrenme_orani: float, katman_ac: int, ince_epok: int,
         ince_ogrenme: float, siniflar: list):
    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import classification_report, confusion_matrix
    from torchvision.transforms import (CenterCrop, ColorJitter, Compose,
                                        Normalize, RandomHorizontalFlip,
                                        RandomResizedCrop, RandomRotation,
                                        Resize, ToTensor)
    from transformers import (AutoImageProcessor, Dinov2ForImageClassification,
                              Trainer, TrainingArguments)

    islemci = AutoImageProcessor.from_pretrained(model_adi)
    id2label = {i: s for i, s in enumerate(siniflar)}
    model = Dinov2ForImageClassification.from_pretrained(
        model_adi, num_labels=len(siniflar), id2label=id2label,
        label2id={s: i for i, s in id2label.items()},
        ignore_mismatched_sizes=True)

    veri = load_dataset('imagefolder', data_dir=str(kok))
    # HuggingFace kendi sırasını kullanır — bizimkiyle aynı mı, DOĞRULA.
    hf = veri['train'].features['label'].names
    if list(hf) != list(siniflar):
        raise SystemExit(
            f'⛔ Sınıf sırası uyuşmuyor.\n  bizim : {siniflar}\n  hf    : {hf}\n'
            'Model id2label yanlış olurdu; klasör adlarını kontrol edin.')

    kenar = islemci.size.get('height') or islemci.size.get('shortest_edge')
    ort, std = islemci.image_mean, islemci.image_std
    # ColorJitter ÖLÇÜLÜ: renk değişimi hastalığın asıl belirtisi, aşırı
    # oynatmak modelin gerçek renk sinyalini öğrenmesini bozar.
    egitim_donusum = Compose([
        RandomResizedCrop(kenar, scale=(0.75, 1.0)), RandomHorizontalFlip(),
        RandomRotation(15), ColorJitter(brightness=0.15, contrast=0.15,
                                        saturation=0.10),
        ToTensor(), Normalize(mean=ort, std=std)])
    olcum_donusum = Compose([Resize(kenar + 32), CenterCrop(kenar),
                             ToTensor(), Normalize(mean=ort, std=std)])

    def _uygula(donusum):
        def f(yigin_):
            yigin_['pixel_values'] = [donusum(g.convert('RGB'))
                                      for g in yigin_['image']]
            return yigin_
        return f

    veri['train'].set_transform(_uygula(egitim_donusum))
    for b in ('validation', 'val', 'test'):
        if b in veri:
            veri[b].set_transform(_uygula(olcum_donusum))
    dogrulama = veri.get('validation') or veri.get('val')

    def harmanla(ogeler):
        return {'pixel_values': torch.stack([o['pixel_values'] for o in ogeler]),
                'labels': torch.tensor([o['label'] for o in ogeler])}

    def olc(tahmin):
        y = tahmin.predictions.argmax(1)
        return {'accuracy': float((y == tahmin.label_ids).mean())}

    def kos(ad, lr, ep):
        args = TrainingArguments(
            output_dir=str(cikti / ad), eval_strategy='epoch',
            save_strategy='epoch', learning_rate=lr,
            per_device_train_batch_size=yigin, per_device_eval_batch_size=yigin,
            num_train_epochs=ep, weight_decay=0.01,
            load_best_model_at_end=True, metric_for_best_model='accuracy',
            greater_is_better=True, fp16=torch.cuda.is_available(),
            logging_steps=20, save_total_limit=2, report_to='none')
        t = Trainer(model=model, args=args, train_dataset=veri['train'],
                    eval_dataset=dogrulama, data_collator=harmanla,
                    compute_metrics=olc)
        t.train()
        return t

    print('\n=== 1. AŞAMA — omurga DONDURULDU, yalnızca başlık eğitiliyor ===')
    for p in model.dinov2.parameters():
        p.requires_grad = False
    egitici = kos('asama1', ogrenme_orani, epok)

    if katman_ac > 0:
        print(f'\n=== 2. AŞAMA — son {katman_ac} blok açıldı, lr={ince_ogrenme} ===')
        for p in model.dinov2.parameters():
            p.requires_grad = False
        for blok in model.dinov2.encoder.layer[-katman_ac:]:
            for p in blok.parameters():
                p.requires_grad = True
        egitici = kos('asama2', ince_ogrenme, ince_epok)

    print('\n=== TEST ===')
    test = veri.get('test')
    if test is None:
        print('  test bölümü yok, atlandı')
        return None, None
    ct = egitici.predict(test)
    y = ct.predictions.argmax(1)
    rapor = classification_report(ct.label_ids, y, target_names=siniflar,
                                  digits=4, output_dict=True, zero_division=0)
    print(classification_report(ct.label_ids, y, target_names=siniflar,
                                digits=4, zero_division=0))
    km = confusion_matrix(ct.label_ids, y)
    print(km)

    egitici.save_model(str(cikti))
    islemci.save_pretrained(str(cikti))
    (cikti / 'siniflar.json').write_text(
        json.dumps({'siniflar': siniflar, 'id2label': id2label},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n✅ Model kaydedildi: {cikti}')
    return rapor, km


def main():
    ap = argparse.ArgumentParser(description='DINOv2 sınıflandırıcı eğitir')
    ap.add_argument('veri', help='train/val/test içeren klasör')
    ap.add_argument('--cikti', default=None, help='Model kayıt dizini')
    ap.add_argument('--model', default=VARSAYILAN_MODEL)
    ap.add_argument('--epok', type=int, default=15)
    ap.add_argument('--yigin', type=int, default=16)
    ap.add_argument('--ogrenme-orani', type=float, default=1e-3,
                    dest='ogrenme_orani')
    ap.add_argument('--katman-ac', type=int, default=0, dest='katman_ac',
                    help='2. aşamada açılacak son encoder bloğu sayısı')
    ap.add_argument('--ince-epok', type=int, default=8, dest='ince_epok')
    ap.add_argument('--ince-ogrenme', type=float, default=1e-5,
                    dest='ince_ogrenme')
    ap.add_argument('--kuru', action='store_true',
                    help='Yalnızca veri raporu; model yüklenmez')
    a = ap.parse_args()

    kok = Path(a.veri)
    if not kok.exists():
        print(f'❌ Yok: {kok}')
        return 1
    siniflar = siniflari_bul(kok)
    if len(siniflar) < 2:
        print(f'⛔ {kok}/train altında en az iki sınıf klasörü olmalı. '
              f'Bulunan: {siniflar}')
        return 1

    sayim = bolum_sayimi(kok, siniflar)
    print('=' * 72)
    print(f'VERİ: {kok}')
    print('=' * 72)
    print('  sınıflar (alfabetik, modele gömülecek): '
          + ', '.join(f'{i}:{s}' for i, s in enumerate(siniflar)))
    for b in BOLUMLER:
        if b in sayim:
            c = sayim[b]
            print(f'  {b:<6} ' + '  '.join(f'{s}={c[s]}' for s in siniflar)
                  + f'   toplam {sum(c.values())}')
    eksik = [b for b in ('train', 'val') if b not in sayim]
    if eksik:
        print(f'  ⛔ Eksik bölüm: {eksik}')
        print("     Sınıflandırmada klasör adı sözleşmedir; 'valid' değil "
              "'val' olmalı.")
        return 1

    taban = taban_cizgisi(sayim)
    if taban:
        print(f"\n  TABAN ÇİZGİSİ: hep '{taban['sinif']}' de → "
              f"doğruluk {taban['dogruluk']:.4f}")
        print('  Model bunu belirgin geçmiyorsa hiçbir şey öğrenmemiştir.')

    cikti = Path(a.cikti) if a.cikti else KOK / 'models' / kok.name / 'dinov2'
    if a.kuru:
        print(f'\n  çıktı olacaktı: {cikti}')
        print('\n(--kuru: model yüklenmedi, eğitim yapılmadı)')
        (kok / 'DINOV2_RAPORU.md').write_text(
            rapor_metni(kok, siniflar, sayim, taban), encoding='utf-8')
        print(f'📄 {kok / "DINOV2_RAPORU.md"} (veri bölümü)')
        return 0

    cikti.mkdir(parents=True, exist_ok=True)
    rapor, km = egit(kok, cikti, a.model, a.epok, a.yigin, a.ogrenme_orani,
                     a.katman_ac, a.ince_epok, a.ince_ogrenme, siniflar)
    (kok / 'DINOV2_RAPORU.md').write_text(
        rapor_metni(kok, siniflar, sayim, taban, rapor, km), encoding='utf-8')
    print(f'📄 {kok / "DINOV2_RAPORU.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
