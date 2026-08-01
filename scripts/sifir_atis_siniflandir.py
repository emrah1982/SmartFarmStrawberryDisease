"""Sıfır-atış / az-atış sınıflandırma — ve ikisini ÖLÇEREK karşılaştırma.

╔══════════════════════════════════════════════════════════════════════╗
║ DINOv2 GÖMÜSÜ CLIP METİN PROMPTUYLA KARŞILAŞTIRILAMAZ.               ║
║                                                                       ║
║ DINOv2 yalnızca görüntüyle, kendi kendine denetimli eğitildi. Metin  ║
║ kodlayıcısı YOKTUR ve gömü uzayı CLIP'inkiyle HİZALI DEĞİLDİR. İki   ║
║ ayrı uzaydan vektörün kosinüsü anlamsız bir sayıdır — model yine de  ║
║ bir skor basar, bu yüzden hata fark edilmez.                         ║
║                                                                       ║
║ Metin promptu istiyorsan görüntü kodlayıcısı da CLIP olmalı.         ║
╚══════════════════════════════════════════════════════════════════════╝

İKİ YOL, İKİSİ DE BURADA
    clip   : CLIP görüntü + CLIP metin kodlayıcısı. Etiketli örnek
             GEREKMEZ; sınıfı metinle tarif edersin.
             Zayıflığı: ince taneli tarımsal ayrımda genelde kötüdür —
             CLIP fındık hastalığı görmedi.

    dinov2 : DINOv2 gömüsü + sınıf başına k etiketli örnekten prototip.
             Metin YOK. İnce taneli alanlarda genelde CLIP'ten iyidir.

NEDEN ÖLÇÜYORUZ?
    cotanak_saglik'te etiketli test bölümü var. Hangisinin işe
    yaradığı tahmin edilmez, ÖLÇÜLÜR. Taban çizgisi (hep çoğunluk
    sınıfını söyle) de basılır: onu geçemeyen yöntem hiçbir şey
    öğrenmemiştir.

Kullanım:
    # ikisini de ölç ve karşılaştır
    python scripts/sifir_atis_siniflandir.py datasets/findik/cotanak_saglik \
        --yontem ikisi --k 8

    # yalnızca CLIP, kendi prompt'larınla
    python scripts/sifir_atis_siniflandir.py datasets/findik/cotanak_saglik \
        --yontem clip --prompt-dosyasi promptlar.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
GORUNTU_UZANTI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

CLIP_MODEL = 'openai/clip-vit-large-patch14'
DINOV2_MODEL = 'facebook/dinov2-base'

# Sınıf adı → CLIP metin promptları. Birden çok kalıp ortalanır (prompt
# ensembling); tek cümle CLIP'te oynak sonuç verir.
VARSAYILAN_PROMPT = {
    'healthy_cluster': [
        'a photo of a healthy green hazelnut cluster on the tree',
        'healthy hazelnut husks with normal shape and uniform green color',
        'undamaged hazelnut cluster in an orchard',
    ],
    'diseased_cluster': [
        'a photo of a diseased hazelnut cluster with brown rot',
        'hazelnut husks with discoloration, deformation and dark lesions',
        'damaged and shrivelled hazelnut cluster showing early decay',
    ],
}


def goruntuleri_topla(kok: Path, bolum: str = 'test'):
    """→ ([yol], [gercek_sinif]) — sınıf klasör adından gelir."""
    taban = kok / bolum if (kok / bolum).is_dir() else kok
    yollar, etiketler = [], []
    for d in sorted(p for p in taban.iterdir() if p.is_dir()):
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in GORUNTU_UZANTI:
                yollar.append(p)
                etiketler.append(d.name)
    return yollar, etiketler


def taban_cizgisi(etiketler) -> tuple:
    c = Counter(etiketler)
    ad, n = c.most_common(1)[0]
    return ad, n / len(etiketler)


# ─────────────────────────────────────────────────────────────────────────
# Ölçüm — model gerektirmez, test edilebilir
# ─────────────────────────────────────────────────────────────────────────

def karisiklik(gercek, tahmin, siniflar):
    idx = {s: i for i, s in enumerate(siniflar)}
    m = np.zeros((len(siniflar), len(siniflar)), dtype=int)
    for g, t in zip(gercek, tahmin):
        if g in idx and t in idx:
            m[idx[g], idx[t]] += 1
    return m


def olcumler(gercek, tahmin, siniflar) -> dict:
    """Sınıf başına precision/recall/f1 + genel doğruluk.

    sklearn'e bağımlı olmasın diye elle: bu betik Colab dışında da
    (sklearn kurulu olmayan makinede) ölçüm yapabilmeli.
    """
    m = karisiklik(gercek, tahmin, siniflar)
    dogru = int(np.trace(m))
    toplam = int(m.sum())
    cikti = {'accuracy': dogru / toplam if toplam else 0.0, 'siniflar': {}}
    for i, s in enumerate(siniflar):
        tp = int(m[i, i])
        fp = int(m[:, i].sum()) - tp
        fn = int(m[i, :].sum()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        cikti['siniflar'][s] = {'precision': p, 'recall': r, 'f1': f,
                                'destek': int(m[i, :].sum())}
    cikti['karisiklik'] = m
    return cikti


def prototip_kur(tohum_gomuler, tohum_etiketleri, siniflar):
    """Sınıf prototipi = o sınıfın tohum gömülerinin normalize ortalaması."""
    boyut = tohum_gomuler.shape[1]
    proto = np.zeros((len(siniflar), boyut))
    for i, s in enumerate(siniflar):
        uyeler = tohum_gomuler[[j for j, e in enumerate(tohum_etiketleri)
                                if e == s]]
        if not len(uyeler):
            continue
        v = uyeler.mean(0)
        n = np.linalg.norm(v)
        proto[i] = v / n if n else v
    return proto


def prototip_siniflandir(gomuler, prototipler, siniflar):
    """Her gömüyü en yakın prototipe atar (ikisi de normalize → kosinüs)."""
    skor = gomuler @ prototipler.T
    return [siniflar[i] for i in skor.argmax(1)], skor


def rapor_bas(ad, olcum, taban_ad, taban_dogruluk, siniflar):
    fark = olcum['accuracy'] - taban_dogruluk
    isaret = '✅' if fark > 0.02 else ('⚠️' if fark > -0.02 else '⛔')
    print(f'\n  ── {ad} ──')
    print(f'  doğruluk {olcum["accuracy"]:.4f}   '
          f'taban {taban_dogruluk:.4f} ({taban_ad})   '
          f'fark {fark:+.4f}  {isaret}')
    print('  %-22s %9s %9s %9s %8s'
          % ('sınıf', 'precision', 'recall', 'f1', 'destek'))
    for s in siniflar:
        m = olcum['siniflar'][s]
        print('  %-22s %9.4f %9.4f %9.4f %8d'
              % (s, m['precision'], m['recall'], m['f1'], m['destek']))
    print('  karışıklık (satır=gerçek, sütun=tahmin):')
    print('    %-22s %s' % ('', ''.join(f'{s[:14]:>16}' for s in siniflar)))
    for s, satir in zip(siniflar, olcum['karisiklik']):
        print('    %-22s %s' % (s, ''.join(f'{int(x):>16}' for x in satir)))


# ─────────────────────────────────────────────────────────────────────────

def _tensor_al(cikti):
    """transformers sürüm farkını soğurur.

    4.x: get_text_features / get_image_features doğrudan tensör döner.
    5.x: BaseModelOutputWithPooling döner; yansıtılmış öznitelik
         `pooler_output` alanındadır.

    DOĞRULANDI (clip-vit-base-patch32): pooler_output ile elle hesaplanan
    kosinüs × logit_scale, modelin kendi logits_per_image değerini birebir
    veriyor (30.43 / 17.32). Yani pooler_output ORTAK yansıtılmış uzaydadır.
    """
    if hasattr(cikti, 'pooler_output'):
        return cikti.pooler_output
    if hasattr(cikti, 'keys') and 'pooler_output' in cikti:
        return cikti['pooler_output']
    return cikti


def clip_sinifla(yollar, siniflar, promptlar, model_adi, yigin):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    islemci = CLIPProcessor.from_pretrained(model_adi)
    model = CLIPModel.from_pretrained(model_adi).eval()
    aygit = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(aygit)
    print(f'  CLIP: {model_adi}  aygıt: {aygit}')

    # Metin tarafı: sınıf başına birden çok kalıp ortalanır.
    metin_proto = []
    for s in siniflar:
        kaliplar = promptlar.get(s) or [s.replace('_', ' ')]
        g = islemci(text=kaliplar, return_tensors='pt', padding=True)
        with torch.no_grad():
            t = _tensor_al(model.get_text_features(
                **{k: v.to(aygit) for k, v in g.items()}))
        t = torch.nn.functional.normalize(t, dim=-1).mean(0)
        metin_proto.append(torch.nn.functional.normalize(t, dim=-1))
    metin_proto = torch.stack(metin_proto)

    tahmin, skorlar = [], []
    for i in range(0, len(yollar), yigin):
        obek = [Image.open(p).convert('RGB') for p in yollar[i:i + yigin]]
        g = islemci(images=obek, return_tensors='pt')
        with torch.no_grad():
            v = _tensor_al(model.get_image_features(
                **{k: x.to(aygit) for k, x in g.items()}))
        v = torch.nn.functional.normalize(v, dim=-1)
        s = (v @ metin_proto.T).cpu().numpy()
        skorlar.append(s)
        tahmin += [siniflar[j] for j in s.argmax(1)]
        for im in obek:
            im.close()
    return tahmin, np.concatenate(skorlar, 0)


def dinov2_gomu(yollar, model_adi, yigin):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    islemci = AutoImageProcessor.from_pretrained(model_adi)
    model = AutoModel.from_pretrained(model_adi).eval()
    aygit = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(aygit)
    print(f'  DINOv2: {model_adi}  aygıt: {aygit}')

    cikti = []
    for i in range(0, len(yollar), yigin):
        obek = [Image.open(p).convert('RGB') for p in yollar[i:i + yigin]]
        g = islemci(images=obek, return_tensors='pt')
        with torch.no_grad():
            h = model(**{k: v.to(aygit) for k, v in g.items()})
        # CLS belirteci = görüntü düzeyi gösterim
        v = torch.nn.functional.normalize(h.last_hidden_state[:, 0], dim=-1)
        cikti.append(v.cpu().numpy())
        for im in obek:
            im.close()
    return np.concatenate(cikti, 0)


def main():
    ap = argparse.ArgumentParser(
        description='Sıfır-atış (CLIP) ve az-atış (DINOv2) karşılaştırması')
    ap.add_argument('veri', help='Sınıf klasörleri içeren dizin')
    ap.add_argument('--bolum', default='test')
    ap.add_argument('--yontem', default='ikisi',
                    choices=('clip', 'dinov2', 'ikisi'))
    ap.add_argument('--k', type=int, default=8,
                    help='DINOv2 az-atış: sınıf başına tohum örnek sayısı')
    ap.add_argument('--tohum-bolum', default='train', dest='tohum_bolum')
    ap.add_argument('--prompt-dosyasi', default=None, dest='prompt_dosyasi',
                    help='JSON: {"sinif": ["prompt1", "prompt2"]}')
    ap.add_argument('--clip-model', default=CLIP_MODEL, dest='clip_model')
    ap.add_argument('--dinov2-model', default=DINOV2_MODEL,
                    dest='dinov2_model')
    ap.add_argument('--yigin', type=int, default=16)
    ap.add_argument('--tohum', type=int, default=0)
    ap.add_argument('--kuru', action='store_true')
    a = ap.parse_args()

    kok = Path(a.veri)
    if not kok.exists():
        print(f'❌ Yok: {kok}')
        return 1

    yollar, gercek = goruntuleri_topla(kok, a.bolum)
    if not yollar:
        print(f'❌ {kok}/{a.bolum} içinde sınıf klasörü/görüntü yok.')
        return 1
    siniflar = sorted(set(gercek))
    taban_ad, taban_dogruluk = taban_cizgisi(gercek)

    print('=' * 78)
    print(f'VERİ: {kok} / {a.bolum}   ({len(yollar)} görüntü)')
    print('=' * 78)
    for s in siniflar:
        print(f'  {s:<22} {gercek.count(s):>6}')
    print(f"\n  TABAN ÇİZGİSİ: hep '{taban_ad}' de → {taban_dogruluk:.4f}")
    print('  Bir yöntem bunu geçemiyorsa hiçbir şey öğrenmemiştir.')

    promptlar = dict(VARSAYILAN_PROMPT)
    if a.prompt_dosyasi:
        promptlar.update(json.loads(
            Path(a.prompt_dosyasi).read_text(encoding='utf-8')))
    if a.yontem in ('clip', 'ikisi'):
        eksik = [s for s in siniflar if s not in promptlar]
        if eksik:
            print(f'\n  ⚠️ Prompt tanımlanmamış sınıflar: {eksik}')
            print('     Sınıf adının kendisi kullanılacak — zayıf sonuç verir.')

    if a.kuru:
        print('\n(--kuru: model yüklenmedi)')
        return 0

    sonuclar = {}

    if a.yontem in ('clip', 'ikisi'):
        print('\n--- CLIP SIFIR-ATIŞ ---')
        tahmin, skor = clip_sinifla(yollar, siniflar, promptlar,
                                    a.clip_model, a.yigin)
        sonuclar['CLIP sıfır-atış'] = olcumler(gercek, tahmin, siniflar)
        # CLIP skorları sıkışıktır; ayrımın ne kadar dar olduğunu göster.
        fark = np.sort(skor, 1)[:, -1] - np.sort(skor, 1)[:, -2]
        print(f'  en iyi ile ikinci arası fark: medyan {np.median(fark):.4f}')
        if np.median(fark) < 0.02:
            print('  ⚠️ Skorlar birbirine çok yakın — argmax kararsız.')

    if a.yontem in ('dinov2', 'ikisi'):
        print(f'\n--- DINOv2 AZ-ATIŞ (sınıf başına k={a.k} tohum) ---')
        t_yollar, t_gercek = goruntuleri_topla(kok, a.tohum_bolum)
        rng = np.random.default_rng(a.tohum)
        secili = []
        for s in siniflar:
            aday = [i for i, e in enumerate(t_gercek) if e == s]
            if len(aday) < a.k:
                print(f'  ⚠️ {s}: yalnızca {len(aday)} örnek var')
            secili += list(rng.choice(aday, min(a.k, len(aday)),
                                      replace=False))
        tohum_yollar = [t_yollar[i] for i in secili]
        tohum_etiket = [t_gercek[i] for i in secili]
        print(f'  tohum: {len(tohum_yollar)} kare '
              f'({a.tohum_bolum} bölümünden, test\'e DOKUNULMADI)')

        gomu_hepsi = dinov2_gomu(tohum_yollar + yollar, a.dinov2_model,
                                 a.yigin)
        n_t = len(tohum_yollar)
        prototipler = prototip_kur(gomu_hepsi[:n_t], tohum_etiket, siniflar)
        tahmin, skor = prototip_siniflandir(gomu_hepsi[n_t:], prototipler,
                                            siniflar)
        fark = np.sort(skor, 1)[:, -1] - np.sort(skor, 1)[:, -2]
        print(f'  en iyi ile ikinci arası fark: medyan {np.median(fark):.4f}')
        sonuclar[f'DINOv2 az-atış (k={a.k})'] = olcumler(gercek, tahmin,
                                                         siniflar)

    print('\n' + '=' * 78)
    print('SONUÇ')
    print('=' * 78)
    for ad, olcum in sonuclar.items():
        rapor_bas(ad, olcum, taban_ad, taban_dogruluk, siniflar)

    if len(sonuclar) > 1:
        en_iyi = max(sonuclar, key=lambda k: sonuclar[k]['accuracy'])
        print(f'\n  → En iyi: {en_iyi} '
              f'({sonuclar[en_iyi]["accuracy"]:.4f})')
        if sonuclar[en_iyi]['accuracy'] - taban_dogruluk < 0.05:
            print('  ⛔ Hiçbir yöntem taban çizgisini anlamlı geçemedi.')
            print('     Etiketli veriyle EĞİTİM gerekir: scripts/dinov2_egit.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
