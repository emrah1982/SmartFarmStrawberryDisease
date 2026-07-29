"""Eğitilen modeli boru hattına kurar (doğrulayarak).

NEDEN BETİK?
    Eğitim `runs/.../weights/best.pt` üretir; boru hattı ise modeli kütükteki
    adla arar (`models/<urun>/organ.pt` gibi). Elle kopyalarken üç sessiz hata
    olur:

      1. Yanlış ada kopyalama → model hiç kullanılmaz, kimse fark etmez
      2. Yanlış modeli kopyalama → örn. yaprak modeli olgunluk yerine geçer
      3. Sınıfları uymayan model → boru hattı çalışır ama sonuçlar saçmadır

    Bu betik kopyalamadan ÖNCE modelin sınıflarını kütükle karşılaştırır.

KULLANIM
    python scripts/model_kur.py --listele
    python scripts/model_kur.py organ runs/train/organ/weights/best.pt
    python scripts/model_kur.py olgunluk .../best.pt --urun cilek
"""

import argparse
import shutil
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))


def agirlik_siniflari(yol: Path):
    """Kontrol noktasındaki sınıf adları. Okunamazsa None."""
    try:
        import torch
        ckpt = torch.load(str(yol), map_location='cpu', weights_only=False)
        model = ckpt.get('model') or ckpt.get('ema')
        adlar = getattr(model, 'names', None) or ckpt.get('names')
        if not adlar:
            return None
        return [adlar[i] for i in sorted(adlar)] if isinstance(adlar, dict) else list(adlar)
    except Exception as e:
        print(f'⚠️ Sınıflar okunamadı: {e}')
        return None


def listele(urun=None):
    from app import modeller
    print(f"{'model':16} {'rol':16} {'durum':8} {'beklenen sınıflar'}")
    print('-' * 92)
    for d in modeller.durum(urun):
        durum = 'HAZIR' if d['var'] else 'eksik'
        print(f"{d['ad']:16} {d['rol']:16} {durum:8} {', '.join(d['siniflar']) or '—'}")
        print(f"{'':16} {'':16} {'':8} → {d['yol']}")


def kur(ad: str, kaynak: Path, urun=None, zorla=False) -> int:
    from app import modeller, urunler

    tanim = modeller.tanim(ad, urun)
    if tanim is None:
        print(f'❌ Kütükte böyle bir model yok: {ad}')
        print(f"   Tanımlılar: {', '.join(modeller.tanimlar(urun))}")
        return 1
    if not kaynak.exists():
        print(f'❌ Kaynak dosya yok: {kaynak}')
        return 1

    # --- Sınıf doğrulaması ------------------------------------------------
    beklenen = tanim.siniflar
    gercek = agirlik_siniflari(kaynak)
    if beklenen and gercek is not None:
        if [s.lower() for s in gercek] != [s.lower() for s in beklenen]:
            print('')
            print('=' * 74)
            print('⛔ KURULUM YAPILMADI — modelin sınıfları kütükle uyuşmuyor')
            print('=' * 74)
            print(f'  Kütükte beklenen : {beklenen}')
            print(f'  Modelde bulunan  : {gercek}')
            print('')
            print('  Olası sebepler:')
            print('   • Yanlış koşunun best.pt dosyası verildi')
            print(f"   • {ad} yerine başka bir modeli kuruyorsunuz")
            print('   • Dataset sınıf sırası değişti (etiketler bozulmuş olabilir)')
            print('')
            print('  Kütükteki listeyi düzeltmek istiyorsanız:')
            print(f'    configs/urunler/{urunler.slug(urun) if urun else urunler.VARSAYILAN}'
                  '/modeller.yaml')
            print('  Riski bilerek devam: --zorla')
            print('=' * 74)
            if not zorla:
                return 1
        else:
            print(f'✅ Sınıf uyumu tamam: {len(gercek)} sınıf')

    hedef = tanim.yol
    hedef.parent.mkdir(parents=True, exist_ok=True)
    if hedef.exists():
        yedek = hedef.with_suffix('.pt.onceki')
        shutil.copy2(hedef, yedek)
        print(f'ℹ️ Önceki model yedeklendi: {yedek.name}')

    shutil.copy2(kaynak, hedef)
    print(f'📦 Kuruldu: {kaynak}  →  {hedef}')

    # --- Kurulum sonrası durum -------------------------------------------
    modeller.bosalt()            # bellekteki eski modeli bırak
    modeller.bosalt_kutuk()
    hazir = modeller.hiyerarsik_hazir(urun)
    eksik = modeller.eksikler(urun)
    print('')
    print(f'Hiyerarşik boru hattı: {"AKTİF" if hazir else "pasif (organ modeli yok)"}')
    if eksik:
        print(f'Eksik modeller       : {", ".join(eksik)}')
        print('   (eksik olanlar için boru hattı miras modele düşer)')
    else:
        print('Tüm uzman modeller hazır — miras model artık kullanılmıyor.')
    print('')
    print('⚠️ Uygulamayı yeniden başlatın: docker compose restart')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Eğitilen modeli boru hattına kurar')
    ap.add_argument('ad', nargs='?', help='Kütükteki model adı (organ, olgunluk, ...)')
    ap.add_argument('kaynak', nargs='?', help='Eğitim çıktısı best.pt')
    ap.add_argument('--urun', default=None)
    ap.add_argument('--listele', action='store_true')
    ap.add_argument('--zorla', action='store_true',
                    help='Sınıf uyumsuzluğuna rağmen kur (ÖNERİLMEZ)')
    a = ap.parse_args()

    if a.listele or not a.ad:
        listele(a.urun)
        return 0
    if not a.kaynak:
        print('Kaynak dosya belirtin: python scripts/model_kur.py <ad> <best.pt>')
        return 1
    return kur(a.ad, Path(a.kaynak), a.urun, a.zorla)


if __name__ == '__main__':
    sys.exit(main())
