"""Önceki eğitimlerin eğrisine bakarak epoch sayısı önerir.

NEDEN TAHMİN DEĞİL ÖLÇÜM?
    "200 epoch yapalım, ezberlerse durur" yaklaşımının iki sorunu var:

    1. Öğrenme oranı takvimi TOPLAM epoch'a göre hesaplanır
       (Ultralytics: cos_lr → one_cycle(1, lrf, epochs)). 200 planlayıp 70'te
       durursanız model lr0'ın hâlâ ~%73'ünde kalır: ağırlıklar "sıcak",
       oturmamış olur. 70 planlayıp 70'te bitirseydiniz %1'e inmiş olurdu.
    2. close_mosaic son N epoch'ta devreye girer. 200 planlanıp 70'te
       durulursa mozaik hiç kapanmaz ve o son iyileşme alınmaz.

    Yani epoch sayısı "üst sınır" değil, eğitimin ŞEKLİNİ belirleyen bir
    parametredir. Bu yüzden gelişigüzel büyük verilmez.

    Doğru yol, önceki koşuların results.csv eğrisini okuyup nerede
    doygunlaştığını ÖLÇMEKTİR. Bu betik onu yapar.

KULLANIM
    python scripts/epoch_oner.py --results "G:/.../results"
    python scripts/epoch_oner.py --results results --ince-ayar
"""

import argparse
import csv
import sys
from pathlib import Path

# Ultralytics'te fitness = mAP50-95 (utils/metrics.py: w = [0, 0, 0, 1.0])
OLCUT_ADAYLARI = ('metrics/mAP50-95(B)', 'metrics/mAP50-95', 'mAP50-95(B)', 'mAP50-95')

# Son dilimdeki bu kazancın altı "plato" sayılır. Ölçüm gürültüsü (val setinin
# rastgeleliği, AMP) tipik olarak bu mertebededir; altındaki fark gerçek
# öğrenme değildir.
ANLAMLI_KAZANC = 0.002


def egri_oku(csv_yolu: Path):
    """[(epoch, mAP50-95)] listesi döner."""
    with open(csv_yolu, encoding='utf-8') as f:
        satirlar = list(csv.DictReader(f))
    if not satirlar:
        return []
    basliklar = {k.strip(): k for k in satirlar[0]}
    olcut = next((basliklar[a] for a in OLCUT_ADAYLARI if a in basliklar), None)
    if olcut is None:
        olcut = next((v for k, v in basliklar.items() if 'mAP50-95' in k), None)
    if olcut is None:
        return []
    epoch_k = basliklar.get('epoch', 'epoch')
    egri = []
    for s in satirlar:
        try:
            egri.append((int(float(s[epoch_k])), float(s[olcut])))
        except (ValueError, KeyError, TypeError):
            continue
    return egri


def coz(egri):
    """Eğriden karar için gereken sayıları çıkarır."""
    en_iyi_epoch, en_iyi = max(egri, key=lambda x: x[1])
    son_epoch = egri[-1][0]

    # En iyinin %X'ine ilk ne zaman ulaşıldı
    esikler = {}
    for oran in (0.90, 0.95, 0.98, 0.99):
        hedef = en_iyi * oran
        esikler[oran] = next((e for e, v in egri if v >= hedef), None)

    # En uzun "iyileşme yok" serisi → patience bu değerin üstünde olmalı,
    # yoksa eğitim gerçek bir platoda değil, geçici duraklamada kesilir.
    en_uzun_duraklama = suanki = 0
    tepe = -1.0
    for _, v in egri:
        if v > tepe:
            tepe, suanki = v, 0
        else:
            suanki += 1
            en_uzun_duraklama = max(en_uzun_duraklama, suanki)

    # Son %10'luk dilimde kazanç
    kesme = max(1, len(egri) // 10)
    son_kazanc = egri[-1][1] - egri[-kesme - 1][1] if len(egri) > kesme else 0.0

    return {
        'toplam': son_epoch, 'en_iyi': en_iyi, 'en_iyi_epoch': en_iyi_epoch,
        'esikler': esikler, 'en_uzun_duraklama': en_uzun_duraklama,
        'son_dilim': kesme, 'son_kazanc': son_kazanc,
        # "Sonda hâlâ iyileşiyor" demek için en iyinin sona yakın olması YETMEZ:
        # plato sırasında da argmax teknik olarak sonlarda çıkabilir (0.0001'lik
        # oynamalar). Son dilimde ANLAMLI bir kazanç da aranır.
        'sonda_hala_iyilesiyor': (en_iyi_epoch >= son_epoch - max(2, son_epoch // 20)
                                  and son_kazanc > ANLAMLI_KAZANC),
        'platoda': son_kazanc <= ANLAMLI_KAZANC,
    }


def rapor(ad, d):
    print(f'\n■ {ad}')
    print(f'   {d["toplam"]} epoch · en iyi mAP50-95 = {d["en_iyi"]:.4f} '
          f'(epoch {d["en_iyi_epoch"]})')
    for oran, e in d['esikler'].items():
        print(f'   en iyinin %{oran * 100:.0f}\'ine ulaşma: epoch {e}')
    print(f'   en uzun iyileşmesiz seri: {d["en_uzun_duraklama"]} epoch')
    print(f'   son {d["son_dilim"]} epoch kazancı: {d["son_kazanc"]:+.4f}')
    if d['sonda_hala_iyilesiyor']:
        print('   ⚠️ Sonda HÂLÂ İYİLEŞİYORDU → daha uzun eğitim kazanç verebilir.')
    elif d['platoda']:
        print('   ✅ Sonda PLATODA → bu uzunluk yeterliydi; uzatmanın kazancı yok.')


def oner(analizler, ince_ayar: bool):
    print('\n' + '=' * 72)
    print('ÖNERİ')
    print('=' * 72)
    if not analizler:
        print('Geçmiş koşu bulunamadı. İlk eğitim için: epochs 150-200, patience 50.')
        return {'epochs': 200, 'patience': 50}

    en_uzun = max(d['en_uzun_duraklama'] for _, d in analizler)
    # patience, gözlenen en uzun geçici duraklamanın üstünde olmalı
    patience = max(20, int(en_uzun * 1.5))

    # En uzun (en bilgilendirici) koşuya bak: kısa koşu zaten sonda iyileşiyor görünür
    en_uzun_kosu = max(analizler, key=lambda x: x[1]['toplam'])[1]
    hala = en_uzun_kosu['sonda_hala_iyilesiyor']
    # %98'e ulaşma epoch'u: eğrinin "işi biten" noktası
    doyma = [d['esikler'][0.98] for _, d in analizler if d['esikler'][0.98]]
    ortalama_doyma = sum(doyma) / len(doyma) if doyma else 100

    if ince_ayar:
        # Warm start zaten tepeye yakın başlar; sıfırdan doyma süresinin
        # yarısı pratikte yeterlidir, ama takvimin tamamlanması için
        # gereğinden kısa da tutulmaz.
        epochs = max(50, int(ortalama_doyma * 0.6))
        gerekce = (f'ince ayar: sıfırdan eğitimde %98 doyma ~{ortalama_doyma:.0f}. '
                   f'epoch; warm start tepeye yakın başladığı için ~%60\'ı')
    else:
        epochs = int(ortalama_doyma * 1.6) if not hala else max(
            200, int(max(d['toplam'] for _, d in analizler) * 1.25))
        gerekce = ('sıfırdan: sondaki iyileşme sürüyordu, önceki koşudan uzun'
                   if hala else f'sıfırdan: %98 doyma ~{ortalama_doyma:.0f}. epoch')

    print(f'  epochs   : {epochs}')
    print(f'  patience : {patience}')
    print(f'\n  Gerekçe: {gerekce}.')
    print(f'  patience, gözlenen en uzun geçici duraklamanın ({en_uzun} epoch)')
    print('  1.5 katı — gerçek plato ile geçici duraklama karışmasın.')
    print('\n  ⚠️ epochs\'u "üst sınır" diye şişirmeyin: öğrenme oranı takvimi ve')
    print('     close_mosaic bu sayıya göre hesaplanır. Erken durdurma tetiklenirse')
    print('     model takvimin ortasında, yüksek öğrenme oranında kalır.')
    if hala:
        print('\n  ℹ️ En uzun koşu sonda hâlâ iyileşiyordu: bu veri setinde asıl risk')
        print('     EZBERLEME değil, YETERSİZ EĞİTİM. Erken durdurma muhtemelen hiç')
        print('     tetiklenmeyecek; epoch sayısını bütçenize göre seçin.')
    else:
        print(f'\n  ℹ️ En uzun koşu ({en_uzun_kosu["toplam"]} epoch) sonda PLATODAYDI:')
        print(f'     son {en_uzun_kosu["son_dilim"]} epoch kazancı yalnızca '
              f'{en_uzun_kosu["son_kazanc"]:+.4f} (ölçüm gürültüsü mertebesinde).')
        print('     Yani EZBERLEME değil DOYMA var — daha uzun eğitmek boşa GPU.')
        print(f'     Eğrinin işi epoch ~{en_uzun_kosu["esikler"][0.98]} civarında bitmiş;')
        print(f'     kalan {en_uzun_kosu["toplam"] - (en_uzun_kosu["esikler"][0.98] or 0)}'
              ' epoch yalnızca %2 kazandırmış.')
    print('=' * 72)
    return {'epochs': epochs, 'patience': patience}


def main():
    ap = argparse.ArgumentParser(description='Geçmiş eğitim eğrisinden epoch önerir')
    ap.add_argument('--results', default='results',
                    help='Koşu dizinlerini içeren klasör (her birinde results.csv)')
    ap.add_argument('--ince-ayar', action='store_true',
                    help='Öneri warm start (fine-tuning) için hesaplansın')
    a = ap.parse_args()

    kok = Path(a.results)
    if not kok.exists():
        print(f'Klasör yok: {kok}')
        return 1

    analizler = []
    for csv_yolu in sorted(kok.rglob('results.csv')):
        egri = egri_oku(csv_yolu)
        if len(egri) < 5:
            continue
        d = coz(egri)
        analizler.append((csv_yolu.parent.name, d))
        rapor(csv_yolu.parent.name, d)

    oner(analizler, a.ince_ayar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
