"""Sınıf kütüğüne yeni zararlı/hastalık ekler ve eğitim yapılandırmasıyla eşitler.

NEDEN BETİK?
    Yeni sınıf eklemek iki dosyayı birden ilgilendirir:
      configs/siniflar.yaml        → görünen ad, grup, eşik
      configs/strawberry_data.yaml → eğitimde kullanılan `names` ve `nc`
    İkisi elle güncellenirse kaçınılmaz olarak birbirinden sapar ve etiketler
    yanlış sınıfa kayar. Betik tek kaynaktan ikisini de yazar.

ID KURALI
    Yeni sınıfa bir sonraki boş ID verilir. VERİLEN ID BİR DAHA DEĞİŞTİRİLMEZ:
    etiket dosyalarında sayı olarak saklandığı için, ID kayarsa geçmişte
    etiketlenen her şey yanlış sınıfa dönüşür.

KULLANIM
    python scripts/sinif_ekle.py "Spider Mites"
    python scripts/sinif_ekle.py "Spider Mites" --tr "Kırmızı Örümcek" --grup zararli
    python scripts/sinif_ekle.py --listele

SONRASI (önemli)
    Kütüğe eklemek modele ÖĞRETMEZ. Sıra:
      1. Bu betik → sınıf etiketleme ekranında görünür
      2. Saha görüntüleri toplanır ve bu sınıfla etiketlenir (en az 100-200 örnek)
      3. Dışa aktarım → merge_datasets.py → yeniden eğitim
      4. Yeni best.pt modeli sınıfı tanır
"""

import argparse
import sys
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parent.parent
# --- Ürün kapsamı (çok bitkili kurulum) -------------------------------------
# Her ürünün KENDİ sınıf kütüğü vardır; ID'ler ürün içinde 0..n-1'dir.
# Ürünler arası ID çakışması bu sayede imkânsızdır.
def _urun_yolu(dosya: str, urun: str = None):
    import os
    urun = urun or os.environ.get('VARSAYILAN_URUN', 'cilek')
    yeni = KOK / 'configs' / 'urunler' / urun / dosya
    if yeni.exists():
        return yeni
    eski = {'siniflar.yaml': 'siniflar.yaml',
            'veri.yaml': 'strawberry_data.yaml'}[dosya]
    return KOK / 'configs' / eski


KUTUK = _urun_yolu('siniflar.yaml')
EGITIM = _urun_yolu('veri.yaml')
GRUPLAR = ('hastalik', 'zararli', 'olgunluk', 'diger')


def _oku(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding='utf-8')) or {} if p.exists() else {}


def id_haritasi(kutuk: dict, egitim: dict) -> dict:
    harita = {}
    isimler = egitim.get('names', {})
    if isinstance(isimler, list):
        harita = {i: a for i, a in enumerate(isimler)}
    else:
        harita = {int(k): v for k, v in isimler.items()}
    for ad, d in kutuk.items():
        if (d or {}).get('id') is not None:
            harita[int(d['id'])] = ad
    return dict(sorted(harita.items()))


def listele():
    kutuk, egitim = _oku(KUTUK), _oku(EGITIM)
    harita = id_haritasi(kutuk, egitim)
    print(f'{"ID":>3}  {"eğitimde":8}  {"grup":9}  {"ad":32}  Türkçe')
    print('-' * 88)
    for kimlik, ad in harita.items():
        d = kutuk.get(ad) or {}
        egitimde = 'evet' if d.get('egitimde', True) is not False else 'HAYIR'
        print(f'{kimlik:>3}  {egitimde:8}  {d.get("grup", "-"):9}  {ad:32}  {d.get("tr", "-")}')

    # Planlanan = kütükte var ama hiçbir ID'ye bağlanmamış (eğitim yaml'ında da yok)
    kayitli = set(harita.values())
    planlanan = [(a, d) for a, d in kutuk.items() if a not in kayitli]
    if planlanan:
        print(f'\nPlanlanan ({len(planlanan)} sınıf, henüz ID yok — etiketlemede çıkmaz):')
        for ad, d in planlanan:
            print(f'  {ad:32}  {(d or {}).get("tr", "")}')
        print('\nEklemek için:  python scripts/sinif_ekle.py "<ad>"')


def ekle(ad: str, tr: str, grup: str, esik):
    kutuk, egitim = _oku(KUTUK), _oku(EGITIM)
    harita = id_haritasi(kutuk, egitim)

    if ad in harita.values():
        kimlik = [k for k, v in harita.items() if v == ad][0]
        print(f'⚠️ "{ad}" zaten kayıtlı (ID {kimlik}). ID değiştirilmez.')
        return 1

    kimlik = (max(harita) + 1) if harita else 0
    mevcut = kutuk.get(ad) or {}
    kutuk[ad] = {
        'tr': tr or mevcut.get('tr') or ad,
        'en': mevcut.get('en') or ad,
        'grup': grup or mevcut.get('grup') or 'diger',
        'id': kimlik,
        'egitimde': False,          # model henüz tanımıyor
        **({'esik': esik} if esik is not None else {}),
    }
    harita[kimlik] = ad

    KUTUK.write_text(
        '# Sınıf kütüğü — scripts/sinif_ekle.py tarafından da güncellenir.\n'
        '# ID alanları BİR DAHA DEĞİŞTİRİLMEMELİDİR (etiket dosyaları sayıya bağlıdır).\n\n'
        + yaml.dump(kutuk, allow_unicode=True, sort_keys=False), encoding='utf-8')

    egitim['names'] = {int(k): v for k, v in sorted(harita.items())}
    egitim['nc'] = len(harita)
    EGITIM.write_text(yaml.dump(egitim, allow_unicode=True, sort_keys=False),
                      encoding='utf-8')

    print(f'✅ "{ad}" eklendi → ID {kimlik}, görünen ad: {kutuk[ad]["tr"]}')
    print(f'   configs/siniflar.yaml ve strawberry_data.yaml eşitlendi (nc={egitim["nc"]}).')
    print('\nSıradaki adımlar:')
    print('   1. Etiketleme ekranında sınıf artık seçilebilir.')
    print(f'   2. Bu sınıftan 100-200 örnek toplayıp etiketleyin (canlı 🗃️ modu işe yarar).')
    print('   3. Dışa aktar → merge_datasets.py → yeniden eğitim.')
    print('   4. Yeni modelde `egitimde: true` yapın.')
    print('\n⚠️ YENİDEN EĞİTİM ŞART: kütüğe eklemek modele öğretmez, yalnızca')
    print('   etiketlemeyi ve arayüzü hazırlar.')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Sınıf kütüğüne yeni sınıf ekler.')
    ap.add_argument('ad', nargs='?', help='Eğitimdeki İngilizce ad (örn. "Spider Mites")')
    ap.add_argument('--tr', default='', help='Ekranda görünecek Türkçe ad')
    ap.add_argument('--grup', default='', choices=('', *GRUPLAR))
    ap.add_argument('--esik', type=float, default=None, help='Sınıfa özel güven eşiği')
    ap.add_argument('--listele', action='store_true', help='Mevcut sınıfları göster')
    ap.add_argument('--urun', default=None, help='Ürün kapsamı (varsayılan: cilek)')
    a = ap.parse_args()

    if a.urun:
        global KUTUK, EGITIM
        KUTUK = _urun_yolu('siniflar.yaml', a.urun)
        EGITIM = _urun_yolu('veri.yaml', a.urun)

    if a.listele or not a.ad:
        listele()
        return 0
    return ekle(a.ad.strip(), a.tr.strip(), a.grup, a.esik)


if __name__ == '__main__':
    sys.exit(main())
