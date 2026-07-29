"""Notebook'taki koşu keşif mantığı (devam etme) testleri.

Bu fonksiyonlar notebook hücresinde yaşıyor; buradan çıkarılıp sınanır.
Yanlış çalışırlarsa iki pahalı hata olur:
  - yanlış koşudan devam edilir (sıfırdan eğitim ince ayarın üstüne yazar),
  - biten bir koşu "yarım" sanılıp gereksiz yere sürdürülür.
"""

import json
from pathlib import Path

import pytest
import yaml

KOK = Path(__file__).resolve().parent.parent
NOTEBOOK = KOK / 'StrawberryVision_Colab_Production.ipynb'


def _yardimcilari_yukle(proje_dizini, egitilecek=None, ad='strawberry_exp'):
    """Notebook'un eğitim hücresinden koşu yardımcılarını çıkarır.

    egitilecek=None → değişken hiç tanımlı değil (4️⃣ hücresi çalıştırılmamış);
    hücre bu durumda da kırılmamalı, birleşik kuruluma düşmeli.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    egitim = next(''.join(c['source']) for c in nb['cells']
                  if c['cell_type'] == 'code' and 'def _kosular' in ''.join(c['source']))
    bas = egitim.index('INCE_EK =')
    son = egitim.index('# Mevcut koşuları göster')
    ortam = {'TRAIN_CONFIG': {'project': str(proje_dizini),
                              'name': ad, 'epochs': 200},
             'Path': Path}
    if egitilecek is not None:
        ortam['EGITILECEK'] = egitilecek
    exec(egitim[bas:son], ortam)
    return ortam


def _kosu_olustur(kok, ad, hedef_epoch, tamamlanan, agirlik=True):
    d = kok / ad
    (d / 'weights').mkdir(parents=True)
    if agirlik:
        (d / 'weights' / 'last.pt').write_bytes(b'x')
    (d / 'args.yaml').write_text(yaml.dump({'epochs': hedef_epoch}), encoding='utf-8')
    satirlar = ['epoch'] + [str(i) for i in range(1, tamamlanan + 1)]
    (d / 'results.csv').write_text(chr(10).join(satirlar), encoding='utf-8')
    return d


@pytest.fixture
def ortam(tmp_path):
    _kosu_olustur(tmp_path, 'strawberry_exp', 200, 200)            # sıfırdan, bitti
    _kosu_olustur(tmp_path, 'strawberry_exp-4', 200, 120)          # sıfırdan, yarım
    _kosu_olustur(tmp_path, 'strawberry_exp_ince_ayar', 60, 60)    # ince ayar, bitti
    _kosu_olustur(tmp_path, 'strawberry_exp_ince_ayar-2', 60, 18)  # ince ayar, yarım
    return _yardimcilari_yukle(tmp_path), tmp_path


def test_ince_ayar_kosusu_ayirt_edilir(ortam):
    g, kok = ortam
    assert g['_ince_mi'](kok / 'strawberry_exp_ince_ayar') is True
    assert g['_ince_mi'](kok / 'strawberry_exp-4') is False


def test_hedef_epoch_kosunun_kendisinden_okunur(ortam):
    """60 epochluk ince ayar, 200 hedefiyle kıyaslanmamalı."""
    g, kok = ortam
    assert g['_hedef_epoch'](kok / 'strawberry_exp_ince_ayar') == 60
    assert g['_hedef_epoch'](kok / 'strawberry_exp') == 200


def test_biten_ince_ayar_yarim_sayilmaz(ortam):
    """En sinsi hata: 60/60 biten koşu, 200 hedefiyle 'yarım' görünürdü."""
    g, kok = ortam
    assert g['_durum'](kok / 'strawberry_exp_ince_ayar')[0] == 'bitti'


def test_sifirdan_devam_ince_ayari_secmez(ortam):
    g, _ = ortam
    assert g['_devam_edilebilir'](ince=False)[0].name == 'strawberry_exp-4'


def test_ince_ayar_devam_kendi_kosusunu_secer(ortam):
    g, _ = ortam
    assert g['_devam_edilebilir'](ince=True)[0].name == 'strawberry_exp_ince_ayar-2'


def test_en_ilerlemis_kosu_secilir(tmp_path):
    """'En yeni' değil 'en ilerlemiş': yanlışlıkla açılan 1 epochluk koşu
    120 epoch ilerlemiş koşunun önüne geçmemeli."""
    _kosu_olustur(tmp_path, 'strawberry_exp-1', 200, 120)
    _kosu_olustur(tmp_path, 'strawberry_exp-9', 200, 1)     # daha yeni ama boş
    g = _yardimcilari_yukle(tmp_path)
    assert g['_devam_edilebilir'](ince=False)[0].name == 'strawberry_exp-1'


def test_checkpointsiz_kosu_devam_adayi_degil(tmp_path):
    _kosu_olustur(tmp_path, 'strawberry_exp-3', 200, 50, agirlik=False)
    g = _yardimcilari_yukle(tmp_path)
    assert g['_durum'](tmp_path / 'strawberry_exp-3')[0] == 'yok'
    assert g['_devam_edilebilir'](ince=False)[0] is None


def test_hic_kosu_yoksa_none(tmp_path):
    g = _yardimcilari_yukle(tmp_path)
    assert g['_devam_edilebilir']()[0] is None


def test_farkli_adli_ince_ayar_kosusu_bulunur(tmp_path):
    """GERÇEK HATA: ince ayar koşusunun adı sıfırdan eğitiminkinden bağımsızdır.

    train_config: name=strawberry_exp,  finetune_config: name=strawberry_ince_ayar
    Arama yalnızca 'strawberry_exp*' desenine dayansaydı ince ayar koşusu hiç
    görünmez ve yarım kalan eğitime devam edilemezdi.
    """
    _kosu_olustur(tmp_path, 'strawberry_ince_ayar', 60, 52)   # ön eki farklı
    _kosu_olustur(tmp_path, 'strawberry_exp-4', 200, 120)
    g = _yardimcilari_yukle(tmp_path)

    adlar = [d.name for d in g['_kosular'](ince=True)]
    assert 'strawberry_ince_ayar' in adlar, 'farklı adlı ince ayar koşusu bulunamadı'
    assert g['_devam_edilebilir'](ince=True)[0].name == 'strawberry_ince_ayar'
    # Sıfırdan tarafı etkilenmemeli
    assert g['_devam_edilebilir'](ince=False)[0].name == 'strawberry_exp-4'


def test_ilgisiz_klasor_kosu_sayilmaz(tmp_path):
    (tmp_path / 'baska_bir_sey').mkdir()
    _kosu_olustur(tmp_path, 'strawberry_exp-1', 200, 10)
    g = _yardimcilari_yukle(tmp_path)
    assert [d.name for d in g['_kosular']()] == ['strawberry_exp-1']


# ──────────────── koşular MODELE göre ayrılmalı (hiyerarşik mimari)
#
# GERÇEK HATA: EGITILECEK='organ_detection' seçilip eğitim başlatıldığında
# birleşik modelin `strawberry_ince_ayar` koşusu "bitmiş" sayıldı. Sonuç:
#   1. organ eğitimi hiç başlamadı ("zaten tamamlanmış" denildi),
#   2. o koşunun 10 sınıflı ağırlığı `best_models/cilek/organ.pt` adıyla
#      kopyalandı — boru hattı yanlış modelle çalışacaktı.
# Süzgeç ince ayar koşularını AD EKİNDEN buluyordu ama model adına hiç
# bakmıyordu; bütün modeller aynı results/ klasörünü paylaştığı için karıştı.

DRIVE_KOSULARI = ('strawberry_exp', 'strawberry_exp-2', 'strawberry_exp-3',
                  'strawberry_ince_ayar', 'strawberry_ince_ayar-2')


def _drive_benzeri(kok):
    """Kullanıcının Drive'ındaki gerçek koşu klasörlerini kurar."""
    for ad in DRIVE_KOSULARI:
        hedef = 60 if 'ince_ayar' in ad else 200
        _kosu_olustur(kok, ad, hedef, hedef)          # hepsi tamamlanmış
    return kok


def test_uzman_model_baska_modelin_kosusunu_gormez(tmp_path):
    """Asıl hata: organ eğitimi, birleşik modelin koşusunu kendi sanıyordu."""
    _drive_benzeri(tmp_path)
    g = _yardimcilari_yukle(tmp_path, 'organ_detection', ad='organ_detection')
    assert [d.name for d in g['_kosular']()] == [], (
        'organ modeli için koşu YOK; birleşik koşular sayılırsa eğitim atlanır')


def test_uzman_model_kendi_kosularini_gorur(tmp_path):
    _drive_benzeri(tmp_path)
    _kosu_olustur(tmp_path, 'organ_detection', 200, 120)            # yarım
    _kosu_olustur(tmp_path, 'organ_detection_ince_ayar', 60, 20)    # yarım
    g = _yardimcilari_yukle(tmp_path, 'organ_detection', ad='organ_detection')

    assert set(d.name for d in g['_kosular']()) == {
        'organ_detection', 'organ_detection_ince_ayar'}
    assert g['_devam_edilebilir'](ince=False)[0].name == 'organ_detection'
    assert g['_devam_edilebilir'](ince=True)[0].name == 'organ_detection_ince_ayar'


def test_modeller_birbirinin_kosusunu_secmez(tmp_path):
    for ad in ('organ_detection', 'leaf_disease', 'fruit_disease'):
        _kosu_olustur(tmp_path, ad, 200, 50)
    for ad in ('organ_detection', 'leaf_disease', 'fruit_disease'):
        g = _yardimcilari_yukle(tmp_path, ad, ad=ad)
        assert [d.name for d in g['_kosular']()] == [ad]


def test_birlesik_kurulum_bozulmadi(tmp_path):
    """Eski kurulumda iki koşu ailesi vardır; ortak önek 'strawberry'.

    'strawberry_ince_ayar' adı 'strawberry_exp' ile BAŞLAMAZ — önek yanlış
    seçilseydi ince ayara devam edilemezdi.
    """
    _drive_benzeri(tmp_path)
    g = _yardimcilari_yukle(tmp_path, 'birlesik')
    assert len(g['_kosular']()) == len(DRIVE_KOSULARI)
    assert set(d.name for d in g['_kosular'](ince=True)) == {
        'strawberry_ince_ayar', 'strawberry_ince_ayar-2'}


def test_egitilecek_tanimsizsa_birlesike_duser(tmp_path):
    """4️⃣ hücresi çalıştırılmadıysa hücre NameError ile ölmemeli."""
    _drive_benzeri(tmp_path)
    g = _yardimcilari_yukle(tmp_path, egitilecek=None)
    assert g['KOSU_ONEKI'] == 'strawberry'
    assert len(g['_kosular']()) == len(DRIVE_KOSULARI)


def test_boru_hatti_kopyasi_sinif_dogrulamasi_yapar():
    """Eğitim atlandığında RUN_DIR başka bir koşuyu gösterebilir.

    Doğrulama olmadan o koşunun ağırlığı boru hattı adıyla kopyalanır ve
    sistem sessizce yanlış modelle çalışır (bu bir kez gerçekten oldu:
    10 sınıflı birleşik model `organ.pt` olarak kopyalandı).
    """
    src = _hucre('BORU_HATTI_ADLARI')
    assert '_agirlik_sin' in src, 'ağırlığın sınıfları okunmalı'
    assert 'BORU HATTI ADIYLA KOPYALANMADI' in src, 'uyuşmazlıkta kopyalama durmalı'
    assert '_dosya = None' in src, 'uyuşmazlıkta kopyalama iptal edilmeli'


def test_ince_ayar_kosu_adi_modele_gore_ayrilir():
    """finetune_config'teki sabit ad kullanılsaydı organ ince ayarı
    birleşik modelin klasörüne yazar, ikisi birbirini ezerdi."""
    src = _hucre('_ince')
    assert "_ince['name'] = f'{EGITILECEK}{INCE_EK}'" in src


# ────────────────────────────── eğitim çıktıları Drive'a yazılmalı
def _hucre(anahtar):
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] == 'code' and anahtar in ''.join(c['source']):
            return ''.join(c['source'])
    raise AssertionError(f'hücre bulunamadı: {anahtar}')


def test_kosu_dizini_drive_altinda():
    """Colab oturumu kapanınca yerel disk silinir. Eğitim çıktıları (checkpoint,
    results.csv, ağırlıklar) Drive'a yazılmazsa eğitim tamamen kaybolur ve
    devam da edilemez."""
    src = _hucre("TRAIN_CONFIG['project']")
    assert "TRAIN_CONFIG['project'] = str(RESULTS_DIR)" in src,         "project, Drive icindeki RESULTS_DIR ile degistirilmeli"


def test_results_dir_drive_kokunde_tanimli():
    src = _hucre('DRIVE_ROOT')
    assert "RESULTS_DIR = DRIVE_ROOT / 'results'" in src
    assert "MODELS_DIR = DRIVE_ROOT / 'best_models'" in src


def test_ince_ayar_project_devralir():
    """GERÇEK HATA: ince ayar yapılandırmayı taze yüklerken 'project'i
    devralmıyordu; sonuçlar Colab'in geçici diskine yazılıyordu."""
    src = _hucre('_ince')
    assert "'device', 'project'" in src,         "ince ayar TRAIN_CONFIG'ten 'project' anahtarını devralmalı"


def test_egitilen_model_boru_hatti_adiyla_kopyalanir():
    """Eğitim çıktısı hep 'best.pt'; boru hattı ise kütükteki adı arar.
    Elle adlandırma sessiz hataya yol açtığı için otomatik kopyalanır."""
    src = _hucre('BORU_HATTI_ADLARI')
    duz = src.replace('"', "'")          # tırnak stiline bağlı kalmasın
    assert "'organ_detection': 'organ.pt'" in duz
    assert "'fruit_ripeness': 'fruit_ripeness.pt'" in duz
    assert 'MODELS_DIR / _urun' in src, 'ürün klasörü altına kopyalanmalı'


def test_checkpointler_belirli_araliklarla_kaydedilir():
    """Oturum koparsa kaldığı yerden devam edilebilsin."""
    import yaml
    for ad in ('train_config.yaml', 'finetune_config.yaml'):
        v = yaml.safe_load((KOK / 'configs' / ad).read_text(encoding='utf-8'))
        assert v.get('save_period', 0) > 0, f'{ad}: save_period tanımlı olmalı'


# ──────────────────────── notebook'taki dosya yolları gerçekten var mı
def test_notebook_kod_hucrelerinde_eski_yol_kalmadi():
    """GERÇEK HATA: configs/strawberry_data.yaml -> configs/urunler/cilek/veri.yaml
    taşınınca notebook'un depo doğrulaması Colab'de AssertionError verdi.

    Yorum satırlarında geçmesi sorun değil; ÇALIŞAN kodda kalmamalı.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    sorunlu = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        kod_satirlari = [l.split('#')[0] for l in src.splitlines()]
        if not any('strawberry_data.yaml' in l for l in kod_satirlari):
            continue
        # Eski yola atıf YALNIZCA yeni yolun da denendiği hücrelerde kabul
        # edilir (geriye dönük destek). Tek başına duruyorsa bayat demektir.
        if 'urunler' not in src:
            sorunlu.append(f'hücre {i}: yalnızca eski yol kullanılıyor')
    assert not sorunlu, 'bayat yapılandırma yolu: ' + '; '.join(sorunlu)


def test_depo_dogrulamasi_urun_yolunu_kabul_eder():
    src = _hucre('Depo hazır')
    assert 'urunler' in src, 'ürün kapsamlı yol denenmeli'
    assert 'strawberry_data.yaml' in src, 'eski yola geriye dönük destek kalmalı'


def test_depo_dogrulamasi_gercek_depoda_gecer():
    """Doğrulama bloğunu gerçek depo kökünde çalıştır — Colab'de patlamasın."""
    src = _hucre('Depo hazır')
    blok = src[src.index('_urun = globals()'):]
    ortam = {'Path': Path}
    exec(blok, ortam)                      # AssertionError atmamalı
    assert ortam['_bulunan'].exists()


# ─────────────────────────── bayat sekme tespiti
def test_notebook_surumu_tanimli():
    """Colab, GitHub'dan açılan notebook'u TARAYICIDA önbelleğe alır: depo
    git pull ile güncellenir ama sekmedeki hücre kodu eski kalır. Bu fark
    defalarca 'dosya yok' hatasına yol açtı."""
    src = _hucre('NOTEBOOK_SURUM')
    assert 'NOTEBOOK_SURUM = "' in src


def test_bayat_sekme_uyarisi_calisir(tmp_path, capsys, monkeypatch):
    """Sürüm farklıysa uyarı basılmalı, aynıysa sessiz kalmalı."""
    import shutil
    shutil.copy(NOTEBOOK, tmp_path / NOTEBOOK.name)
    monkeypatch.chdir(tmp_path)

    src = _hucre('NOTEBOOK_SURUM')
    blok = src[src.index('# ── Sekme bayat'):]

    exec(blok, {'Path': Path})
    assert 'ESKİ' not in capsys.readouterr().out, 'aynı sürümde uyarı çıkmamalı'

    eski = blok.replace(blok.split('NOTEBOOK_SURUM = "')[1].split('"')[0],
                        '2000-01-01-0', 1)
    exec(eski, {'Path': Path})
    cikti = capsys.readouterr().out
    assert 'ESKİ' in cikti and 'Ctrl+Shift+R' in cikti


def test_surum_kontrolu_egitimi_engellemez(tmp_path, monkeypatch):
    """Notebook dosyası bulunamasa bile sürüm kontrolü hata fırlatmamalı."""
    monkeypatch.chdir(tmp_path)          # burada .ipynb yok
    src = _hucre('NOTEBOOK_SURUM')
    exec(src[src.index('# ── Sekme bayat'):], {'Path': Path})
