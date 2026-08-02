

# ─────────────────────────────────────────────────────────────────────────
# ORTAK KAPSAM — urune bagli OLMAYAN varliklar
#
# Hastalik urune baglidir ('Leaf Spot' cilekte Mycosphaerella, findikta
# Piggotia). BOCEK TURU DEGILDIR: danaburnu her bitkide ayni turdur.
# ─────────────────────────────────────────────────────────────────────────

def test_bocek_modeli_her_uruncte_gorunur():
    from app import modeller
    for urun in ('cilek', 'findik'):
        t = modeller.tanimlar(urun).get('bocek_teshis')
        assert t is not None, f'{urun} icin bocek_teshis yok'
        assert t.ortak is True


def test_ortak_model_kapsamsiz_kokte_aranir():
    """models/<urun>/ altinda DEGIL, models/ kokunde."""
    from app import modeller, urunler
    t = modeller.tanimlar('findik')['bocek_teshis']
    assert t.yol == urunler.MODEL_KOK / 'bocek_teshis.pt'
    assert t.yol.parent.name != 'findik'


def test_urun_verilmese_de_ortak_kutuk_eklenir():
    """Bir kez atlandi: urun=None yolu ortagi okumuyordu ve bocek
    modeli 'kurulu degil' gorunuyordu."""
    from app import modeller
    assert 'bocek_teshis' in modeller.tanimlar()


def test_ortak_siniflar_her_uruncte_cozulur():
    from app import siniflar
    for urun in ('cilek', 'findik'):
        assert siniflar.bilgi('Mole Cricket', urun).get('tr') == 'Danaburnu'


def test_urune_ozgu_kayit_ortagi_EZER():
    """Ozellestirme mumkun kalmali: ayni ad iki yerdeyse urun kazanir."""
    from app import siniflar
    # Spider Mites hem ortak kutukte hem cilek kutugunde tanimli
    assert siniflar.bilgi('Spider Mites', 'cilek').get('tr') == 'Kırmızı Örümcek'


def test_hastaliklar_ORTAK_DEGIL():
    """Ayni ad, farkli etken -> urune ozgu kalmali."""
    from app import tedavi
    c = tedavi.coz(tedavi.yukle('cilek'), 'Leaf Spot').get('etken', '')
    f = tedavi.coz(tedavi.yukle('findik'), 'Leaf Spot').get('etken', '')
    assert 'Mycosphaerella' in c
    assert 'Piggotia' in f


def test_bocek_hala_ROI_boru_hattina_girmiyor():
    """Ortak olmasi tetigi acmaz."""
    from app import modeller
    for urun in ('cilek', 'findik'):
        for organ in ('leaf', 'fruit', 'nut', 'husk', 'branch'):
            adlar = [t.ad for t in modeller.tetiklenen(organ, urun)]
            assert 'bocek_teshis' not in adlar
