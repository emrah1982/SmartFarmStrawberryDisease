"""Koyu temada okunabilirlik.

GERÇEK HATA:
    Öneri kutusuna `background: var(--zemin2, #fafafa)` yazılmıştı ama
    `--zemin2` HİÇ TANIMLI DEĞİLDİ. Her temada #fafafa'ya (açık) düşüyordu.
    Koyu temada yazı rengi `--metin: #ece7e0` (açık) olduğu için
    AÇIK ZEMİN ÜSTÜNDE AÇIK YAZI çıktı — tedavi önerileri okunamıyordu.

    Aynı hata `tr.ilk-aday` satırında da vardı.

KURAL:
    Bir kural açık bir zemin veriyorsa YA yazı rengini de vermeli YA da
    temaya bağlı bir değişken kullanmalı. Sabit açık zemin + tema yazısı
    kombinasyonu koyu temada mutlaka okunamaz olur.
"""

import re
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parent.parent
CSS = (KOK / 'app' / 'static' / 'style.css').read_text(encoding='utf-8')

# Açık sayılan sabit renkler (kısa ve uzun biçim)
ACIK_DESEN = re.compile(
    r'#(?:f[0-9a-f]{5}|e[0-9a-f]{5}|f{3}|[ef][0-9a-f]{2})\b', re.I)


def _kurallar():
    """CSS'i kaba biçimde kurallara böler: (seçici, gövde)."""
    for eslesme in re.finditer(r'([^{}]+)\{([^{}]*)\}', CSS):
        yield eslesme.group(1).strip(), eslesme.group(2)


class TestDegiskenler:
    def test_zemin2_tanimli(self):
        """Tanımsız değişken sessizce yedeğe düşer — hata görünmez."""
        assert '--zemin2:' in CSS, '--zemin2 tanımlanmamış'

    def test_zemin2_HER_IKI_temada_tanimli(self):
        koyu = CSS[CSS.index('prefers-color-scheme: dark'):]
        assert '--zemin2:' in koyu, 'koyu temada --zemin2 yok'

    def test_zemin2_yedegi_kalmadi(self):
        """`var(--zemin2, #fafafa)` yedeği hatayı gizliyordu."""
        assert 'var(--zemin2, #' not in CSS


class TestOkunabilirlik:
    """Açık zemin veren her kural ya yazı rengi vermeli ya temaya bağlı olmalı."""

    # Bilerek her iki temada da açık kalan yerler (üstünde yazı yok ya da
    # yazı rengi ayrıca veriliyor)
    MUAF = {'.qr', '.canli-sahne video', 'header nav a.etkin', '.rozet-sayac'}

    def test_acik_zemin_veren_kural_yazi_rengi_de_verir(self):
        sorunlu = []
        for secici, govde in _kurallar():
            if secici.startswith('@') or secici in self.MUAF:
                continue
            arka = re.search(r'background(?:-color)?:\s*([^;]+);?', govde)
            if not arka:
                continue
            deger = arka.group(1)
            if not ACIK_DESEN.search(deger):
                continue                       # açık sabit renk değil
            if 'color:' in re.sub(r'background[^;]*;?', '', govde):
                continue                       # yazı rengi de verilmiş
            sorunlu.append(secici)
        assert not sorunlu, (
            'açık zemin veriyor ama yazı rengi vermiyor — koyu temada '
            f'okunamaz: {sorunlu}')


class TestBilinenHatalar:
    def test_oneri_kutusu_temaya_bagli(self):
        blok = CSS[CSS.index('.oneri-kutu {'):]
        blok = blok[:blok.index('}')]
        assert 'var(--zemin2)' in blok
        assert '#fafafa' not in blok

    def test_ilk_aday_satiri_sabit_acik_degil(self):
        blok = CSS[CSS.index('tr.ilk-aday'):]
        blok = blok[:blok.index('}')]
        assert '#f5faf5' not in blok, 'sabit açık zemin koyu temada okunmaz'

    def test_adres_kutusu_temaya_bagli(self):
        blok = CSS[CSS.index('.adres-kutu {'):]
        blok = blok[:blok.index('}')]
        assert 'var(--zemin2)' in blok
