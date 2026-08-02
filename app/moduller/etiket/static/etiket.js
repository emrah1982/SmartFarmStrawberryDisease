/* Kutu düzenleyici — canvas üzerinde çiz, taşı, boyutlandır, sil.
 *
 * NEDEN DIŞ KÜTÜPHANE YOK?
 *   Uygulama serada, internetsiz çalışabilmeli. CDN'den yüklenen bir
 *   editör bağlantı yokken sayfayı boş bırakırdı.
 *
 * KOORDİNAT DÜZENİ
 *   Kutular her zaman NORMALİZE (0-1) saklanır — görüntü ölçeği
 *   değişse de etiket bozulmaz. Ekrana çizerken canvas boyutuyla
 *   çarpılır. Fare olayları ters yönde çevrilir.
 */
(function () {
  'use strict';

  const kok = document.getElementById('duzenleyici');
  if (!kok) return;

  const AYAR = JSON.parse(document.getElementById('etiket-ayar').textContent);
  const tuval = document.getElementById('tuval');
  const ctx = tuval.getContext('2d');
  const gorsel = new Image();

  const RENK = ['#e53935', '#43a047', '#1e88e5', '#fb8c00', '#8e24aa',
                '#00acc1', '#c0ca33', '#d81b60'];
  const TUTAMAK = 8;          // köşe tutamağının piksel yarıçapı
  const EN_KUCUK = 4;         // bundan küçük sürükleme kutu sayılmaz (px)

  let kutular = AYAR.kutular.map(k => ({...k}));
  let secili = -1;
  let aktifSinif = 0;
  let kirli = false;

  let surukleme = null;   // {tur:'yeni'|'tasi'|'boyut', kose, bx, by, kutu}

  // ── çizim ──────────────────────────────────────────────────────────

  function boyutlandir() {
    const en = kok.clientWidth;
    const oran = gorsel.naturalHeight / gorsel.naturalWidth || 0.75;
    tuval.width = en;
    tuval.height = Math.round(en * oran);
    ciz();
  }

  function pikselKutu(k) {
    return {
      x: (k.cx - k.w / 2) * tuval.width,
      y: (k.cy - k.h / 2) * tuval.height,
      w: k.w * tuval.width,
      h: k.h * tuval.height,
    };
  }

  function ciz() {
    ctx.clearRect(0, 0, tuval.width, tuval.height);
    if (gorsel.complete && gorsel.naturalWidth) {
      ctx.drawImage(gorsel, 0, 0, tuval.width, tuval.height);
    }
    kutular.forEach((k, i) => {
      const p = pikselKutu(k);
      const renk = RENK[k.sinif % RENK.length];
      ctx.lineWidth = i === secili ? 4 : 2;
      ctx.strokeStyle = renk;
      ctx.strokeRect(p.x, p.y, p.w, p.h);

      // Çok küçük kutu uyarısı: YOLO ızgarası 8 px adımlı, altı öğrenilemez.
      const kisaKenar = Math.min(p.w, p.h) * (gorsel.naturalWidth / tuval.width);
      if (kisaKenar < AYAR.en_kucuk_kenar) {
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = '#ff9800';
        ctx.strokeRect(p.x - 2, p.y - 2, p.w + 4, p.h + 4);
        ctx.setLineDash([]);
      }

      const etiket = AYAR.siniflar[k.sinif] || ('sınıf ' + k.sinif);
      ctx.font = 'bold 13px system-ui, sans-serif';
      const g = ctx.measureText(etiket).width + 10;
      ctx.fillStyle = renk;
      ctx.fillRect(p.x, Math.max(0, p.y - 19), g, 19);
      ctx.fillStyle = '#fff';
      ctx.fillText(etiket, p.x + 5, Math.max(13, p.y - 5));

      if (i === secili) {
        ctx.fillStyle = renk;
        koseler(p).forEach(c => {
          ctx.fillRect(c.x - TUTAMAK / 2, c.y - TUTAMAK / 2, TUTAMAK, TUTAMAK);
        });
      }
    });
    listeCiz();
  }

  function koseler(p) {
    return [
      {ad: 'sol-ust', x: p.x, y: p.y},
      {ad: 'sag-ust', x: p.x + p.w, y: p.y},
      {ad: 'sol-alt', x: p.x, y: p.y + p.h},
      {ad: 'sag-alt', x: p.x + p.w, y: p.y + p.h},
    ];
  }

  // ── fare ───────────────────────────────────────────────────────────

  function konum(e) {
    const r = tuval.getBoundingClientRect();
    const nokta = e.touches ? e.touches[0] : e;
    return {x: nokta.clientX - r.left, y: nokta.clientY - r.top};
  }

  function vurulanKose(pt) {
    if (secili < 0) return null;
    const p = pikselKutu(kutular[secili]);
    for (const c of koseler(p)) {
      if (Math.abs(pt.x - c.x) <= TUTAMAK && Math.abs(pt.y - c.y) <= TUTAMAK) {
        return c.ad;
      }
    }
    return null;
  }

  function vurulanKutu(pt) {
    // Üstteki (sonra çizilen) önce seçilsin diye tersten bakılır.
    for (let i = kutular.length - 1; i >= 0; i--) {
      const p = pikselKutu(kutular[i]);
      if (pt.x >= p.x && pt.x <= p.x + p.w && pt.y >= p.y && pt.y <= p.y + p.h) {
        return i;
      }
    }
    return -1;
  }

  function bas(e) {
    const pt = konum(e);
    const kose = vurulanKose(pt);
    if (kose) {
      surukleme = {tur: 'boyut', kose: kose, kutu: {...kutular[secili]}};
      e.preventDefault();
      return;
    }
    const i = vurulanKutu(pt);
    if (i >= 0) {
      secili = i;
      surukleme = {tur: 'tasi', bx: pt.x, by: pt.y, kutu: {...kutular[i]}};
    } else {
      secili = -1;
      surukleme = {tur: 'yeni', bx: pt.x, by: pt.y};
    }
    ciz();
    e.preventDefault();
  }

  function hareket(e) {
    if (!surukleme) return;
    const pt = konum(e);
    const W = tuval.width, H = tuval.height;

    if (surukleme.tur === 'yeni') {
      const x0 = Math.min(surukleme.bx, pt.x), x1 = Math.max(surukleme.bx, pt.x);
      const y0 = Math.min(surukleme.by, pt.y), y1 = Math.max(surukleme.by, pt.y);
      ciz();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = RENK[aktifSinif % RENK.length];
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
    } else if (surukleme.tur === 'tasi') {
      const k = surukleme.kutu;
      const dx = (pt.x - surukleme.bx) / W, dy = (pt.y - surukleme.by) / H;
      const yeni = {...k, cx: k.cx + dx, cy: k.cy + dy};
      // Kutu kadraj dışına taşmasın — taşan etiket eğitimde uyarı üretir.
      yeni.cx = Math.min(1 - yeni.w / 2, Math.max(yeni.w / 2, yeni.cx));
      yeni.cy = Math.min(1 - yeni.h / 2, Math.max(yeni.h / 2, yeni.cy));
      kutular[secili] = yeni;
      kirli = true;
      ciz();
    } else if (surukleme.tur === 'boyut') {
      const k = surukleme.kutu;
      let x0 = k.cx - k.w / 2, y0 = k.cy - k.h / 2;
      let x1 = k.cx + k.w / 2, y1 = k.cy + k.h / 2;
      const nx = Math.min(1, Math.max(0, pt.x / W));
      const ny = Math.min(1, Math.max(0, pt.y / H));
      if (surukleme.kose.includes('sol')) x0 = nx; else x1 = nx;
      if (surukleme.kose.includes('ust')) y0 = ny; else y1 = ny;
      const a = Math.min(x0, x1), b = Math.max(x0, x1);
      const c = Math.min(y0, y1), d = Math.max(y0, y1);
      if (b - a > 0.002 && d - c > 0.002) {
        kutular[secili] = {...k, cx: (a + b) / 2, cy: (c + d) / 2,
                           w: b - a, h: d - c};
        kirli = true;
        ciz();
      }
    }
    e.preventDefault();
  }

  function birak(e) {
    if (!surukleme) return;
    if (surukleme.tur === 'yeni') {
      const pt = konum(e.changedTouches ? {touches: e.changedTouches} : e);
      const x0 = Math.min(surukleme.bx, pt.x), x1 = Math.max(surukleme.bx, pt.x);
      const y0 = Math.min(surukleme.by, pt.y), y1 = Math.max(surukleme.by, pt.y);
      if (x1 - x0 > EN_KUCUK && y1 - y0 > EN_KUCUK) {
        kutular.push({
          sinif: aktifSinif,
          cx: (x0 + x1) / 2 / tuval.width,
          cy: (y0 + y1) / 2 / tuval.height,
          w: (x1 - x0) / tuval.width,
          h: (y1 - y0) / tuval.height,
        });
        secili = kutular.length - 1;
        kirli = true;
      }
    }
    surukleme = null;
    ciz();
  }

  // ── kutu listesi ───────────────────────────────────────────────────

  function listeCiz() {
    const el = document.getElementById('kutu-listesi');
    if (!el) return;
    if (!kutular.length) {
      el.innerHTML = '<p class="ipucu">Kutu yok. ' +
        'Görüntü üzerinde sürükleyerek ekleyin — <strong>kutusuz kare ' +
        'geçerli bir negatif örnektir</strong>, boş bırakmak da bir karardır.</p>';
      return;
    }
    el.innerHTML = kutular.map((k, i) => {
      const secenekler = AYAR.siniflar.map((s, j) =>
        `<option value="${j}" ${j === k.sinif ? 'selected' : ''}>${s}</option>`
      ).join('');
      const alan = (k.w * k.h * 100).toFixed(1);
      return `<li class="${i === secili ? 'secili' : ''}" data-i="${i}">
        <span class="renk-nokta" style="background:${RENK[k.sinif % RENK.length]}"></span>
        <select data-sinif="${i}">${secenekler}</select>
        <small>alan %${alan}</small>
        <button class="btn kucuk" data-sil="${i}" title="Sil">✕</button>
      </li>`;
    }).join('');
  }

  // ── kaydet ─────────────────────────────────────────────────────────

  async function kaydet(onayla) {
    const d = document.getElementById('durum');
    d.textContent = 'Kaydediliyor…';
    d.className = 'ipucu';
    try {
      const r = await fetch(AYAR.kaydet_yolu, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({kutular: kutular, onayla: !!onayla}),
      });
      const veri = await r.json();
      if (!r.ok) {
        d.textContent = veri.detail || 'Kaydedilemedi';
        d.className = 'ipucu hatali';
        return;
      }
      kirli = false;
      d.textContent = veri.mesaj;
      d.className = 'ipucu basarili';
      if (onayla) {
        if (veri.sonraki) {
          location.href = AYAR.kare_yolu_kalibi.replace('__KARE__',
            encodeURIComponent(veri.sonraki));
        } else {
          d.textContent = veri.mesaj + ' — incelenmemiş kare kalmadı.';
        }
      }
    } catch (e) {
      d.textContent = 'Kaydedilemedi: ' + e;
      d.className = 'ipucu hatali';
    }
  }

  // ── olaylar ────────────────────────────────────────────────────────

  gorsel.onload = boyutlandir;
  gorsel.onerror = function () {
    const d = document.getElementById('durum');
    d.textContent = 'Görüntü yüklenemedi: ' + AYAR.goruntu_yolu;
    d.className = 'ipucu hatali';
  };
  gorsel.src = AYAR.goruntu_yolu;
  window.addEventListener('resize', boyutlandir);

  tuval.addEventListener('mousedown', bas);
  tuval.addEventListener('mousemove', hareket);
  window.addEventListener('mouseup', birak);
  tuval.addEventListener('touchstart', bas, {passive: false});
  tuval.addEventListener('touchmove', hareket, {passive: false});
  tuval.addEventListener('touchend', birak);

  document.getElementById('kutu-listesi').addEventListener('change', e => {
    const i = e.target.dataset.sinif;
    if (i !== undefined) {
      kutular[+i].sinif = +e.target.value;
      kirli = true;
      ciz();
    }
  });
  document.getElementById('kutu-listesi').addEventListener('click', e => {
    const sil = e.target.dataset.sil;
    if (sil !== undefined) {
      kutular.splice(+sil, 1);
      secili = -1;
      kirli = true;
      ciz();
      return;
    }
    const li = e.target.closest('li[data-i]');
    if (li) { secili = +li.dataset.i; ciz(); }
  });

  document.querySelectorAll('[data-aktif-sinif]').forEach(b => {
    b.addEventListener('click', () => {
      aktifSinif = +b.dataset.aktifSinif;
      document.querySelectorAll('[data-aktif-sinif]').forEach(
        x => x.classList.toggle('etkin', x === b));
      if (secili >= 0) { kutular[secili].sinif = aktifSinif; kirli = true; }
      ciz();
    });
  });

  document.getElementById('kaydet').addEventListener('click', () => kaydet(false));
  document.getElementById('onayla').addEventListener('click', () => kaydet(true));
  document.getElementById('temizle').addEventListener('click', () => {
    if (kutular.length && !confirm('Bu karedeki ' + kutular.length +
        ' kutunun hepsi silinsin mi?')) return;
    kutular = []; secili = -1; kirli = true; ciz();
  });

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if ((e.key === 'Delete' || e.key === 'Backspace') && secili >= 0) {
      kutular.splice(secili, 1); secili = -1; kirli = true; ciz();
      e.preventDefault();
    } else if (e.key === 'Escape') {
      secili = -1; ciz();
    } else if (e.key >= '1' && e.key <= '9') {
      const j = +e.key - 1;
      if (j < AYAR.siniflar.length) {
        aktifSinif = j;
        if (secili >= 0) { kutular[secili].sinif = j; kirli = true; }
        document.querySelectorAll('[data-aktif-sinif]').forEach(
          x => x.classList.toggle('etkin', +x.dataset.aktifSinif === j));
        ciz();
      }
    } else if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault(); kaydet(false);
    }
  });

  // Kaydedilmemiş değişiklikle sayfadan çıkmak, yapılan işi sessizce
  // çöpe atar — kullanıcı düzeltmeyi kaybettiğini fark etmez.
  window.addEventListener('beforeunload', e => {
    if (kirli) { e.preventDefault(); e.returnValue = ''; }
  });
})();
