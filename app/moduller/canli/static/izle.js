// Yapıştırıcı: kamera + akış + çizim bileşenlerini birbirine bağlar.
// Kendi başına iş mantığı içermez; her parça kendi dosyasında test edilebilir.

import { Kamera } from '/statik/canli/kamera.js';
import { Akis } from '/statik/canli/akis.js';
import { Cizim } from '/statik/canli/cizim.js';

const A = window.CANLI_AYAR || {};
const $ = id => document.getElementById(id);

const video = $('video'), tuval = $('tuval');
const kamera = new Kamera(video);
const cizim = new Cizim(tuval, video);
const akis = new Akis({ onSonuc: sonucGeldi, onDurum: m => (durum.textContent = m) });

const durum = $('durum'), sayac = $('sayac'), liste = $('liste'), kayitlar = $('kayitlar');
let calisiyor = false, sonKare = 0, fpsGecmisi = [];

function sonucGeldi(s) {
  if (s.tip === 'hata') { durum.textContent = s.mesaj; return; }
  cizim.guncelle(s.kutular);

  const gecen = performance.now() - sonKare;
  fpsGecmisi.push(gecen); if (fpsGecmisi.length > 10) fpsGecmisi.shift();
  const ort = fpsGecmisi.reduce((a, b) => a + b, 0) / fpsGecmisi.length;
  sayac.textContent = `${(1000 / ort).toFixed(1)} kare/sn · model ${s.ms} ms`;

  durum.textContent = s.bulanik
    ? '⚠️ Görüntü bulanık — sabit tutun, bu kare atlandı.'
    : (s.kutular.length ? `${s.kutular.length} tespit` : 'Tespit yok');

  liste.innerHTML = s.kutular.length
    ? s.kutular.map(k => `<span class="rozet">${k.ad} %${(k.guven * 100).toFixed(0)}</span>`).join(' ')
    : '';

  if (s.kayit_id) kayitEkle(s.kayit_id, s.kayit_tipi, s.kutular);
  if (calisiyor) sonrakiKare();
}

function kayitEkle(id, tip, kutular) {
  const ad = kutular.length ? kutular[0].ad : 'kayıt';
  const el = document.createElement('a');
  el.href = `/kayit/${id}`;
  el.className = 'rozet kayit-rozet';
  el.textContent = `${tip === 'elle' ? '💾' : '⚡'} #${id} ${ad}`;
  kayitlar.prepend(el);
}

async function sonrakiKare() {
  // En az aralık: sunucu çok hızlıysa boşuna CPU yakmayalım.
  const bekle = Math.max(0, A.en_az_aralik - (performance.now() - sonKare));
  if (bekle) await new Promise(r => setTimeout(r, bekle));
  if (!calisiyor) return;
  const blob = await kamera.kare(A.genislik, A.kalite);
  sonKare = performance.now();
  if (!(await akis.gonder(blob))) setTimeout(sonrakiKare, 300);   // bağlantı yoksa tekrar dene
}

async function basla() {
  try {
    durum.textContent = 'Kamera açılıyor...';
    await kamera.ac();
    $('sahne').hidden = false;
    await akis.baglan();
    akis.seraSec($('sera') ? $('sera').value : null);
    calisiyor = true;
    cizim.baslat();
    $('baslaBtn').hidden = true;
    $('durdurBtn').hidden = false;
    $('kaydetBtn').hidden = false;
    $('cevirBtn').hidden = false;
    durum.textContent = 'Analiz ediliyor...';
    sonrakiKare();
  } catch (e) {
    durum.textContent = '❌ ' + e.message;
  }
}

function durdur() {
  calisiyor = false;
  cizim.durdur();
  akis.kapat();
  kamera.kapat();
  $('sahne').hidden = true;
  $('baslaBtn').hidden = false;
  $('durdurBtn').hidden = true;
  $('kaydetBtn').hidden = true;
  $('cevirBtn').hidden = true;
  durum.textContent = 'Durduruldu.';
}

$('baslaBtn').onclick = basla;
$('durdurBtn').onclick = durdur;
$('kaydetBtn').onclick = () => { akis.kaydetIste(); durum.textContent = 'Bu kare kaydediliyor...'; };
$('cevirBtn').onclick = () => kamera.cevir();
if ($('sera')) $('sera').onchange = e => akis.seraSec(e.target.value);

// Sekme arkaya alınınca kamerayı ve akışı boşuna çalıştırma (pil/veri).
document.addEventListener('visibilitychange', () => {
  if (document.hidden && calisiyor) durdur();
});
window.addEventListener('pagehide', () => { if (calisiyor) durdur(); });
