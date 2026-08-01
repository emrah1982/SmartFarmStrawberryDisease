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
const modSecim = $('mod'), modNotu = $('modNotu');
let calisiyor = false, sonKare = 0, fpsGecmisi = [];

// Kullanıcı ne kadar kayıt açacağını baştan bilsin — 'hepsi' modu diski hızla doldurur.
const MOD_NOTU = {
  akilli: '🎯 Yalnızca kararlı bulgular kaydedilir. Akış kaydedilmez.',
  tespitli: `📋 Tespit içeren her kare kaydedilir (${A.mod_aralik} sn'de en fazla bir tane).`,
  hepsi: `🗃️ Tespit olmayanlar dahil her kare kaydedilir — modelin kaçırdıklarını ` +
         `toplamak için. Oturum sınırı: ${A.azami} kayıt.`,
};
function modNotuYaz() { modNotu.textContent = MOD_NOTU[modSecim.value] || ''; }
modNotuYaz();

function sonucGeldi(s) {
  if (s.tip === 'hata') { durum.textContent = s.mesaj; return; }
  cizim.guncelle(s.kutular);

  const gecen = performance.now() - sonKare;
  fpsGecmisi.push(gecen); if (fpsGecmisi.length > 10) fpsGecmisi.shift();
  const ort = fpsGecmisi.reduce((a, b) => a + b, 0) / fpsGecmisi.length;
  sayac.textContent = `${(1000 / ort).toFixed(1)} kare/sn · model ${s.ms} ms`;

  const kayitBilgi = s.doldu
    ? ` · ⛔ oturum sınırı doldu (${s.sayac}) — kayıt durdu, tespit sürüyor`
    : (s.sayac ? ` · 💾 ${s.sayac}/${A.azami} kayıt` : '');
  // Ekrandaki kutu sayısı ANLIK, tur boyunca görülen FARKLI nesne sayısı
  // ayrı bir bilgidir: aynı meyve onlarca karede görünür, kutuları toplamak
  // anlamsız olurdu (bkz. app/takip.py).
  const turBilgi = s.benzersiz
    ? ` · 🍓 turda ${s.benzersiz} farklı nesne`
    : '';

  // Çizgi sayımı açıksa geçiş sayısı; SABİT çekimde uyarı.
  // Çizgi yalnızca kamera ilerlerken anlamlıdır, sabitken hep 0 verir.
  let cizgiBilgi = '';
  if (s.cizgi && Object.keys(s.cizgi).length) {
    const sabit = s.oneri && s.oneri.kamera === 'sabit';
    cizgiBilgi = ` · 📏 çizgiden geçen: ${s.cizgi.toplam}`;
    if (sabit && s.cizgi.toplam === 0) {
      cizgiBilgi += ' (kamera sabit — çizgi sayımı bu çekimde işe yaramaz)';
    }
  }

  durum.textContent = (s.bulanik
    ? '⚠️ Görüntü bulanık — sabit tutun, bu kare atlandı.'
    : (s.kutular.length ? `${s.kutular.length} tespit (ekranda)` : 'Tespit yok'))
    + turBilgi + cizgiBilgi + kayitBilgi;

  liste.innerHTML = s.kutular.length
    ? s.kutular.map(k => {
        const ad = (window.CANLI_ADLAR || {})[k.ad] || k.ad;
        return `<span class="rozet">${ad} %${(k.guven * 100).toFixed(0)}</span>`;
      }).join(' ')
    : '';

  if (s.kayit_id) kayitEkle(s.kayit_id, s.kayit_tipi, s.kutular);
  if (calisiyor) sonrakiKare();
}

function kayitEkle(id, tip, kutular) {
  const ad = kutular.length
    ? ((window.CANLI_ADLAR || {})[kutular[0].ad] || kutular[0].ad) : 'kayıt';
  const el = document.createElement('a');
  el.href = `/kayit/${id}`;
  el.className = 'rozet kayit-rozet';
  const simge = { elle: '💾', akilli: '⚡', tespitli: '📋', hepsi: '🗃️' }[tip] || '⚡';
  el.textContent = `${simge} #${id} ${ad}`;
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
    akis.modSec(modSecim.value);
    cizgiUygula();                  // çizgi ayarı oturuma taşınsın
    modSecim.disabled = true;       // oturum ortasında mod değişimi kafa karıştırır
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
  modSecim.disabled = false;
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
modSecim.onchange = modNotuYaz;

// ───────────────────────────────────────────── sanal çizgi sayacı
const cizgiAcik = $('cizgiAcik'), cizgiSecenek = $('cizgiSecenek');
const cizgiEksen = $('cizgiEksen'), cizgiKonum = $('cizgiKonum');
const cizgiKonumYazi = $('cizgiKonumYazi');

function cizgiAyari() {
  if (!cizgiAcik || !cizgiAcik.checked) return null;
  return {
    acik: true,
    eksen: cizgiEksen.value,
    konum: Number(cizgiKonum.value) / 100,
  };
}

function cizgiUygula() {
  if (!cizgiAcik) return;
  const ayar = cizgiAyari();
  if (cizgiSecenek) cizgiSecenek.hidden = !cizgiAcik.checked;
  if (cizgiKonumYazi) cizgiKonumYazi.textContent = `%${cizgiKonum.value}`;
  // Tuvalde göster: kullanıcı nesnenin nereyi geçince sayıldığını görmeli
  cizim.cizgiGuncelle(ayar);
  akis.cizgiSec(ayar ? { ...ayar } : { acik: false });
}

if (cizgiAcik) {
  cizgiAcik.onchange = cizgiUygula;
  cizgiEksen.onchange = cizgiUygula;
  cizgiKonum.oninput = cizgiUygula;
  cizgiUygula();
}

// Sekme arkaya alınınca kamerayı ve akışı boşuna çalıştırma (pil/veri).
document.addEventListener('visibilitychange', () => {
  if (document.hidden && calisiyor) durdur();
});
window.addEventListener('pagehide', () => { if (calisiyor) durdur(); });
