// Akış bileşeni: kareyi sunucuya gönderir, sonucu geri verir.
// WebSocket kullanır; engelliyse REST'e düşer. Kamera ve çizim bilmez.

export class Akis {
  /**
   * @param {object} o
   * @param {(sonuc:object)=>void} o.onSonuc  her karenin sonucu
   * @param {(m:string)=>void}     o.onDurum  kullanıcıya gösterilecek durum
   */
  constructor({ onSonuc, onDurum }) {
    this.onSonuc = onSonuc;
    this.onDurum = onDurum || (() => {});
    this.ws = null;
    this.restModu = false;
    this.bekleyen = false;          // uçuşta kare var mı (geri basınç)
    this.oturum = Math.random().toString(36).slice(2, 10);
    this.seraId = null;
    this.mod = null;
  }

  bagli() { return this.restModu || (this.ws && this.ws.readyState === WebSocket.OPEN); }

  baglan() {
    return new Promise(cozum => {
      let bitti = false;
      const tamam = (rest) => { if (!bitti) { bitti = true; this.restModu = rest; cozum(); } };
      try {
        const p = location.protocol === 'https:' ? 'wss' : 'ws';
        this.ws = new WebSocket(`${p}://${location.host}/canli/ws`);
        this.ws.onopen = () => { this._ayarGonder(); tamam(false); };
        this.ws.onmessage = e => {
          this.bekleyen = false;
          try { this.onSonuc(JSON.parse(e.data)); } catch (_) {}
        };
        this.ws.onerror = () => {
          // Vekil sunucu/ağ WebSocket'i engelliyor olabilir; REST yedeği var.
          this.onDurum('WebSocket kurulamadı, yedek yönteme geçildi.');
          tamam(true);
        };
        this.ws.onclose = () => { this.bekleyen = false; };
        setTimeout(() => tamam(true), 4000);          // takılırsa yedeğe geç
      } catch (_) {
        tamam(true);
      }
    });
  }

  seraSec(id) {
    this.seraId = id || null;
    this._ayarGonder();
  }

  modSec(mod) {
    this.mod = mod || null;
    this._ayarGonder();
  }

  /** Sanal cizgi sayaci ayari. acik=false ise sayac kapatilir. */
  cizgiSec(ayar) {
    this.cizgi = ayar || null;
    this._ayarGonder();
  }

  _ayarGonder() {
    if (!this.restModu && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ tip: 'ayar', sera_id: this.seraId,
                                    mod: this.mod, cizgi: this.cizgi }));
    }
  }

  /** Sıradaki karenin kaydedilmesini ister (kullanıcı düğmesi). */
  kaydetIste() {
    this._kaydet = true;
    if (!this.restModu && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ tip: 'kaydet' }));
      this._kaydet = false;
    }
  }

  /** Kareyi gönderir. Uçuşta kare varken yenisi gönderilmez: sunucu ne kadar
   *  hızlıysa akış o hızda ilerler, kuyruk birikmez. */
  async gonder(blob) {
    if (this.bekleyen || !blob) return false;
    this.bekleyen = true;
    if (!this.restModu) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) { this.bekleyen = false; return false; }
      this.ws.send(blob);
      return true;
    }
    try {
      const fd = new FormData();
      fd.append('kare', blob, 'kare.jpg');
      fd.append('oturum', this.oturum);
      if (this.seraId) fd.append('sera_id', this.seraId);
      if (this.mod) fd.append('mod', this.mod);
      if (this._kaydet) { fd.append('kaydet', '1'); this._kaydet = false; }
      const y = await fetch('/canli/kare', { method: 'POST', body: fd });
      this.onSonuc(await y.json());
    } catch (e) {
      this.onDurum('Sunucuya ulaşılamadı: ' + e.message);
    } finally {
      this.bekleyen = false;
    }
    return true;
  }

  kapat() {
    if (this.ws) { try { this.ws.close(); } catch (_) {} }
    this.ws = null; this.bekleyen = false;
  }
}
