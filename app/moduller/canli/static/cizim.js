// Çizim bileşeni: kutuları görüntünün üzerine bindirir.
// Ağ ve kamera bilmez; yalnızca "kutu listesi → tuval" işini yapar.

const RENKLER = ['#e53935', '#8e24aa', '#1e88e5', '#00897b', '#f4511e',
                 '#3949ab', '#c0ca33', '#6d4c41', '#039be5', '#7cb342'];

export class Cizim {
  /** @param {HTMLCanvasElement} tuval  video ile aynı yere bindirilmiş katman
   *  @param {HTMLVideoElement}  video  boyut kaynağı */
  constructor(tuval, video) {
    this.tuval = tuval;
    this.video = video;
    this.kutular = [];
    this._surer = false;
  }

  guncelle(kutular) { this.kutular = kutular || []; }

  baslat() {
    if (this._surer) return;
    this._surer = true;
    const dongu = () => {
      if (!this._surer) return;
      this.ciz();
      requestAnimationFrame(dongu);
    };
    requestAnimationFrame(dongu);
  }

  durdur() { this._surer = false; this.temizle(); }

  temizle() {
    const c = this.tuval.getContext('2d');
    c.clearRect(0, 0, this.tuval.width, this.tuval.height);
  }

  ciz() {
    const v = this.video;
    // Tuval, videonun EKRANDAKİ boyutuna eşitlenir; kutular normalize (0-1)
    // geldiği için çözünürlük değişse de hizalama bozulmaz.
    const g = v.clientWidth, y = v.clientHeight;
    if (!g || !y) return;
    if (this.tuval.width !== g || this.tuval.height !== y) {
      this.tuval.width = g; this.tuval.height = y;
    }

    const c = this.tuval.getContext('2d');
    c.clearRect(0, 0, g, y);
    c.lineWidth = Math.max(2, Math.round(g / 320));
    c.font = `${Math.max(12, Math.round(g / 34))}px system-ui, sans-serif`;
    c.textBaseline = 'top';

    for (const k of this.kutular) {
      const renk = RENKLER[k.sinif_id % RENKLER.length];
      const x1 = (k.x - k.w / 2) * g, y1 = (k.y - k.h / 2) * y;
      const gw = k.w * g, gy = k.h * y;

      c.strokeStyle = renk;
      c.strokeRect(x1, y1, gw, gy);

      const etiket = `${k.ad} ${(k.guven * 100).toFixed(0)}%`;
      const genislik = c.measureText(etiket).width + 8;
      const yukseklik = parseInt(c.font, 10) + 6;
      const etiketY = y1 > yukseklik ? y1 - yukseklik : y1;   // ekran dışına taşmasın
      c.fillStyle = renk;
      c.fillRect(x1 - c.lineWidth / 2, etiketY, genislik, yukseklik);
      c.fillStyle = '#fff';
      c.fillText(etiket, x1 + 4 - c.lineWidth / 2, etiketY + 3);
    }
  }
}
