// Kamera bileşeni: cihaz kamerasını açar/kapatır ve kare üretir.
// Yalnızca kamerayla ilgilenir — ağ, çizim, kayıt bilmez.

export class Kamera {
  constructor(videoEl) {
    this.video = videoEl;
    this.akis = null;
    this.tuval = document.createElement('canvas');   // kare almak için gizli tuval
    this.yon = 'environment';                        // arka kamera (bitkiye bakan)
  }

  get acik() { return !!this.akis; }

  async ac() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Bu tarayıcı kamera erişimini desteklemiyor.');
    }
    if (!window.isSecureContext) {
      // Tarayıcılar http:// üzerinden kamerayı engeller (localhost hariç).
      throw new Error('Kamera yalnızca güvenli bağlantıda (https veya localhost) açılır.');
    }
    this.kapat();
    this.akis = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: this.yon, width: { ideal: 1280 } },
      audio: false,
    });
    this.video.srcObject = this.akis;
    await this.video.play();
    return this.akis;
  }

  async cevir() {
    this.yon = this.yon === 'environment' ? 'user' : 'environment';
    if (this.acik) await this.ac();
  }

  kapat() {
    if (this.akis) this.akis.getTracks().forEach(t => t.stop());
    this.akis = null;
    this.video.srcObject = null;
  }

  /** Anlık kareyi JPEG blob olarak verir. Genişlik küçültülür: ağda taşınan
   *  veri ve modele giren boyut küçülünce akış belirgin hızlanır. */
  async kare(genislik = 640, kalite = 0.6) {
    const v = this.video;
    if (!v.videoWidth) return null;
    const oran = v.videoHeight / v.videoWidth;
    this.tuval.width = genislik;
    this.tuval.height = Math.round(genislik * oran);
    this.tuval.getContext('2d').drawImage(v, 0, 0, this.tuval.width, this.tuval.height);
    return new Promise(c => this.tuval.toBlob(c, 'image/jpeg', kalite));
  }
}
