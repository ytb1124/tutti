// AudioWorklet: 32 sources -> 16 speaker outputs (+ stereo monitor output).
// Per source: distance HF 1st-order filter -> normalisation shelf biquad -> gain matrix (direct + 24-sample tap), plus
// aux sends taken from the unfiltered input. Gains are ramped linearly when a new state arrives; filter coefficients
// switch at block boundaries. This is the reference renderer: helper/TuttiBridge/Renderer.swift does the same.
const NS = 32, NO = 16, TAP = 24, B = 128;

class RenderProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const m = () => ({ d: new Float32Array(NS * NO), e: new Float32Array(NS * NO), a: new Float32Array(NS * NO) });
    this.cur = m(); this.tgt = m(); this.inc = m();
    this.left = 0;
    this.bq = new Float32Array(NS * 5);
    this.hf = new Float32Array(NS * 3);
    this.mon = new Float32Array(NO * 2);
    for (let k = 0; k < NS; k++) { this.bq[k * 5] = 1; this.hf[k * 3] = 1; }
    this.zb = new Float64Array(NS * 2);          // biquad state (DF2T)
    this.zh = new Float64Array(NS * 2);          // HF state x1, y1
    this.hist = new Float32Array(NS * (TAP + B)); // filtered signal with TAP samples of history
    this.tmp = new Float32Array(B);
    this.peakIn = new Float32Array(NS);
    this.peakOut = new Float32Array(NO);
    this.blocks = 0;
    this.meterEvery = 16;
    this.busy = 0;
    const st = options && options.processorOptions && options.processorOptions.state;
    if (st) this.apply(st, 0);
    this.port.onmessage = ev => {
      const m = ev.data;
      if (m.type === 'state') this.apply(m.state, m.ramp ?? 1024);
      else if (m.type === 'meterRate') this.meterEvery = m.blocks;
    };
  }

  apply(s, ramp) {
    this.bq.set(s.biquad); this.hf.set(s.hf); this.mon.set(s.monitor);
    this.tgt.d.set(s.direct); this.tgt.e.set(s.delayed);
    if (s.aux) this.tgt.a.set(s.aux); else this.tgt.a.fill(0);
    if (!ramp) {
      this.cur.d.set(this.tgt.d); this.cur.e.set(this.tgt.e); this.cur.a.set(this.tgt.a); this.left = 0; return;
    }
    for (let i = 0; i < NS * NO; i++) {
      this.inc.d[i] = (this.tgt.d[i] - this.cur.d[i]) / ramp;
      this.inc.e[i] = (this.tgt.e[i] - this.cur.e[i]) / ramp;
      this.inc.a[i] = (this.tgt.a[i] - this.cur.a[i]) / ramp;
    }
    this.left = ramp;
  }

  process(inputs, outputs) {
    const t0 = currentTime;
    const inp = inputs[0] || [];
    const out = outputs[0], mon = outputs[1];
    const n = out && out[0] ? out[0].length : B;
    const nOut = out ? out.length : 0;
    for (let j = 0; j < nOut; j++) out[j].fill(0);
    const steps = Math.min(n, this.left);
    const { cur, inc, tgt, tmp, hist } = this;
    for (let k = 0; k < NS; k++) {
      const x = inp[k];
      let pk = 0;
      // filters
      const kk = this.hf[k * 3], a = this.hf[k * 3 + 1], bh = this.hf[k * 3 + 2];
      const b0 = this.bq[k * 5], b1 = this.bq[k * 5 + 1], b2 = this.bq[k * 5 + 2], a1 = this.bq[k * 5 + 3], a2 = this.bq[k * 5 + 4];
      let x1 = this.zh[k * 2], y1 = this.zh[k * 2 + 1], z1 = this.zb[k * 2], z2 = this.zb[k * 2 + 1];
      let energy = 0, raw = 0;
      for (let i = 0; i < n; i++) {
        const xi = x ? x[i] : 0;
        const ax = xi < 0 ? -xi : xi; if (ax > pk) pk = ax;
        raw += ax;
        const h = a * y1 + kk * (xi - bh * x1); x1 = xi; y1 = h;
        const y = b0 * h + z1; z1 = b1 * h - a1 * y + z2; z2 = b2 * h - a2 * y;
        tmp[i] = y; energy += y < 0 ? -y : y;
      }
      this.zh[k * 2] = x1; this.zh[k * 2 + 1] = y1; this.zb[k * 2] = z1; this.zb[k * 2 + 1] = z2;
      if (pk > this.peakIn[k]) this.peakIn[k] = pk;
      // history buffer: [TAP previous samples | this block]
      const hb = k * (TAP + B);
      hist.copyWithin(hb, hb + n, hb + n + TAP);
      hist.set(tmp.subarray(0, n), hb + TAP);
      const silent = energy === 0 && this.isSilentHistory(hb);
      for (let j = 0; j < nOut; j++) {
        const idx = k * NO + j;
        let gd = cur.d[idx], ge = cur.e[idx];
        const td = tgt.d[idx], te = tgt.e[idx];
        if (steps === 0 && gd === 0 && ge === 0) continue;
        if (silent) { if (steps) { cur.d[idx] += inc.d[idx] * steps; cur.e[idx] += inc.e[idx] * steps; } continue; }
        const o = out[j];
        const did = inc.d[idx], die = inc.e[idx];
        let i = 0;
        for (; i < steps; i++) { o[i] += gd * hist[hb + TAP + i] + ge * hist[hb + i]; gd += did; ge += die; }
        if (steps && steps === this.left) { gd = td; ge = te; }
        for (; i < n; i++) o[i] += gd * hist[hb + TAP + i] + ge * hist[hb + i];
        cur.d[idx] = gd; cur.e[idx] = ge;
      }
      // aux sends: the unfiltered input
      for (let j = 0; j < nOut; j++) {
        const idx = k * NO + j;
        let ga = cur.a[idx];
        if (steps === 0 && ga === 0) continue;
        if (raw === 0) { if (steps) cur.a[idx] += inc.a[idx] * steps; continue; }
        const o = out[j], dia = inc.a[idx];
        let i = 0;
        for (; i < steps; i++) { o[i] += ga * x[i]; ga += dia; }
        if (steps && steps === this.left) ga = tgt.a[idx];
        for (; i < n; i++) o[i] += ga * x[i];
        cur.a[idx] = ga;
      }
    }
    if (this.left > 0) {
      this.left -= steps;
      if (this.left <= 0) { cur.d.set(tgt.d); cur.e.set(tgt.e); cur.a.set(tgt.a); }
    }
    for (let j = 0; j < nOut; j++) {
      const o = out[j]; let pk = this.peakOut[j];
      for (let i = 0; i < n; i++) { const v = o[i] < 0 ? -o[i] : o[i]; if (v > pk) pk = v; }
      this.peakOut[j] = pk;
    }
    if (mon && mon.length >= 2) {
      const L = mon[0], R = mon[1];
      L.fill(0); R.fill(0);
      for (let j = 0; j < nOut; j++) {
        const gl = this.mon[j * 2], gr = this.mon[j * 2 + 1];
        if (!gl && !gr) continue;
        const o = out[j];
        for (let i = 0; i < n; i++) { L[i] += gl * o[i]; R[i] += gr * o[i]; }
      }
    }
    if (++this.blocks >= this.meterEvery) {
      this.blocks = 0;
      this.port.postMessage({ type: 'meters', inPeak: this.peakIn.slice(), outPeak: this.peakOut.slice() });
      this.peakIn.fill(0); this.peakOut.fill(0);
    }
    void t0;
    return true;
  }

  isSilentHistory(hb) {
    for (let i = 0; i < TAP; i++) if (this.hist[hb + i] !== 0) return false;
    return true;
  }
}

registerProcessor('render-processor', RenderProcessor);
