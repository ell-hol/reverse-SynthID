// Client-side port of comfy_bypass.py. No upload — all processing is local.
//
// Pipeline:
//   1. Denoise: attenuate the high-frequency residual that carries the
//      watermark noise (spatial, edge-preserving-ish via Gaussian separation).
//   2. Carrier scramble: at the detector's 512px scale, perturb FFT phase at
//      the documented SynthID carrier bins, then add the resulting smooth
//      low-frequency delta back onto the full-resolution image.
//
// Carrier bins are the exact ones from artifacts/codebook/robust_codebook.pkl
// (image_size = 512). Listed as [fy, fx] offsets from the spectrum center.
const CARRIERS = [
  [-5,-3],[5,3],[-5,3],[5,-3],[-3,-4],[3,4],[-3,4],[3,-4],[-4,-3],[4,3],
  [-4,3],[4,-3],[-5,-1],[5,1],[-5,1],[5,-1],[-5,-2],[5,2],[-5,2],[5,-2],
  [-2,-5],[2,5],[-2,5],[2,-5],[-1,-5],[1,5],[-1,5],[1,-5],[-4,-4],[4,4],
  [-4,4],[4,-4],[-1,-6],[1,6],[-3,-5],[3,5],                 // dark set
  [0,-7],[0,7],[0,-8],[0,8],[0,-9],[0,9],[0,-10],[0,10],[0,-11],[0,11],
  [0,-12],[0,12],[0,-20],[0,20],[0,-21],[0,21],[0,-22],[0,22],[0,-23],[0,23], // white set
];
const CB_SIZE = 512;

const $ = (id) => document.getElementById(id);
let imgEl = null;          // original image element
let processedCanvas = null; // full-res processed result (for detector + download)

$('file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  imgEl = new Image();
  imgEl.onload = () => {
    drawTo($('before'), imgEl);
    $('run').disabled = false;
    $('check').disabled = false;
    processedCanvas = null;
    $('status').textContent = `Loaded ${imgEl.naturalWidth}×${imgEl.naturalHeight}.`;
    $('download').classList.add('hidden');
    $('psnr').textContent = '';
    $('verdictBefore').className = 'verdict';
    $('verdictBefore').textContent = '';
    $('verdictAfter').className = 'verdict';
    $('verdictAfter').textContent = '';
    const ctx = $('after').getContext('2d');
    ctx.clearRect(0, 0, $('after').width, $('after').height);
  };
  imgEl.src = url;
});

['denoise', 'spectral'].forEach((id) => {
  $(id).addEventListener('input', () => {
    $(id + 'Out').textContent = parseFloat($(id).value).toFixed(2);
  });
});

$('run').addEventListener('click', () => {
  if (!imgEl) return;
  $('status').textContent = 'Processing…';
  $('run').disabled = true;
  // let the UI paint the status before the heavy work
  setTimeout(() => {
    try {
      process(imgEl);
    } catch (err) {
      $('status').textContent = 'Error: ' + err.message;
    } finally {
      $('run').disabled = false;
    }
  }, 30);
});

$('check').addEventListener('click', () => {
  if (!imgEl) return;
  renderVerdict($('verdictBefore'), detect(imgEl), 'Original');
  if (processedCanvas) {
    renderVerdict($('verdictAfter'), detect(processedCanvas), 'Processed');
  } else {
    $('verdictAfter').className = 'verdict';
    $('verdictAfter').textContent = 'Process the image first to check the result.';
  }
});

function renderVerdict(el, r, label) {
  el.className = 'verdict ' + (r.watermarked ? 'bad' : 'good');
  el.innerHTML =
    `<strong>${r.watermarked ? 'WATERMARK DETECTED' : 'no watermark'}</strong>` +
    `<span>confidence ${r.confidence.toFixed(3)} · phase ${r.phase.toFixed(3)} ` +
    `(${r.set} set) · cvr ${r.cvr.toFixed(2)}</span>`;
}

function drawTo(canvas, img) {
  const maxW = 460;
  const scale = Math.min(1, maxW / img.naturalWidth);
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
}

// Separable Gaussian blur on one Float64 channel (radius small, fixed sigma).
function gaussBlur(src, w, h, radius) {
  const k = [];
  const sigma = radius / 2 || 1;
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k.push(v); sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  // horizontal
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -radius; i <= radius; i++) {
        const xx = Math.min(w - 1, Math.max(0, x + i));
        acc += src[y * w + xx] * k[i + radius];
      }
      tmp[y * w + x] = acc;
    }
  }
  // vertical
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -radius; i <= radius; i++) {
        const yy = Math.min(h - 1, Math.max(0, y + i));
        acc += tmp[yy * w + x] * k[i + radius];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

// Compute the carrier-scramble delta at 512px for one channel.
// Returns a Float64Array (512*512) = scrambled - original.
function carrierDelta(chan512, strength, rng) {
  const N = CB_SIZE;
  const re = Float64Array.from(chan512);
  const im = new Float64Array(N * N);
  fft2d(re, im, N, false);

  // canonical conjugate pairs: pick (a,b) with a>0 || (a==0 && b>0)
  const seen = new Set();
  for (const [a, b] of CARRIERS) {
    const [ca, cb] = (a > 0 || (a === 0 && b > 0)) ? [a, b] : [-a, -b];
    const key = ca + ',' + cb;
    if (seen.has(key)) continue;
    seen.add(key);
    const delta = (rng() * 2 - 1) * Math.PI * strength;
    applyPhase(re, im, N, ca, cb, delta);
    applyPhase(re, im, N, -ca, -cb, -delta); // keep conjugate symmetry -> real output
  }

  fft2d(re, im, N, true);
  const out = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) out[i] = re[i] - chan512[i];
  return out;
}

function applyPhase(re, im, N, r, c, delta) {
  const row = ((r % N) + N) % N;
  const col = ((c % N) + N) % N;
  const idx = row * N + col;
  const cos = Math.cos(delta), sin = Math.sin(delta);
  const nr = re[idx] * cos - im[idx] * sin;
  const ni = re[idx] * sin + im[idx] * cos;
  re[idx] = nr; im[idx] = ni;
}

// deterministic PRNG so results are reproducible
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function process(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const denoise = parseFloat($('denoise').value);
  const spectral = parseFloat($('spectral').value);

  // full-res pixels
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const cx = cv.getContext('2d');
  cx.drawImage(img, 0, 0);
  const id = cx.getImageData(0, 0, w, h);
  const px = id.data;

  // split channels (float)
  const ch = [new Float64Array(w * h), new Float64Array(w * h), new Float64Array(w * h)];
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    ch[0][i] = px[p]; ch[1][i] = px[p + 1]; ch[2][i] = px[p + 2];
  }

  // Stage 1 — denoise: attenuate the watermark-carrying high-freq residual
  if (denoise > 0) {
    const radius = 2;
    for (let c = 0; c < 3; c++) {
      const base = gaussBlur(ch[c], w, h, radius);
      for (let i = 0; i < w * h; i++) {
        const residual = ch[c][i] - base[i];
        ch[c][i] = base[i] + (1 - denoise) * residual; // shrink residual by `denoise`
      }
    }
  }

  // Stage 2 — carrier scramble at 512px, delta upscaled to full res
  if (spectral > 0) {
    const small = downscaleChannels(ch, w, h, CB_SIZE);
    const rng = mulberry32(12345);
    for (let c = 0; c < 3; c++) {
      const delta = carrierDelta(small[c], spectral, rng);
      const deltaFull = upscale(delta, CB_SIZE, CB_SIZE, w, h);
      for (let i = 0; i < w * h; i++) ch[c][i] += deltaFull[i];
    }
  }

  // write back + PSNR
  let mse = 0;
  const out = new ImageData(w, h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    for (let c = 0; c < 3; c++) {
      const v = Math.min(255, Math.max(0, Math.round(ch[c][i])));
      out.data[p + c] = v;
      const d = v - px[p + c];
      mse += d * d;
    }
    out.data[p + 3] = 255;
  }
  mse /= (w * h * 3);
  const psnr = mse === 0 ? Infinity : 10 * Math.log10(255 * 255 / mse);

  // show + offer download
  const afterCv = $('after');
  drawTo(afterCv, img); // size it
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').putImageData(out, 0, 0);
  afterCv.getContext('2d').drawImage(tmp, 0, 0, afterCv.width, afterCv.height);
  processedCanvas = tmp; // full-res result for detector + download

  $('psnr').textContent = `· PSNR ${isFinite(psnr) ? psnr.toFixed(1) + ' dB' : '∞'}`;
  $('status').textContent = `Done (${w}×${h}, denoise ${denoise.toFixed(2)}, scramble ${spectral.toFixed(2)}).`;
  tmp.toBlob((blob) => {
    const dl = $('download');
    dl.href = URL.createObjectURL(blob);
    dl.classList.remove('hidden');
  }, 'image/png');

  // auto-refresh detector verdicts on both panes
  renderVerdict($('verdictBefore'), detect(imgEl), 'Original');
  renderVerdict($('verdictAfter'), detect(tmp), 'Processed');
}

// Downscale all 3 channels to N×N using a canvas (bilinear).
function downscaleChannels(ch, w, h, N) {
  const src = document.createElement('canvas');
  src.width = w; src.height = h;
  const sctx = src.getContext('2d');
  const id = sctx.createImageData(w, h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    id.data[p] = ch[0][i]; id.data[p + 1] = ch[1][i]; id.data[p + 2] = ch[2][i]; id.data[p + 3] = 255;
  }
  sctx.putImageData(id, 0, 0);
  const dst = document.createElement('canvas');
  dst.width = N; dst.height = N;
  const dctx = dst.getContext('2d');
  dctx.drawImage(src, 0, 0, N, N);
  const d = dctx.getImageData(0, 0, N, N).data;
  const out = [new Float64Array(N * N), new Float64Array(N * N), new Float64Array(N * N)];
  for (let i = 0, p = 0; i < N * N; i++, p += 4) {
    out[0][i] = d[p]; out[1][i] = d[p + 1]; out[2][i] = d[p + 2];
  }
  return out;
}

// Upscale a single-channel float field from sw×sh to dw×dh via canvas.
function upscale(field, sw, sh, dw, dh) {
  // shift into [0,255] range around 128 to survive 8-bit canvas, then undo
  const src = document.createElement('canvas');
  src.width = sw; src.height = sh;
  const sctx = src.getContext('2d');
  const id = sctx.createImageData(sw, sh);
  for (let i = 0, p = 0; i < sw * sh; i++, p += 4) {
    const v = Math.min(255, Math.max(0, 128 + field[i]));
    id.data[p] = id.data[p + 1] = id.data[p + 2] = v; id.data[p + 3] = 255;
  }
  sctx.putImageData(id, 0, 0);
  const dst = document.createElement('canvas');
  dst.width = dw; dst.height = dh;
  const dctx = dst.getContext('2d');
  dctx.imageSmoothingEnabled = true;
  dctx.drawImage(src, 0, 0, dw, dh);
  const d = dctx.getImageData(0, 0, dw, dh).data;
  const out = new Float64Array(dw * dh);
  for (let i = 0, p = 0; i < dw * dh; i++, p += 4) out[i] = d[p] - 128;
  return out;
}
