// Client-side port of robust_extractor.detect_array (phase-dominant scoring).
//
// Reference phases and carrier bins are taken verbatim from
// artifacts/codebook/robust_codebook.pkl (image_size = 512).
//
// The Python detector's confidence is 0.80*phase_score + 0.20*cvr_score.
// We replicate the phase term exactly (resize -> grayscale FFT -> phase match
// at carrier bins vs reference, best of the dark/white sets). The cvr term is
// approximated with a Gaussian-residual noise carrier ratio (the Python uses a
// fancier fused denoiser). Verdict threshold (confidence > 0.50) matches.
const DET = {
  size: 512,
  dark_carriers: [[-5,-3],[5,3],[-5,3],[5,-3],[-3,-4],[3,4],[-3,4],[3,-4],[-4,-3],[4,3],[-4,3],[4,-3],[-5,-1],[5,1],[-5,1],[5,-1],[-5,-2],[5,2],[-5,2],[5,-2],[-2,-5],[2,5],[-2,5],[2,-5],[-1,-5],[1,5],[-1,5],[1,-5],[-4,-4],[4,4],[-4,4],[4,-4],[-1,-6],[1,6],[-3,-5],[3,5]],
  dark_ref: [-2.797338,2.797338,-0.70364,0.70364,-2.445601,2.445601,0.34576,-0.34576,-2.44792,2.44792,-0.355335,0.355335,-2.100774,2.100774,-1.405538,1.405538,-2.453837,2.453837,-1.053632,1.053632,-2.448291,2.448291,1.05578,-1.05578,-2.101113,2.101113,1.403686,-1.403686,-2.795057,2.795057,-0.001543,0.001543,-2.447903,2.447903,-2.792331,2.792331],
  white_carriers: [[0,-7],[0,7],[0,-8],[0,8],[0,-9],[0,9],[0,-10],[0,10],[0,-11],[0,11],[0,-12],[0,12],[0,-20],[0,20],[0,-21],[0,21],[0,-22],[0,22],[0,-23],[0,23]],
  white_ref: [-3.033965,3.033965,-2.869006,2.869006,-2.855589,2.855589,-2.896355,2.896355,-2.794533,2.794533,-2.735395,2.735395,-2.579497,2.579497,-2.615955,2.615955,-2.570547,2.570547,-2.480709,2.480709],
};

function wrapAngle(a) { // -> [-pi, pi]
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

// Resize an image element to N×N grayscale Float64Array.
function toGray512(img, N) {
  const c = document.createElement('canvas');
  c.width = N; c.height = N;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, N, N);
  const d = ctx.getImageData(0, 0, N, N).data;
  const g = new Float64Array(N * N);
  for (let i = 0, p = 0; i < N * N; i++, p += 4) {
    g[i] = (d[p] + d[p + 1] + d[p + 2]) / 3; // matches np.mean over channels
  }
  return g;
}

function phaseAt(re, im, N, fy, fx) {
  const row = ((fy % N) + N) % N;
  const col = ((fx % N) + N) % N;
  const idx = row * N + col;
  return Math.atan2(im[idx], re[idx]);
}

function setMatch(re, im, N, carriers, ref) {
  let sum = 0, n = 0;
  for (let i = 0; i < carriers.length; i++) {
    const [fy, fx] = carriers[i];
    const ph = phaseAt(re, im, N, fy, fx);
    const diff = Math.abs(wrapAngle(ph - ref[i]));
    sum += 1 - diff / Math.PI;
    n++;
  }
  return n ? sum / n : 0;
}

// Approximate cvr_noise: carrier vs random magnitude in the residual spectrum.
function cvrNoise(gray, N) {
  // residual = gray - blur(gray)
  const base = gaussBlur(gray, N, N, 2);
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) re[i] = gray[i] - base[i];
  fft2d(re, im, N, false);
  const mag = (fy, fx) => {
    const row = ((fy % N) + N) % N, col = ((fx % N) + N) % N, idx = row * N + col;
    return Math.hypot(re[idx], im[idx]);
  };
  const carriers = DET.dark_carriers.concat(DET.white_carriers);
  let cs = 0;
  for (const [fy, fx] of carriers) cs += mag(fy, fx);
  cs /= carriers.length;
  // random bins away from DC
  let rng = mulberry32(42), rs = 0, rn = 0;
  const tries = carriers.length * 4;
  for (let k = 0; k < tries; k++) {
    const ry = 10 + Math.floor(rng() * (N - 20));
    const rx = 10 + Math.floor(rng() * (N - 20));
    if (Math.abs(ry) < 5 && Math.abs(rx) < 5) continue;
    rs += Math.hypot(re[ry * N + rx], im[ry * N + rx]); rn++;
  }
  rs = rn ? rs / rn : 1e-10;
  return cs / (rs + 1e-10);
}

// Run the detector on an image element. Returns {watermarked, confidence, phase, set}.
function detect(img) {
  const N = DET.size;
  const gray = toGray512(img, N);
  const re = Float64Array.from(gray), im = new Float64Array(N * N);
  fft2d(re, im, N, false);

  const dark = setMatch(re, im, N, DET.dark_carriers, DET.dark_ref);
  const white = setMatch(re, im, N, DET.white_carriers, DET.white_ref);
  const best = Math.max(dark, white);
  const bestSet = dark >= white ? 'dark' : 'white';

  const cvr = cvrNoise(gray, N);

  const phaseScore = 1 / (1 + Math.exp(-20 * (best - 0.78)));
  const cvrScore = 1 / (1 + Math.exp(-2 * (cvr - 2.0)));
  const confidence = Math.min(1, 0.80 * phaseScore + 0.20 * cvrScore);

  return {
    watermarked: confidence > 0.50,
    confidence,
    phase: best,
    set: bestSet,
    cvr,
  };
}
