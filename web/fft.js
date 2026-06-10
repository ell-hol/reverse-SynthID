// Minimal iterative radix-2 FFT (in-place, complex) + 2D transform helpers.
// Real/imag stored as separate Float64Arrays. Lengths must be powers of two.

function fft1d(re, im, inverse) {
  const n = re.length;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k], aIm = im[i + k];
        const bRe = re[i + k + len / 2], bIm = im[i + k + len / 2];
        const tRe = bRe * curRe - bIm * curIm;
        const tIm = bRe * curIm + bIm * curRe;
        re[i + k] = aRe + tRe;
        im[i + k] = aIm + tIm;
        re[i + k + len / 2] = aRe - tRe;
        im[i + k + len / 2] = aIm - tIm;
        const nRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nRe;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
}

// In-place 2D FFT on row-major arrays of size n*n.
function fft2d(re, im, n, inverse) {
  const rowRe = new Float64Array(n), rowIm = new Float64Array(n);
  // rows
  for (let y = 0; y < n; y++) {
    const off = y * n;
    for (let x = 0; x < n; x++) { rowRe[x] = re[off + x]; rowIm[x] = im[off + x]; }
    fft1d(rowRe, rowIm, inverse);
    for (let x = 0; x < n; x++) { re[off + x] = rowRe[x]; im[off + x] = rowIm[x]; }
  }
  // columns
  const colRe = new Float64Array(n), colIm = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { colRe[y] = re[y * n + x]; colIm[y] = im[y * n + x]; }
    fft1d(colRe, colIm, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = colRe[y]; im[y * n + x] = colIm[y]; }
  }
}

function nextPow2(v) { let p = 1; while (p < v) p <<= 1; return p; }
