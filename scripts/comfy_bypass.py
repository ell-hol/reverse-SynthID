#!/usr/bin/env python3
"""
comfy_bypass.py — a CPU, no-diffusion re-implementation of the core logic of the
ComfyUI workflow in 00quebec/Synthid-Bypass (Synthid-Bypass-v2.0.json).

The original workflow performs a *low-denoise, structure-locked redraw*: a Qwen
Image diffusion model regenerates the image at a small denoise strength while a
Canny controlnet pins the composition. The hypothesis (from that repo's README)
is that SynthID lives in low-level image noise, so regenerating that noise layer
from scratch — without touching composition — discards the watermark-carrying
pixels.

We can't run the diffusion redraw here (no GPU / model weights), so we reproduce
its *intent* with signal processing:

  1. Adaptive denoise strength from resolution  (Synthid-Bypass-AdaptiveDenoise
     node: ref ~6 MP, clamp [0.08, 0.15]).
  2. Structure lock                              (edge-preserving base = "Canny
     controlnet": low/mid frequencies + edges are kept verbatim).
  3. Noise-layer "redraw"                        (the high-frequency residual is
     regenerated: we keep its power spectrum but RANDOMIZE ITS PHASE, so grain
     statistics are preserved while the watermark's phase-encoded carrier — the
     thing the detector measures — is destroyed). Strength = the adaptive
     denoise value, matching the KSampler denoise≈0.10.
  4. Optional mild elastic warp                  (sub-pixel; breaks any residual
     spatial phase consensus, off by default to maximize fidelity).

This is an approximation, not the diffusion pipeline. It targets the phase-based
detector in this repo. Verify with robust_extractor.py before/after.
"""

import argparse
import numpy as np
import cv2
from numpy.fft import fft2, ifft2


# --- Stage 1: AdaptiveDenoise node (widgets [6, 0.08, 0.15]) -----------------
def adaptive_denoise(h, w, ref_mp=6.0, dmin=0.08, dmax=0.15):
    """Map image resolution to a denoise strength, like the workflow's node.

    Larger images -> closer to dmax (more regeneration headroom); smaller ->
    closer to dmin. Reference point ~6 megapixels, clamped to [dmin, dmax].
    """
    mp = (h * w) / 1e6
    # smooth interpolation by megapixel ratio, clamped
    t = np.clip(mp / ref_mp, 0.0, 1.0)
    return float(dmin + (dmax - dmin) * t)


# --- Stage 2: structure lock (the "Canny controlnet") ------------------------
def structure_base(channel, d=9, sigma_color=0.08, sigma_space=7):
    """Edge-preserving smooth of one float[0,1] channel.

    Keeps composition and edges (what the controlnet locks); everything removed
    here is the high-frequency layer we will regenerate.
    """
    return cv2.bilateralFilter(
        channel.astype(np.float32), d, sigma_color, sigma_space
    )


# --- Stage 3: phase-randomized noise "redraw" --------------------------------
def redraw_noise(residual, strength, rng):
    """Regenerate the residual: keep its power spectrum, scramble its phase.

    A diffusion redraw produces fresh fine detail with the same look but no
    correlation to the original watermark. Phase randomization is the linear
    analogue: |F| (grain energy / spectrum) is preserved, arg(F) is replaced by
    uniform random phase, so the watermark's phase-encoded carriers vanish.

    `strength` blends original->regenerated residual (the KSampler denoise).
    """
    F = fft2(residual)
    mag = np.abs(F)
    random_phase = rng.uniform(-np.pi, np.pi, size=residual.shape)
    # Keep the DC term real/unchanged to avoid a global brightness shift.
    random_phase[0, 0] = np.angle(F[0, 0])
    F_new = mag * np.exp(1j * random_phase)
    regenerated = np.real(ifft2(F_new))
    return (1.0 - strength) * residual + strength * regenerated


# --- Stage 4 (optional): mild elastic warp -----------------------------------
def elastic_warp(img, rng, amplitude=0.6, smooth=51):
    """Sub-pixel low-frequency warp to fragment residual spatial phase consensus."""
    h, w = img.shape[:2]
    dx = rng.standard_normal((h, w)).astype(np.float32)
    dy = rng.standard_normal((h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (smooth, smooth), 0) * amplitude
    dy = cv2.GaussianBlur(dy, (smooth, smooth), 0) * amplitude
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --- Stage 3b (optional): targeted carrier-phase perturbation ----------------
def spectral_carrier_scramble(image_bgr, codebook_path, strength, rng):
    """Perturb full-image FFT phase at the documented SynthID carrier bins.

    The repo's detector measures phase agreement at very-low-frequency carrier
    bins (within ~+-7 of DC at 512px). A diffusion redraw resamples these while a
    controlnet re-imposes composition; with no diffusion we instead add a bounded
    random phase delta at exactly those bins (and their conjugates, to keep the
    output real). Because the bins are low-frequency, the induced change is a
    smooth, low-amplitude, large-scale ripple -> high PSNR, but it breaks the
    phase match the detector relies on.

    `strength` in [0,1] scales the max phase delta (* pi).
    """
    import pickle
    cb = pickle.load(open(codebook_path, "rb"))
    refs = cb.get("carrier_refs", {})
    size = int(cb.get("image_size", 512))
    carriers = []
    for key in ("dark_carriers", "white_carriers"):
        c = refs.get(key)
        if c is not None:
            carriers.extend([tuple(map(int, p)) for p in np.asarray(c)])
    carriers = sorted(set(carriers))

    out = image_bgr.astype(np.float32).copy()
    for ch in range(3):
        F = np.fft.fftshift(fft2(out[:, :, ch]))
        h, w = F.shape
        cy, cx = h // 2, w // 2
        for (fy, fx) in carriers:
            for sy, sx in ((fy, fx), (-fy, -fx)):  # conjugate pair -> real output
                y, x = cy + sy, cx + sx
                if 0 <= y < h and 0 <= x < w:
                    delta = rng.uniform(-np.pi, np.pi) * strength
                    F[y, x] *= np.exp(1j * delta)
        out[:, :, ch] = np.real(ifft2(np.fft.ifftshift(F)))
    return np.clip(out, 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def bypass(image_bgr, warp=False, seed=0, codebook=None, spectral=0.0):
    rng = np.random.RandomState(seed)
    h, w = image_bgr.shape[:2]
    d = adaptive_denoise(h, w)

    img = image_bgr.astype(np.float32) / 255.0
    out = np.empty_like(img)
    for c in range(3):
        ch = img[:, :, c]
        base = structure_base(ch)                 # Canny lock
        residual = ch - base                       # watermark-carrying layer
        new_res = redraw_noise(residual, d, rng)   # low-denoise redraw
        out[:, :, c] = base + new_res

    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).round().astype(np.uint8)

    if spectral > 0.0 and codebook:
        out = spectral_carrier_scramble(out, codebook, spectral, rng)

    if warp:
        out = elastic_warp(out, rng)

    return out, d


def main():
    ap = argparse.ArgumentParser(description="ComfyUI-flow-inspired SynthID watermark bypass (CPU)")
    ap.add_argument("input", help="input image")
    ap.add_argument("output", help="output image")
    ap.add_argument("--warp", action="store_true", help="add mild elastic warp stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codebook", default=None, help="codebook .pkl for --spectral stage")
    ap.add_argument("--spectral", type=float, default=0.0,
                    help="carrier-phase scramble strength [0..1] (needs --codebook)")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Could not read {args.input}")

    out, d = bypass(img, warp=args.warp, seed=args.seed,
                    codebook=args.codebook, spectral=args.spectral)
    cv2.imwrite(args.output, out)

    print(f"  input        : {args.input} ({img.shape[1]}x{img.shape[0]})")
    print(f"  adaptive d   : {d:.3f}  (workflow clamp [0.08, 0.15])")
    print(f"  spectral     : {args.spectral}")
    print(f"  elastic warp : {args.warp}")
    print(f"  PSNR vs input: {psnr(img, out):.2f} dB")
    print(f"  output       : {args.output}")


if __name__ == "__main__":
    main()
