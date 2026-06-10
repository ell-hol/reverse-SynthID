# SynthID Bypass — browser demo

A zero-backend, fully client-side web app that ports the `comfy_bypass.py`
pipeline (drawn from the ComfyUI *Synthid-Bypass* flow) to JavaScript, with the
repo's `robust_extractor` detector ported alongside it so you can check results
in the browser. **No image is ever uploaded** — all processing and detection run
locally in the browser via Canvas + an in-browser FFT.

## What it does

1. **Denoise** — attenuates the high-frequency residual that carries the
   watermark noise (spatial Gaussian separation).
2. **Carrier scramble** — at the detector's 512px scale, perturbs the FFT phase
   at the documented SynthID carrier bins (from
   `artifacts/codebook/robust_codebook.pkl`), then adds the resulting smooth
   low-frequency delta back onto the full-resolution image.
3. **Detector** — `detector.js` reproduces the phase-match scoring of
   `src/extraction/robust_extractor.py` exactly (verified: identical phase-match
   on `synthid_black.jpg` = 0.8719 and `fable.png` = 0.5970). The `cvr` noise
   term is approximated (the Python uses a heavier fused denoiser).

## Deploy to Vercel

This is a static site — no build step.

**Option A — dashboard:** import the repo, set **Root Directory** to `web`,
framework preset **Other**, leave build command empty.

**Option B — CLI:**

```bash
cd web
vercel
```

## Run locally

```bash
cd web
python3 -m http.server 8000   # then open http://localhost:8000
```

## Honesty / scope

- This targets a **reimplementation** of the detector, not Google's actual
  SynthID verifier. A clean result here is not proof an image is watermark-free.
- The carrier-scramble stage trades fidelity for efficacy on content-rich images
  (the watermark bins overlap real composition); the original ComfyUI flow uses
  a GPU diffusion redraw + Canny controlnet to avoid that, which this CPU/JS
  port cannot do.
- Research / AI-safety demo only.
