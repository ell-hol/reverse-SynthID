#!/usr/bin/env python3
"""
Phase-2 driver for the reverse-SynthID V4 manual-validation loop.

Takes an input folder of watermarked images and emits one or more strength
variants per image (``A``, ``B``, ``C``, ... by default). Writes a
``manifest.csv`` that pairs each variant with:

- source image path
- output path
- strength preset
- profile key used
- PSNR / SSIM achieved

You then paste the variants into the Gemini app, run SynthID detection, and
fill in a small ``tally.csv`` (columns: ``source,variant,still_watermarked``,
values ``y/n``). Feed both files into ``calibrate_from_feedback.py`` to
update the codebook's per-carrier weights and iterate.

Usage::

    python scripts/dissolve_batch.py \\
        --input /path/to/input_images \\
        --output /path/to/out_dir \\
        --codebook artifacts/spectral_codebook_v4.npz \\
        --model nano-banana-pro-preview \\
        --strengths gentle moderate aggressive

Strengths map to filesystem-safe single-letter variants (A,B,C,D) in
manifest order, which makes the tally CSV trivial to fill by hand.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from typing import List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "extraction"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from synthid_bypass_v4 import SpectralCodebookV4, SynthIDBypassV4  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_STRENGTHS = ("gentle", "moderate", "aggressive")
VARIANT_LETTERS = "ABCDEFGH"


def iter_input_images(input_path: str) -> List[str]:
    """Resolve ``--input`` (file, directory, or glob) to a sorted list."""
    if os.path.isdir(input_path):
        out: List[str] = []
        for ext in IMAGE_EXTS:
            out.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            out.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(set(out))
    if os.path.isfile(input_path):
        return [input_path]
    # Treat as a glob pattern.
    return sorted(glob.glob(input_path))


def dissolve_one(
    bypass: SynthIDBypassV4,
    codebook: SpectralCodebookV4,
    src: str,
    out_dir: str,
    variant_letter: str,
    strength: str,
    model: Optional[str],
) -> dict:
    """Dissolve one image at one strength; return a manifest row."""
    base = os.path.splitext(os.path.basename(src))[0]
    out_name = f"{base}__{variant_letter}_{strength}.png"
    out_path = os.path.join(out_dir, out_name)

    t0 = time.time()
    try:
        result = bypass.bypass_v4_file(
            src, out_path, codebook,
            strength=strength, model=model, verify=False,
        )
        row = {
            "source": os.path.abspath(src),
            "variant": variant_letter,
            "strength": strength,
            "output": os.path.abspath(out_path),
            "profile_key": result.details["profile_key"],
            "exact_match": int(bool(result.details["exact_match"])),
            "psnr": round(result.psnr, 3),
            "ssim": round(result.ssim, 5),
            "n_passes_applied": result.details["n_passes_applied"],
            "n_passes_rolled_back": result.details["n_passes_rolled_back"],
            "elapsed_sec": round(time.time() - t0, 3),
            "still_watermarked": "",  # filled by you during validation
            "notes": "",
        }
    except Exception as e:
        row = {
            "source": os.path.abspath(src),
            "variant": variant_letter,
            "strength": strength,
            "output": "",
            "profile_key": "",
            "exact_match": 0,
            "psnr": "",
            "ssim": "",
            "n_passes_applied": 0,
            "n_passes_rolled_back": 0,
            "elapsed_sec": round(time.time() - t0, 3),
            "still_watermarked": "",
            "notes": f"ERROR: {e}",
        }
    return row


def run(
    input_path: str,
    out_dir: str,
    codebook_path: str,
    strengths: List[str],
    model: Optional[str] = None,
    limit: Optional[int] = None,
    manifest_name: str = "manifest.csv",
) -> str:
    sources = iter_input_images(input_path)
    if limit is not None:
        sources = sources[:limit]
    if not sources:
        print(f"No images found in {input_path}")
        sys.exit(2)

    os.makedirs(out_dir, exist_ok=True)

    codebook = SpectralCodebookV4()
    codebook.load(codebook_path)

    if model is not None and model not in codebook.models:
        print(f"WARNING: --model {model} not found in codebook. "
              f"Available: {codebook.models}. Proceeding anyway "
              "(best-effort fallback across models).")

    bypass = SynthIDBypassV4()

    if len(strengths) > len(VARIANT_LETTERS):
        raise ValueError(
            f"Too many strengths ({len(strengths)}); "
            f"max supported: {len(VARIANT_LETTERS)}"
        )
    letters = list(VARIANT_LETTERS[:len(strengths)])

    manifest_path = os.path.join(out_dir, manifest_name)
    fieldnames = [
        "source", "variant", "strength", "output", "profile_key",
        "exact_match", "psnr", "ssim",
        "n_passes_applied", "n_passes_rolled_back",
        "elapsed_sec", "still_watermarked", "notes",
    ]

    print(f"Dissolving {len(sources)} image(s) × {len(strengths)} variant(s) "
          f"→ {out_dir}")
    if model:
        print(f"Model hint: {model}")

    rows = []
    for i, src in enumerate(sources):
        print(f"[{i + 1}/{len(sources)}] {os.path.basename(src)}")
        for letter, strength in zip(letters, strengths):
            row = dissolve_one(
                bypass=bypass,
                codebook=codebook,
                src=src,
                out_dir=out_dir,
                variant_letter=letter,
                strength=strength,
                model=model,
            )
            rows.append(row)
            if row["notes"].startswith("ERROR"):
                print(f"    {letter}/{strength:12s} {row['notes']}")
            else:
                print(f"    {letter}/{strength:12s} "
                      f"psnr={row['psnr']:>6} ssim={row['ssim']:>7} "
                      f"profile={row['profile_key']} "
                      f"exact={row['exact_match']}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest: {manifest_path}")
    print("\nNext steps:")
    print("  1. Upload each ABS-path output to the Gemini app and run "
          "SynthID detection.")
    print("  2. For each row, fill the `still_watermarked` column with "
          "`y` or `n` (leave blank to skip).")
    print(f"  3. Save the filled file as tally.csv and run:")
    print(f"       python scripts/calibrate_from_feedback.py "
          f"--manifest {manifest_path} --tally <your_tally.csv> "
          f"--codebook {codebook_path}")
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Emit bypass variants for manual Gemini validation.",
    )
    p.add_argument("--input", required=True,
                   help="Path to an image, a directory, or a glob pattern.")
    p.add_argument("--output", required=True,
                   help="Directory to write variants and manifest.csv into.")
    p.add_argument("--codebook", required=True,
                   help="Path to the V4 codebook .npz.")
    p.add_argument("--strengths", nargs="+", default=list(DEFAULT_STRENGTHS),
                   choices=["gentle", "moderate", "aggressive", "maximum",
                            "demolish", "annihilate", "combo",
                            "blog_pure", "blog_plus", "blog_combo",
                            "residual_pure", "residual_plus", "residual_combo",
                            "regen_pure", "regen_plus", "regen_combo",
                            "final", "nuke"],
                   help=f"Strengths to emit (default: {DEFAULT_STRENGTHS}).")
    p.add_argument("--model", default=None,
                   help=(
                       "Optional model hint (e.g. nano-banana-pro-preview). "
                       "Omit to let the codebook auto-select by resolution."
                   ))
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many input images (for quick tests).")
    p.add_argument("--manifest-name", default="manifest.csv",
                   help="Manifest filename inside --output (default: manifest.csv).")
    args = p.parse_args()

    run(
        input_path=args.input,
        out_dir=args.output,
        codebook_path=args.codebook,
        strengths=args.strengths,
        model=args.model,
        limit=args.limit,
        manifest_name=args.manifest_name,
    )


if __name__ == "__main__":
    main()
