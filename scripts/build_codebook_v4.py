#!/usr/bin/env python3
"""
Build the reverse-SynthID V4 codebook from a hierarchical dataset.

Expected layout::

    <root>/
        <model>/
            black/     HxW/*.png
            white/     HxW/*.png
            blue/      HxW/*.png
            green/     HxW/*.png
            red/       HxW/*.png
            gray/      HxW/*.png
            gradient/  HxW/*.png
            diverse/   HxW/*.png

The script produces one ``ProfileV4`` per ``(model, H, W)`` that has at least
``--min-consensus-colors`` consensus colours (``black``, ``white``, ``blue``,
``green``, ``red``, ``gray``) with enough reference images. ``gradient/`` and
``diverse/`` are used as content-baseline only, never as carrier sources.

Usage::

    python scripts/build_codebook_v4.py \\
        --root /Users/aoxo/vscode/reverse-synthid-data \\
        --output artifacts/spectral_codebook_v4.npz

    # Restrict to a single model:
    python scripts/build_codebook_v4.py --root <root> --models nano-banana-pro-preview

    # Also emit a 'union' pseudo-model that averages profiles across models:
    python scripts/build_codebook_v4.py --root <root> --add-union
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "extraction"))

from synthid_bypass_v4 import (  # noqa: E402
    ALL_COLORS,
    SpectralCodebookV4,
)


DEFAULT_DATASET_ROOT = "/Users/aoxo/vscode/reverse-synthid-data"
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "artifacts", "spectral_codebook_v4.npz")


def build(
    root: str,
    output: str,
    models: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    min_refs_per_color: int = 3,
    min_consensus_colors: int = 3,
    max_per_bucket: Optional[int] = None,
    add_union: bool = False,
) -> None:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    codebook = SpectralCodebookV4()
    codebook._bind_root(root)  # type: ignore[attr-defined]
    codebook.build_from_hierarchical_dataset(
        root=root,
        models=models,
        colors=colors,
        min_refs_per_color=min_refs_per_color,
        min_consensus_colors=min_consensus_colors,
        max_per_bucket=max_per_bucket,
        verbose=True,
    )

    if not codebook.profiles:
        print("\nNo profiles built. Check that --root points at a directory "
              "containing <model>/<color>/<HxW>/*.png")
        sys.exit(2)

    if add_union:
        codebook.add_union_profiles(verbose=True)

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".",
                exist_ok=True)
    codebook.save(output)

    print("\nProfiles:")
    for key in sorted(codebook.profiles):
        model, h, w = key
        prof = codebook.profiles[key]
        refs = ", ".join(
            f"{c}={n}" for c, n in sorted(prof.n_refs_per_color.items())
        )
        print(f"  {model}/{h}x{w}: {refs} (content={prof.n_content_refs})")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the reverse-SynthID V4 codebook.",
    )
    p.add_argument(
        "--root", default=DEFAULT_DATASET_ROOT,
        help=(
            "Hierarchical dataset root (default: "
            f"{DEFAULT_DATASET_ROOT}). Should contain <model>/<color>/<HxW>/*."
        ),
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output .npz path (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--models", nargs="*", default=None,
        help="Restrict to these model subdirectories (default: auto-detect).",
    )
    p.add_argument(
        "--colors", nargs="*", default=None, choices=list(ALL_COLORS),
        help="Colours to include (default: all known).",
    )
    p.add_argument(
        "--min-refs-per-color", type=int, default=3,
        help="Drop (color, resolution) buckets with fewer images than this.",
    )
    p.add_argument(
        "--min-consensus-colors", type=int, default=3,
        help=(
            "Require at least this many consensus colours per (model, HxW) "
            "or the profile is skipped."
        ),
    )
    p.add_argument(
        "--max-per-bucket", type=int, default=None,
        help="Cap images per (color, resolution) bucket (default: unlimited).",
    )
    p.add_argument(
        "--add-union", action="store_true",
        help="Also emit a 'union' pseudo-model averaging across real models.",
    )
    args = p.parse_args()

    build(
        root=args.root,
        output=args.output,
        models=args.models,
        colors=args.colors,
        min_refs_per_color=args.min_refs_per_color,
        min_consensus_colors=args.min_consensus_colors,
        max_per_bucket=args.max_per_bucket,
        add_union=args.add_union,
    )


if __name__ == "__main__":
    main()
