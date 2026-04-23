#!/usr/bin/env python3
"""
Close the manual-validation loop for reverse-SynthID V4.

Reads the ``manifest.csv`` from ``dissolve_batch.py`` plus a ``tally.csv``
you filled by hand after checking each variant in the Gemini app. Updates
``carrier_weights`` in the V4 codebook in place:

- Bins that the **failed** variants (``still_watermarked=y``) tried to subtract
  get their weights **bumped up**, so subsequent dissolves attack those bins
  harder.
- Bins that the **succeeded** variants (``still_watermarked=n``) already
  subtracted get their weights **damped slightly**, to recover fidelity
  without giving up detector immunity.

The tally CSV accepts ``y``/``n``/``yes``/``no``/``1``/``0`` (case-insensitive)
in ``still_watermarked``. Rows with a blank value are ignored.

Usage::

    python scripts/calibrate_from_feedback.py \\
        --manifest runs/round_01/manifest.csv \\
        --tally    runs/round_01/tally.csv \\
        --codebook artifacts/spectral_codebook_v4.npz \\
        --step 0.25

The codebook is rewritten in place; a timestamped backup is made next to it
unless ``--no-backup`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "extraction"))

import numpy as np  # noqa: E402

from synthid_bypass_v4 import SpectralCodebookV4  # noqa: E402


TRUE_TOKENS = {"y", "yes", "1", "true", "t"}
FALSE_TOKENS = {"n", "no", "0", "false", "f"}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _read_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_still_watermarked(value: str) -> Optional[bool]:
    """``y/n`` → ``True/False``; empty/unknown → ``None``."""
    if value is None:
        return None
    v = value.strip().lower()
    if v == "":
        return None
    if v in TRUE_TOKENS:
        return True
    if v in FALSE_TOKENS:
        return False
    return None


def load_feedback(
    manifest_path: str, tally_path: str,
) -> List[Dict]:
    """Join manifest + tally on ``(source, variant)``; return labelled rows.

    Only rows whose tally has a parseable ``still_watermarked`` are returned.
    """
    manifest = _read_csv_dicts(manifest_path)

    # Tally may be the same file as the manifest (user filled in place) or a
    # separate file with at least (source, variant, still_watermarked).
    tally_raw = _read_csv_dicts(tally_path)
    tally: Dict[Tuple[str, str], bool] = {}
    for row in tally_raw:
        still = _parse_still_watermarked(row.get("still_watermarked", ""))
        if still is None:
            continue
        key = (row["source"], row["variant"])
        tally[key] = still

    joined: List[Dict] = []
    for row in manifest:
        key = (row["source"], row["variant"])
        if key not in tally:
            continue
        merged = dict(row)
        merged["still_watermarked"] = tally[key]
        joined.append(merged)
    return joined


# ---------------------------------------------------------------------------
# Calibration logic
# ---------------------------------------------------------------------------

def _parse_profile_key(profile_key: str) -> Optional[Tuple[str, int, int]]:
    """Parse ``'model_name/HxW'`` → ``(model, H, W)``."""
    if not profile_key or "/" not in profile_key:
        return None
    model, res = profile_key.rsplit("/", 1)
    if "x" not in res:
        return None
    try:
        h, w = (int(p) for p in res.lower().split("x"))
    except ValueError:
        return None
    return (model, h, w)


def calibrate(
    codebook: SpectralCodebookV4,
    feedback: List[Dict],
    step: float,
    damp_factor: float,
    consensus_floor: float,
    verbose: bool,
) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    """Update ``carrier_weights`` in-place. Returns per-profile summary stats.

    The update rule, per profile ``P``:

    Let ``F`` = number of feedback rows against ``P`` with
    ``still_watermarked=True`` (failed dissolves).
    Let ``S`` = number with ``still_watermarked=False`` (cleared dissolves).

    If ``F > 0``: scale ``carrier_weights`` by ``1 + step * (F / (F + S))``
    but only on bins with ``consensus_coherence >= consensus_floor``. Non-
    carrier bins are never touched — we don't want to amplify noise.

    If ``F == 0 and S > 0``: scale ``carrier_weights`` by
    ``1 - damp_factor * step`` on carrier bins (gentle fidelity recovery
    once we're clearing the detector).
    """
    groups: Dict[Tuple[str, int, int], Dict[str, List[Dict]]] = {}
    for row in feedback:
        pkey = _parse_profile_key(row.get("profile_key", ""))
        if pkey is None:
            continue
        bucket = groups.setdefault(pkey, {"fail": [], "pass": []})
        target = "fail" if row["still_watermarked"] else "pass"
        bucket[target].append(row)

    summary: Dict[Tuple[str, int, int], Dict[str, float]] = {}

    for pkey, bucket in groups.items():
        if pkey not in codebook.profiles:
            if verbose:
                print(f"  skip {pkey}: no matching profile in codebook")
            continue
        prof = codebook.profiles[pkey]
        F = len(bucket["fail"])
        S = len(bucket["pass"])

        carrier_mask = (prof.consensus_coherence >= consensus_floor).astype(np.float32)

        if F > 0:
            fail_ratio = F / max(F + S, 1)
            scale = 1.0 + step * fail_ratio
            delta = 1.0 + (scale - 1.0) * carrier_mask
            action = f"bump ×{scale:.3f}"
        elif S > 0:
            scale = max(1.0 - damp_factor * step, 0.2)
            delta = 1.0 + (scale - 1.0) * carrier_mask
            action = f"damp ×{scale:.3f}"
        else:
            continue

        before_mean = float(np.mean(prof.carrier_weights[..., 1]))
        codebook.update_carrier_weights(pkey, delta)
        after_mean = float(np.mean(prof.carrier_weights[..., 1]))

        summary[pkey] = {
            "fail": F,
            "pass": S,
            "before_mean_g": before_mean,
            "after_mean_g": after_mean,
            "action": action,
        }
        if verbose:
            print(f"  {pkey[0]}/{pkey[1]}x{pkey[2]}: {action}  "
                  f"fail={F} pass={S}  "
                  f"mean(G) {before_mean:.4f} → {after_mean:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    manifest_path: str,
    tally_path: str,
    codebook_path: str,
    step: float,
    damp_factor: float,
    consensus_floor: float,
    backup: bool,
) -> None:
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not os.path.isfile(tally_path):
        raise FileNotFoundError(f"Tally not found: {tally_path}")
    if not os.path.isfile(codebook_path):
        raise FileNotFoundError(f"Codebook not found: {codebook_path}")

    feedback = load_feedback(manifest_path, tally_path)
    if not feedback:
        print("No usable feedback rows (empty still_watermarked?). Nothing "
              "to do.")
        return

    print(f"Loaded {len(feedback)} labelled rows from tally.")

    codebook = SpectralCodebookV4()
    codebook.load(codebook_path)

    if backup:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = codebook_path + f".bak-{ts}.npz"
        shutil.copyfile(codebook_path, backup_path)
        print(f"Backup → {backup_path}")

    summary = calibrate(
        codebook=codebook,
        feedback=feedback,
        step=step,
        damp_factor=damp_factor,
        consensus_floor=consensus_floor,
        verbose=True,
    )

    if not summary:
        print("No profiles updated.")
        return

    codebook.save(codebook_path)

    n_fail = sum(s["fail"] for s in summary.values())
    n_pass = sum(s["pass"] for s in summary.values())
    print(f"\nCalibration complete. Profiles updated: {len(summary)}")
    print(f"Feedback: {n_pass} cleared / {n_fail} still watermarked "
          f"({n_pass * 100.0 / max(n_pass + n_fail, 1):.1f}% success).")
    if n_fail > 0:
        print("Next: re-run dissolve_batch.py on a fresh batch; weights "
              "are now stronger at persistent carriers.")
    else:
        print("100% cleared — consider lowering strength for better "
              "fidelity on the next batch.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Update V4 carrier_weights from manual Gemini detection tallies."
        ),
    )
    p.add_argument("--manifest", required=True,
                   help="Path to manifest.csv produced by dissolve_batch.py.")
    p.add_argument("--tally", required=True,
                   help=(
                       "Path to tally.csv with (source, variant, "
                       "still_watermarked) columns. May be the manifest file "
                       "itself if you filled it in place."
                   ))
    p.add_argument("--codebook", required=True,
                   help="V4 codebook .npz to update (in place).")
    p.add_argument("--step", type=float, default=0.25,
                   help="Base scale step; 0.25 = up to +25%% per round.")
    p.add_argument("--damp-factor", type=float, default=0.15,
                   help="Damping multiplier applied when all variants "
                        "cleared (fidelity recovery).")
    p.add_argument("--consensus-floor", type=float, default=0.50,
                   help="Only update bins with consensus_coherence >= this.")
    p.add_argument("--no-backup", dest="backup", action="store_false",
                   help="Skip the timestamped backup of the codebook.")
    p.set_defaults(backup=True)
    args = p.parse_args()

    run(
        manifest_path=args.manifest,
        tally_path=args.tally,
        codebook_path=args.codebook,
        step=args.step,
        damp_factor=args.damp_factor,
        consensus_floor=args.consensus_floor,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()
