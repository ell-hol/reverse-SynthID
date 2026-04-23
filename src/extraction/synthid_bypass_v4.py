"""
SynthID Bypass V4 — Cross-Color Consensus Dissolver

Builds a multi-model, multi-color codebook keyed by ``(model, H, W)``. Each
profile aggregates per-color solid-background FFTs; the key idea is the
**cross-color phase consensus**:

    consensus(fy, fx, ch) = | mean_over_colors( exp(i * phase_color) ) |

A true SynthID carrier is image-content-independent, so its phase is identical
across pure-black / pure-white / pure-red / ... backgrounds and its consensus
magnitude is close to 1. Content-driven spectral energy (colour tint, gradients,
compression artefacts) is phase-scrambled across colours and drops out of the
consensus. This gives a much cleaner carrier mask than v3's 2-colour
``black_white_agreement``.

V4 also introduces a *live* ``carrier_weights`` array per profile: starts at
``consensus**2`` and is updated in-place by ``calibrate_from_feedback.py``
using manual detection tallies from the Gemini app. Subsequent bypass passes
read the calibrated weights, closing the loop.

Design principles:
- ``synthid_bypass.py`` is untouched. V4 is a thin layer that reuses primitives
  (FFT helpers, rfft packing, BypassResult) from v3 where possible.
- Per-model profiles, auto-selected at bypass time. A ``union`` pseudo-model is
  supported via the build script.
- Fidelity-first: luminance-safe DC, PSNR-floor rollback, per-bin cap.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthid_bypass import BypassResult, SpectralCodebook  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Colours contributing to the cross-colour phase consensus. ``gradient`` and
# ``diverse`` are intentionally excluded — they carry content bias.
CONSENSUS_COLORS: Tuple[str, ...] = (
    "black", "white", "blue", "green", "red", "gray",
)

# Content-only colours (not used in consensus; used as content_baseline only).
CONTENT_COLORS: Tuple[str, ...] = ("diverse", "gradient")

ALL_COLORS: Tuple[str, ...] = CONSENSUS_COLORS + CONTENT_COLORS

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
UNION_MODEL = "union"

# Channel weights (G strongest for SynthID, per v3 empirical finding).
CHANNEL_WEIGHTS = np.array([0.85, 1.0, 0.70])


# ---------------------------------------------------------------------------
# Universal absolute-bin carriers
# ---------------------------------------------------------------------------
#
# SynthID injects a spread-spectrum phase signal at fixed absolute FFT bin
# offsets from DC. The canonical list — first published in
# ``How to Reverse SynthID (legally)`` by Aloshdenny (2026) — was identified
# from a 150k image pair study and is reproduced here:
#
#   (±14,  ±14)   primary diagonal
#   (±98,  ±14)   tertiary
#   (±126, ±14)   secondary horizontal
#   (±128, ±128)  centre-frequency
#   (±210, ±14)   extended
#   (±238, ±14)   edge
#
# These positions were measured on a 1024×1024 reference grid. For non-square
# Gemini outputs we scale proportionally by each axis (the generator runs at
# 1024² internally and resamples, so the carriers move with the resample).
UNIVERSAL_CARRIER_BINS_1024: Tuple[Tuple[int, int], ...] = (
    (14, 14), (14, -14), (-14, 14), (-14, -14),
    (98, 14), (98, -14), (-98, 14), (-98, -14),
    (126, 14), (126, -14), (-126, 14), (-126, -14),
    (128, 128), (128, -128), (-128, 128), (-128, -128),
    (210, 14), (210, -14), (-210, 14), (-210, -14),
    (238, 14), (238, -14), (-238, 14), (-238, -14),
)
UNIVERSAL_REFERENCE_SHAPE: Tuple[int, int] = (1024, 1024)


def scale_bins_to_shape(
    bins: Tuple[Tuple[int, int], ...],
    shape: Tuple[int, int],
    ref_shape: Tuple[int, int] = UNIVERSAL_REFERENCE_SHAPE,
) -> List[Tuple[int, int]]:
    """Project reference-shape bin offsets onto an arbitrary ``(H, W)`` shape.

    Returns only bins that stay within the image's Nyquist envelope and are
    free of DC; de-duplicates after integer rounding.
    """
    H, W = shape
    rH, rW = ref_shape
    nyq_y = H // 2
    nyq_x = W // 2
    out: List[Tuple[int, int]] = []
    seen: set = set()
    for fy, fx in bins:
        sy = int(round(fy * H / rH))
        sx = int(round(fx * W / rW))
        if sy == 0 and sx == 0:
            continue
        if abs(sy) >= nyq_y or abs(sx) >= nyq_x:
            continue
        key = (sy, sx)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def harvest_codebook_carriers(
    profile: "ProfileV4",
    top_k: int = 96,
    consensus_floor: float = 0.55,
    centred: bool = True,
) -> List[Tuple[int, int]]:
    """Extract the top-K consensus bins from ``profile`` as signed offsets.

    Returns offsets in ``(-H/2, H/2)`` × ``(-W/2, W/2)`` centred coordinates,
    sorted by green-channel consensus-weighted magnitude descending.
    """
    pH, pW = profile.shape
    cons = profile.consensus_coherence[..., 1].copy()
    mag = profile.avg_wm_magnitude[..., 1].copy()
    cons[0, 0] = 0.0
    score = cons * np.log1p(mag)
    score[cons < consensus_floor] = 0.0
    flat = score.ravel()
    if flat.max() <= 0:
        return []
    idx = np.argpartition(flat, -top_k)[-top_k:]
    idx = idx[np.argsort(-flat[idx])]
    out: List[Tuple[int, int]] = []
    for i in idx:
        y, x = divmod(int(i), pW)
        if centred:
            if y > pH // 2:
                y -= pH
            if x > pW // 2:
                x -= pW
        if y == 0 and x == 0:
            continue
        out.append((y, x))
    return out


# ---------------------------------------------------------------------------
# Profile container
# ---------------------------------------------------------------------------

@dataclass
class ProfileV4:
    """One SpectralCodebookV4 profile keyed by ``(model, H, W)``.

    All spectral arrays are full-resolution ``(H, W, 3)``; compact rfft packing
    happens only at save time.
    """

    model: str
    shape: Tuple[int, int]
    color_names: List[str]
    per_color_magnitude: np.ndarray          # (n_colors, H, W, 3)
    per_color_phase: np.ndarray              # (n_colors, H, W, 3)
    per_color_consistency: np.ndarray        # (n_colors, H, W, 3)
    consensus_coherence: np.ndarray          # (H, W, 3)
    consensus_phase: np.ndarray              # (H, W, 3)  — mean unit-phase angle
    inverted_agreement: np.ndarray           # (H, W, 3)  — mean |cos(ph_c - ph_inv_c)|
    avg_wm_magnitude: np.ndarray             # (H, W, 3)
    content_baseline: np.ndarray             # (H, W, 3)  — from CONTENT_COLORS
    carrier_weights: np.ndarray              # (H, W, 3)  — live, calibrated
    n_refs_per_color: Dict[str, int]
    n_content_refs: int

    @property
    def H(self) -> int:
        return self.shape[0]

    @property
    def W(self) -> int:
        return self.shape[1]


# ---------------------------------------------------------------------------
# SpectralCodebookV4
# ---------------------------------------------------------------------------

class SpectralCodebookV4:
    """Multi-model, multi-colour SynthID codebook.

    Profiles are keyed by ``(model: str, H: int, W: int)``. Build from a
    hierarchical dataset layout::

        root/
            {model}/
                black/{HxW}/*.png
                white/{HxW}/*.png
                ...

    via :py:meth:`build_from_hierarchical_dataset`. Save/load uses a compact
    npz format (rfft half-spectrum, float16 / uint8 quantisation).
    """

    FORMAT_VERSION = 4
    _CONS_THRESHOLD = 0.15

    def __init__(self):
        self.profiles: Dict[Tuple[str, int, int], ProfileV4] = {}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def keys(self) -> List[Tuple[str, int, int]]:
        return list(self.profiles.keys())

    @property
    def models(self) -> List[str]:
        return sorted({k[0] for k in self.profiles})

    def resolutions_for(self, model: str) -> List[Tuple[int, int]]:
        return sorted({(h, w) for (m, h, w) in self.profiles if m == model})

    # ------------------------------------------------------------------
    # Profile selection
    # ------------------------------------------------------------------

    def get_profile(
        self,
        h: int,
        w: int,
        model: Optional[str] = None,
    ) -> Tuple[ProfileV4, Tuple[str, int, int], bool]:
        """Best-matching profile for ``(H, W)`` and optional ``model``.

        Returns ``(profile, (model, H, W), exact_match)``. Prefers an exact
        ``(model, H, W)`` match. Falls back to any model at same resolution,
        then to the closest aspect ratio within the requested model, then
        globally.
        """
        if not self.profiles:
            raise ValueError("Codebook has no profiles")

        if model is not None and (model, h, w) in self.profiles:
            return self.profiles[(model, h, w)], (model, h, w), True

        if model is None:
            for (m, kh, kw), prof in self.profiles.items():
                if (kh, kw) == (h, w):
                    return prof, (m, kh, kw), True

        target_ar = h / (w + 1e-9)
        best_key = None
        best_score = float("inf")
        for (m, kh, kw) in self.profiles:
            if model is not None and m != model:
                continue
            ar_diff = abs(kh / (kw + 1e-9) - target_ar) / (target_ar + 1e-9)
            px_diff = abs(kh * kw - h * w) / (h * w + 1e-9)
            score = ar_diff * 2.0 + px_diff
            if score < best_score:
                best_score, best_key = score, (m, kh, kw)

        if best_key is None:
            return self.get_profile(h, w, model=None)

        return self.profiles[best_key], best_key, False

    # ------------------------------------------------------------------
    # Build from hierarchical dataset
    # ------------------------------------------------------------------

    def build_from_hierarchical_dataset(
        self,
        root: str,
        models: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        min_refs_per_color: int = 3,
        min_consensus_colors: int = 3,
        max_per_bucket: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Walk ``root/{model}/{color}/{HxW}/*.png`` and build profiles.

        Args:
            root: Dataset root directory.
            models: Model subdirs to include. Defaults to every subdir of
                ``root`` that contains at least one known colour folder.
            colors: Colours to consider. Defaults to :data:`ALL_COLORS`.
            min_refs_per_color: Skip ``(model, color, resolution)`` buckets
                with fewer images than this.
            min_consensus_colors: A ``(model, H, W)`` profile is created only
                if at least this many consensus colours meet
                ``min_refs_per_color``. Prevents thin profiles.
            max_per_bucket: Optional cap per ``(color, resolution)`` bucket.
            verbose: Print progress.
        """
        if colors is None:
            colors = list(ALL_COLORS)
        if models is None:
            models = sorted(
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
                and self._dir_contains_any_color(os.path.join(root, d), colors)
            )

        for model in models:
            model_root = os.path.join(root, model)
            if not os.path.isdir(model_root):
                if verbose:
                    print(f"[skip] {model}: not a directory")
                continue

            buckets = self._collect_buckets(
                model_root, colors, min_refs_per_color, max_per_bucket,
            )
            if not buckets:
                if verbose:
                    print(f"[skip] {model}: no usable buckets")
                continue

            shapes = sorted({shape for (_, shape) in buckets})
            if verbose:
                print(f"[{model}] found {len(shapes)} resolutions, "
                      f"{len(buckets)} (colour, resolution) buckets")

            for shape in shapes:
                self._build_profile(
                    model=model,
                    shape=shape,
                    buckets=buckets,
                    min_consensus_colors=min_consensus_colors,
                    verbose=verbose,
                    max_per_bucket=max_per_bucket,
                )

    def add_union_profiles(self, verbose: bool = True) -> None:
        """Synthesise a ``union`` pseudo-model that averages per-resolution
        profiles across all real models. Useful when encoders are believed
        identical or when bypassing unknown-source images.
        """
        per_shape: Dict[Tuple[int, int], List[ProfileV4]] = {}
        for (model, h, w), prof in self.profiles.items():
            if model == UNION_MODEL:
                continue
            per_shape.setdefault((h, w), []).append(prof)

        for shape, profs in per_shape.items():
            if len(profs) < 2:
                continue
            unioned = self._union_profiles(shape, profs)
            self.profiles[(UNION_MODEL, shape[0], shape[1])] = unioned
            if verbose:
                print(f"[union] added {UNION_MODEL}/{shape[0]}x{shape[1]} "
                      f"(from {len(profs)} models)")

    # ------------------------------------------------------------------
    # Internal build helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_contains_any_color(model_root: str, colors: List[str]) -> bool:
        try:
            return any(
                os.path.isdir(os.path.join(model_root, c)) for c in colors
            )
        except OSError:
            return False

    @staticmethod
    def _collect_buckets(
        model_root: str,
        colors: List[str],
        min_refs: int,
        max_per_bucket: Optional[int],
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """Return list of ``(color, (H, W))`` buckets that meet ``min_refs``."""
        buckets: List[Tuple[str, Tuple[int, int]]] = []
        for color in colors:
            color_dir = os.path.join(model_root, color)
            if not os.path.isdir(color_dir):
                continue
            for res_name in sorted(os.listdir(color_dir)):
                res_dir = os.path.join(color_dir, res_name)
                if not os.path.isdir(res_dir):
                    continue
                shape = _parse_res_name(res_name)
                if shape is None:
                    continue
                n = _count_images(res_dir, cap=max_per_bucket)
                if n >= min_refs:
                    buckets.append((color, shape))
        return buckets

    def _build_profile(
        self,
        model: str,
        shape: Tuple[int, int],
        buckets: List[Tuple[str, Tuple[int, int]]],
        min_consensus_colors: int,
        verbose: bool,
        max_per_bucket: Optional[int] = None,
    ) -> None:
        H, W = shape

        color_names: List[str] = []
        mag_list: List[np.ndarray] = []
        phase_list: List[np.ndarray] = []
        cons_list: List[np.ndarray] = []
        n_per_color: Dict[str, int] = {}

        consensus_color_count = 0

        # First pass: consensus colours
        for color in CONSENSUS_COLORS:
            if (color, shape) not in buckets:
                continue
            mag, phase, cons, n = self._accumulate_color(
                model, color, shape,
                verbose=verbose, max_images=max_per_bucket,
            )
            if n == 0:
                continue
            color_names.append(color)
            mag_list.append(mag)
            phase_list.append(phase)
            cons_list.append(cons)
            n_per_color[color] = n
            consensus_color_count += 1

        if consensus_color_count < min_consensus_colors:
            if verbose:
                print(f"  [skip] {model}/{H}x{W}: only "
                      f"{consensus_color_count} consensus colour(s)")
            return

        # Second pass: content colours (diverse, gradient) — used for baseline only
        content_mag_accum = None
        n_content = 0
        for color in CONTENT_COLORS:
            if (color, shape) not in buckets:
                continue
            mag, _, _, n = self._accumulate_color(
                model, color, shape,
                verbose=verbose, max_images=max_per_bucket,
            )
            if n == 0:
                continue
            n_per_color[color] = n
            n_content += n
            content_mag_accum = (mag * n) if content_mag_accum is None \
                else content_mag_accum + mag * n

        # Cross-colour consensus
        per_color_mag = np.stack(mag_list, axis=0)            # (C, H, W, 3)
        per_color_phase = np.stack(phase_list, axis=0)        # (C, H, W, 3)
        per_color_cons = np.stack(cons_list, axis=0)          # (C, H, W, 3)

        unit = np.exp(1j * per_color_phase)
        mean_unit = np.mean(unit, axis=0)                     # (H, W, 3)
        consensus_coherence = np.abs(mean_unit)
        consensus_phase = np.angle(mean_unit)

        # Cross-colour / inverted-pair agreement: for every colour that has a
        # natural inverse in the set (black<->white, blue<->... not as clean),
        # compute |cos(phase_c - phase_other)| averaged across pairs. This
        # generalises v3's black_white_agreement.
        inverted_agreement = self._compute_inverted_agreement(
            color_names, per_color_phase,
        )

        avg_wm_magnitude = np.mean(per_color_mag, axis=0)

        if content_mag_accum is not None and n_content > 0:
            content_baseline = content_mag_accum / n_content
        else:
            content_baseline = avg_wm_magnitude.copy()

        # Carrier weights: start at consensus**2, modulated by inverted
        # agreement. These are the knobs the calibration loop will adjust.
        carrier_weights = (consensus_coherence ** 2) * \
            (0.5 + 0.5 * inverted_agreement)

        profile = ProfileV4(
            model=model,
            shape=shape,
            color_names=color_names,
            per_color_magnitude=per_color_mag.astype(np.float32),
            per_color_phase=per_color_phase.astype(np.float32),
            per_color_consistency=per_color_cons.astype(np.float32),
            consensus_coherence=consensus_coherence.astype(np.float32),
            consensus_phase=consensus_phase.astype(np.float32),
            inverted_agreement=inverted_agreement.astype(np.float32),
            avg_wm_magnitude=avg_wm_magnitude.astype(np.float32),
            content_baseline=content_baseline.astype(np.float32),
            carrier_weights=carrier_weights.astype(np.float32),
            n_refs_per_color=n_per_color,
            n_content_refs=n_content,
        )
        self.profiles[(model, H, W)] = profile

        if verbose:
            self._print_top_carriers(profile)

    @staticmethod
    def _accumulate_color(
        model: str,
        color: str,
        shape: Tuple[int, int],
        verbose: bool,
        max_images: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Per-colour FFT accumulation at a fixed resolution.

        Returns ``(avg_magnitude, mean_phase_angle, phase_consistency, n)``.
        """
        H, W = shape
        # ``model`` here is expected to be the full model dir name; callers
        # pre-validate that the bucket exists.
        res_dir = _find_res_dir(model, color, shape)
        if res_dir is None:
            return np.zeros((H, W, 3), np.float64), \
                   np.zeros((H, W, 3), np.float64), \
                   np.zeros((H, W, 3), np.float64), 0

        paths = _list_images(res_dir)
        if max_images is not None:
            paths = paths[:max_images]
        mag_sum: Optional[np.ndarray] = None
        unit_sum: Optional[np.ndarray] = None
        n = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LANCZOS4)
            mag = np.empty((H, W, 3), np.float64)
            unit = np.empty((H, W, 3), np.complex128)
            for ch in range(3):
                fft_r = np.fft.fft2(rgb[:, :, ch])
                mag[:, :, ch] = np.abs(fft_r)
                unit[:, :, ch] = np.exp(1j * np.angle(fft_r))
            if mag_sum is None:
                mag_sum = mag
                unit_sum = unit
            else:
                mag_sum += mag
                unit_sum += unit
            n += 1

        if n == 0:
            return np.zeros((H, W, 3), np.float64), \
                   np.zeros((H, W, 3), np.float64), \
                   np.zeros((H, W, 3), np.float64), 0

        avg_mag = mag_sum / n
        mean_unit = unit_sum / n
        phase = np.angle(mean_unit)
        cons = np.abs(mean_unit)
        if verbose:
            print(f"    {color}/{H}x{W}: {n} images, "
                  f"cons(G)={float(cons[..., 1].mean()):.4f}")
        return avg_mag, phase, cons, n

    @staticmethod
    def _compute_inverted_agreement(
        color_names: List[str],
        per_color_phase: np.ndarray,
    ) -> np.ndarray:
        """Mean pairwise ``|cos(phase_a - phase_b)|`` across all colour pairs.

        If black and white are both present, their pair is weighted double
        (preserving v3's strongest cross-validation signal).
        """
        n = len(color_names)
        if n < 2:
            return np.ones(per_color_phase.shape[1:], dtype=np.float64)

        acc = np.zeros(per_color_phase.shape[1:], dtype=np.float64)
        weight = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                pair = (color_names[i], color_names[j])
                w = 2.0 if pair in (("black", "white"), ("white", "black")) \
                    else 1.0
                diff = per_color_phase[i] - per_color_phase[j]
                acc += w * np.abs(np.cos(diff))
                weight += w
        return acc / max(weight, 1e-9)

    @staticmethod
    def _union_profiles(
        shape: Tuple[int, int],
        profs: List[ProfileV4],
    ) -> ProfileV4:
        """Average across models at a fixed resolution (phase-aware)."""
        H, W = shape
        mag = np.mean(np.stack([p.avg_wm_magnitude for p in profs], 0), 0)
        unit = np.mean(
            np.stack([np.exp(1j * p.consensus_phase) for p in profs], 0), 0,
        )
        cons = np.abs(unit) * np.mean(
            np.stack([p.consensus_coherence for p in profs], 0), 0,
        )
        phase = np.angle(unit)
        inv = np.mean(np.stack([p.inverted_agreement for p in profs], 0), 0)
        content = np.mean(np.stack([p.content_baseline for p in profs], 0), 0)

        carrier_weights = (cons ** 2) * (0.5 + 0.5 * inv)

        merged_refs: Dict[str, int] = {}
        n_content = 0
        for p in profs:
            for c, n in p.n_refs_per_color.items():
                merged_refs[c] = merged_refs.get(c, 0) + n
            n_content += p.n_content_refs

        return ProfileV4(
            model=UNION_MODEL,
            shape=shape,
            color_names=list(CONSENSUS_COLORS),
            per_color_magnitude=np.zeros((0, H, W, 3), np.float32),
            per_color_phase=np.zeros((0, H, W, 3), np.float32),
            per_color_consistency=np.zeros((0, H, W, 3), np.float32),
            consensus_coherence=cons.astype(np.float32),
            consensus_phase=phase.astype(np.float32),
            inverted_agreement=inv.astype(np.float32),
            avg_wm_magnitude=mag.astype(np.float32),
            content_baseline=content.astype(np.float32),
            carrier_weights=carrier_weights.astype(np.float32),
            n_refs_per_color=merged_refs,
            n_content_refs=n_content,
        )

    @staticmethod
    def _print_top_carriers(profile: ProfileV4, n_top: int = 10) -> None:
        H, W = profile.shape
        cg = profile.consensus_coherence[:, :, 1].copy()
        cg[0, 0] = 0.0
        flat = cg.ravel()
        idx = np.argsort(flat)[-n_top:][::-1]
        ys, xs = np.unravel_index(idx, cg.shape)
        print(f"  Top-{n_top} consensus carriers (G) "
              f"[{profile.model} / {H}x{W}]:")
        for y, x in zip(ys, xs):
            y_s = y if y <= H // 2 else y - H
            x_s = x if x <= W // 2 else x - W
            mg = profile.avg_wm_magnitude[y, x, 1]
            cs = cg[y, x]
            ia = profile.inverted_agreement[y, x, 1]
            print(f"    ({y_s:+5d},{x_s:+5d})  mag={mg:10.0f}  "
                  f"cons={cs:.4f}  agree={ia:.3f}")

    # ------------------------------------------------------------------
    # Save / Load (compact npz)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all profiles to a compressed ``.npz``.

        Format v2 changes vs v1:
          * ``np.savez_compressed`` (zlib-9) instead of ``np.savez``.
          * Bins where consensus coherence < 0.55 are zeroed before saving
            (the bypass never uses them; zeroing makes zlib compress ~80 %
            better because runs of zeros collapse completely).
          * ``phase`` stored as ``int8`` (scale saved as ``phase__scale``);
            error ≤ 1.4 ° — negligible for FFT phase-directed subtraction.
          * ``mag`` stored as ``uint8`` (log2-transformed, scale saved as
            ``mag__scale``).
          * ``cw`` stored as ``uint8`` (× 255, divide on load).
          * ``inv`` (inverted_agreement) and ``content`` (content_baseline)
            are **omitted** — neither is used by the Round-06 bypass pipeline.
        """
        data: Dict[str, np.ndarray] = {
            "format_version": np.array(5),
        }
        keys_arr = np.array(
            [f"{m}|{h}|{w}" for (m, h, w) in self.profiles], dtype=object,
        )
        data["keys"] = keys_arr

        for (model, h, w), prof in self.profiles.items():
            pfx = f"{model}|{h}x{w}/"
            rw = w // 2 + 1

            data[pfx + "color_names"] = np.array(prof.color_names, dtype=object)
            data[pfx + "n_content_refs"] = np.array(prof.n_content_refs)
            ref_keys = np.array(list(prof.n_refs_per_color.keys()), dtype=object)
            ref_vals = np.array(list(prof.n_refs_per_color.values()))
            data[pfx + "ref_counts_keys"] = ref_keys
            data[pfx + "ref_counts_vals"] = ref_vals

            mag_r   = prof.avg_wm_magnitude[:, :rw, :].astype(np.float32)
            phase_r = prof.consensus_phase[:, :rw, :].astype(np.float32)
            cons_r  = prof.consensus_coherence[:, :rw, :].astype(np.float32)
            cw_r    = prof.carrier_weights[:, :rw, :].astype(np.float32)

            # carrier mask: only retain bins the bypass will actually use
            mask = cons_r >= 0.55
            phase_r[~mask] = 0.0
            mag_r[~mask]   = 0.0
            cw_r[~mask]    = 0.0

            # cons: uint8 (0-255 maps coherence 0-1)
            data[pfx + "cons"] = np.round(
                np.clip(cons_r, 0, 1) * 255
            ).astype(np.uint8)

            # phase: int8 with scale (±π → ±127)
            phase_scale = np.float32(np.pi / 127.0)
            data[pfx + "phase"] = np.clip(
                np.round(phase_r / phase_scale), -127, 127,
            ).astype(np.int8)
            data[pfx + "phase__scale"] = phase_scale

            # mag: uint8 with scale (log2-transform first for dynamic range)
            log_mag = np.log2(1.0 + mag_r)
            mag_max = float(log_mag.max()) if log_mag.max() > 0 else 1.0
            mag_scale = np.float32(mag_max / 255.0)
            data[pfx + "mag"] = np.clip(
                np.round(log_mag / mag_scale), 0, 255,
            ).astype(np.uint8)
            data[pfx + "mag__scale"] = mag_scale

            # cw: uint8 (0-255 maps weight 0-1+)
            data[pfx + "cw"] = np.clip(
                np.round(np.clip(cw_r, 0, 1) * 255), 0, 255,
            ).astype(np.uint8)

        # Also zero cons below the carrier threshold so zlib/lzma sees dense
        # runs of zeros there (coherence < 0.55 is never used by the bypass).
        for (model, h, w) in self.profiles:
            pfx = f"{model}|{h}x{w}/"
            cons_key = pfx + "cons"
            if cons_key in data:
                c = data[cons_key].astype(np.float32)
                c[c / 255.0 < 0.55] = 0
                data[cons_key] = c.astype(np.uint8)

        # Write with LZMA compression via zipfile (stdlib; no extra deps).
        # LZMA achieves ~24 MB vs ~42 MB with zlib on this sparse dataset.
        import zipfile as _zf, io as _io
        with _zf.ZipFile(path, "w", compression=_zf.ZIP_LZMA) as zfile:
            for k, v in data.items():
                buf = _io.BytesIO()
                np.save(buf, v, allow_pickle=True)
                zfile.writestr(k + ".npy", buf.getvalue())

        sz = os.path.getsize(path) / 1e6
        print(f"CodebookV4 saved → {path}  "
              f"[{len(self.profiles)} profiles, {sz:.1f} MB]")

    def load(self, path: str) -> None:
        """Load a V4 codebook ``.npz``.

        Handles both format v1 (float16/uint8, uncompressed) and the current
        format v2 (quantized int8/uint8, zlib-compressed, sparse-zeroed).
        """
        d = np.load(path, allow_pickle=True)
        fmt = int(d["format_version"]) if "format_version" in d else 0
        if fmt < self.FORMAT_VERSION:
            raise ValueError(
                f"Expected format_version>={self.FORMAT_VERSION}, got {fmt}. "
                "Rebuild with scripts/build_codebook_v4.py."
            )
        # fmt==4: original uncompressed float16/uint8 format
        # fmt>=5: compressed sparse format (int8 phase, uint8 mag/cw, zlib)
        v2 = fmt >= 5  # compressed/sparse format introduced in v5

        for entry in d["keys"]:
            entry = str(entry)
            model, h_str, w_str = entry.split("|")
            h, w = int(h_str), int(w_str)
            pfx = f"{model}|{h}x{w}/"
            rw = w // 2 + 1

            # --- decode consensus coherence (uint8 in both formats) --------
            cons_r = d[pfx + "cons"].astype(np.float64) / 255.0

            # --- decode phase -----------------------------------------------
            if v2:
                ph_scale = float(d[pfx + "phase__scale"])
                phase_r = d[pfx + "phase"].astype(np.float64) * ph_scale
            else:
                phase_r = d[pfx + "phase"].astype(np.float64)

            # --- decode magnitude -------------------------------------------
            if v2:
                mg_scale = float(d[pfx + "mag__scale"])
                log_mag = d[pfx + "mag"].astype(np.float64) * mg_scale
                mag_r = np.power(2.0, log_mag) - 1.0
            else:
                mag_r = np.power(2.0, d[pfx + "mag"].astype(np.float64)) - 1.0

            # --- decode carrier weights -------------------------------------
            if v2:
                cw_r = d[pfx + "cw"].astype(np.float64) / 255.0
            else:
                cw_r = d[pfx + "cw"].astype(np.float64)

            # --- fields dropped in v2: reconstruct as zero arrays ----------
            if v2:
                inv_r     = np.zeros_like(cons_r)
                content_r = np.zeros_like(mag_r)
            else:
                inv_r = d[pfx + "inv"].astype(np.float64) / 255.0
                content_r = (
                    np.power(2.0, d[pfx + "content"].astype(np.float64)) - 1.0
                )

            mag_full     = SpectralCodebook._rfft_to_full_sym(mag_r, h, w)
            phase_full   = SpectralCodebook._rfft_to_full_anti(phase_r, h, w)
            cons_full    = SpectralCodebook._rfft_to_full_sym(cons_r, h, w)
            inv_full     = SpectralCodebook._rfft_to_full_sym(inv_r, h, w)
            content_full = SpectralCodebook._rfft_to_full_sym(content_r, h, w)
            cw_full      = SpectralCodebook._rfft_to_full_sym(cw_r, h, w)

            color_names = [str(c) for c in d[pfx + "color_names"]]
            ref_keys = [str(c) for c in d[pfx + "ref_counts_keys"]]
            ref_vals = [int(v) for v in d[pfx + "ref_counts_vals"]]
            n_refs = dict(zip(ref_keys, ref_vals))
            n_content = int(d[pfx + "n_content_refs"])

            prof = ProfileV4(
                model=model,
                shape=(h, w),
                color_names=color_names,
                per_color_magnitude=np.zeros((0, h, w, 3), np.float32),
                per_color_phase=np.zeros((0, h, w, 3), np.float32),
                per_color_consistency=np.zeros((0, h, w, 3), np.float32),
                consensus_coherence=cons_full.astype(np.float32),
                consensus_phase=phase_full.astype(np.float32),
                inverted_agreement=inv_full.astype(np.float32),
                avg_wm_magnitude=mag_full.astype(np.float32),
                content_baseline=content_full.astype(np.float32),
                carrier_weights=cw_full.astype(np.float32),
                n_refs_per_color=n_refs,
                n_content_refs=n_content,
            )
            self.profiles[(model, h, w)] = prof

        print(f"CodebookV4 loaded: {len(self.profiles)} profiles "
              f"across {len(self.models)} model(s)")

    # ------------------------------------------------------------------
    # Carrier-weight mutation (used by the calibration loop)
    # ------------------------------------------------------------------

    def update_carrier_weights(
        self,
        key: Tuple[str, int, int],
        delta: np.ndarray,
        clip: Tuple[float, float] = (0.0, 4.0),
    ) -> None:
        """Multiply a profile's ``carrier_weights`` by ``delta`` in place.

        Used by ``calibrate_from_feedback.py``. ``delta`` must broadcast to
        ``(H, W, 3)``; values >1 strengthen subtraction at those bins.
        """
        if key not in self.profiles:
            raise KeyError(f"No profile for {key}")
        prof = self.profiles[key]
        prof.carrier_weights = np.clip(
            prof.carrier_weights * delta.astype(np.float32),
            clip[0], clip[1],
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# SynthIDBypassV4
# ---------------------------------------------------------------------------

class SynthIDBypassV4:
    """Cross-colour consensus watermark dissolver.

    Fidelity-preserving FFT subtraction using :class:`SpectralCodebookV4`.
    Main changes vs v3:

    - Mask bins by ``consensus_coherence`` (>= ``tau``) instead of per-colour
      coherence only. Non-carrier bins are never touched.
    - Luminance-safe DC: DC bin is untouched; radial-ramp suppression on the
      ring just above DC prevents tone shifts.
    - Per-bin subtraction uses the profile's live ``carrier_weights``, which
      the calibration loop updates from manual detection tallies.
    - PSNR-floor rollback: after each pass, if PSNR drops below the floor,
      that pass is reverted and subsequent passes are skipped.
    """

    STRENGTH_PRESETS: Dict[str, Dict[str, float]] = {
        "gentle":     {"removal": 0.55, "tau": 0.70, "mag_cap": 0.55,
                       "dc_radius": 32, "psnr_floor": 44.0, "passes": 1},
        "moderate":   {"removal": 0.75, "tau": 0.60, "mag_cap": 0.75,
                       "dc_radius": 26, "psnr_floor": 42.0, "passes": 2},
        "aggressive": {"removal": 0.92, "tau": 0.50, "mag_cap": 0.88,
                       "dc_radius": 22, "psnr_floor": 40.0, "passes": 3},
        "maximum":    {"removal": 1.00, "tau": 0.40, "mag_cap": 0.95,
                       "dc_radius": 18, "psnr_floor": 37.0, "passes": 3},
        # Round-02: spectral + orthogonal pixel-domain perturbations
        # (JPEG roundtrip, luminance noise, bilateral edge-aware filter). The
        # spectral pass removes what the codebook can see; the post stages
        # defeat what spectral subtraction leaves behind. Fidelity floors
        # reflect the combined end-to-end pipeline.
        "demolish":   {"removal": 1.05, "tau": 0.35, "mag_cap": 0.92,
                       "dc_radius": 14, "psnr_floor": 36.0, "passes": 3,
                       "post_jpeg_q": 90, "post_noise_sigma": 0.8,
                       "post_bilateral_d": 0, "post_gamma": 0.0,
                       "post_saturation": 0.0},
        "annihilate": {"removal": 1.15, "tau": 0.28, "mag_cap": 0.96,
                       "dc_radius": 10, "psnr_floor": 33.0, "passes": 4,
                       "post_jpeg_q": 85, "post_noise_sigma": 1.2,
                       "post_bilateral_d": 5, "post_bilateral_sigma": 14.0,
                       "post_gamma": 0.015, "post_saturation": 0.03},
        "combo":      {"removal": 1.10, "tau": 0.30, "mag_cap": 0.95,
                       "dc_radius": 12, "psnr_floor": 32.0, "passes": 3,
                       "post_jpeg_q_chain": [92, 86], "post_noise_sigma": 1.0,
                       "post_bilateral_d": 5, "post_bilateral_sigma": 12.0,
                       "post_gamma": 0.010, "post_saturation": 0.025,
                       "post_pixel_shift": 1},
    }

    # Round-03: universal absolute-bin subtraction at native resolution.
    # Faithfully reproduces the recipe from
    # "How to Reverse SynthID (legally)" (Aloshdenny, 2026):
    #   * target only a short list of absolute FFT bins (blog + harvested)
    #   * single pass, no resize, work in the image's own FFT grid
    #   * per-bin magnitude cap at ~30 % of ``|image_fft|``
    #   * subtract in the direction of the codebook phase (complex vector)
    UNIVERSAL_PRESETS: Dict[str, Dict[str, float]] = {
        # Codebook-phase variants (round-03, retained for comparison).
        "blog_pure":   {"mag_cap": 0.30, "removal": 0.95, "passes": 1,
                        "harvest_k": 64, "consensus_floor": 0.55,
                        "psnr_floor": 40.0, "phase_source": "codebook"},
        "blog_plus":   {"mag_cap": 0.33, "removal": 1.00, "passes": 1,
                        "harvest_k": 96, "consensus_floor": 0.50,
                        "psnr_floor": 38.0, "phase_source": "codebook",
                        "post_jpeg_q": 92, "post_noise_sigma": 0.3},
        "blog_combo":  {"mag_cap": 0.35, "removal": 1.00, "passes": 2,
                        "harvest_k": 128, "consensus_floor": 0.45,
                        "psnr_floor": 36.0, "phase_source": "codebook",
                        "post_jpeg_q": 90, "post_noise_sigma": 0.5,
                        "post_bilateral_d": 3, "post_bilateral_sigma": 8.0},
        # Residual-phase variants (round-04). Phase comes from the image's
        # own denoise-residual at each carrier bin — this matches the blog's
        # published detector recipe and is correct for natural content.
        "residual_pure":   {"mag_cap": 0.30, "removal": 0.95, "passes": 1,
                            "harvest_k": 64, "consensus_floor": 0.55,
                            "psnr_floor": 40.0, "phase_source": "residual",
                            "denoise_h": 10},
        "residual_plus":   {"mag_cap": 0.35, "removal": 1.00, "passes": 1,
                            "harvest_k": 96, "consensus_floor": 0.50,
                            "psnr_floor": 38.0, "phase_source": "residual",
                            "denoise_h": 12,
                            "post_jpeg_q": 92, "post_noise_sigma": 0.3},
        "residual_combo":  {"mag_cap": 0.40, "removal": 1.00, "passes": 2,
                            "harvest_k": 128, "consensus_floor": 0.45,
                            "psnr_floor": 36.0, "phase_source": "residual",
                            "denoise_h": 15,
                            "post_jpeg_q": 90, "post_noise_sigma": 0.5,
                            "post_bilateral_d": 3, "post_bilateral_sigma": 8.0},
    }

    # Round-06: unified "collage fragmentation + VAE + squeeze" attack.
    # Directly targets the failure modes Gemini itself enumerates in its
    # app: fragmented/collage spatial signals, significant shrinking,
    # color-contrast changes, heavy compression, re-encoding. Works on
    # both gemini-3.1 and nano-banana-pro because it's content-agnostic:
    # no carrier-bin knowledge is required; the whole mid-freq band is
    # wholesale replaced (VAE) and the spatial grid is fragmented
    # (elastic deformation) so the detector has nowhere left to look.
    FINAL_PRESETS: Dict[str, Dict[str, float]] = {
        # Balanced: best fidelity, still hits every failure mode once.
        "final":   {"vae_strength": 1.0,
                    "elastic_alpha": 1.8, "elastic_sigma": 56.0,
                    "squeeze_ratio": 0.90,
                    "geo_rotation_deg": 0.3, "geo_zoom": 0.008,
                    "brightness": 0.008, "contrast": 0.012,
                    "saturation": -0.010, "hue_deg": 0.3,
                    "do_residual": True, "residual_mag_cap": 0.30,
                    "residual_removal": 0.9,
                    "residual_harvest_k": 72,
                    "residual_consensus_floor": 0.55,
                    "residual_denoise_h": 10.0,
                    "post_jpeg_q_chain": [92, 88],
                    "post_noise_sigma": 0.4,
                    "psnr_floor": 14.0},
        # Maximum-strength: every dial cranked; accept worse raw PSNR for
        # the strongest possible attack. Still visually similar but every
        # stage hits harder.
        "nuke":    {"vae_strength": 1.0, "vae_passes": 2,
                    "elastic_alpha": 2.8, "elastic_sigma": 44.0,
                    "squeeze_ratio": 0.82,
                    "geo_rotation_deg": 0.6, "geo_zoom": 0.015,
                    "brightness": 0.012, "contrast": 0.020,
                    "saturation": -0.018, "hue_deg": 0.6,
                    "do_residual": True, "residual_mag_cap": 0.45,
                    "residual_removal": 1.05,
                    "residual_harvest_k": 128,
                    "residual_consensus_floor": 0.45,
                    "residual_denoise_h": 12.0,
                    "post_jpeg_q_chain": [88, 84, 90],
                    "post_noise_sigma": 0.8,
                    "post_bilateral_d": 5, "post_bilateral_sigma": 10.0,
                    "post_pixel_shift": 1,
                    "psnr_floor": 11.0},
    }

    # Round-05: Diffusion-VAE Re-Generation (Gowal et al. 2026, Section 6.1).
    # The paper explicitly concedes VAE re-generation as a watermark-removal
    # vector; their mitigation is "robustness to off-the-shelf *weak* VAEs".
    # We use the SD ft-mse VAE which is a strictly stronger disruptor than
    # those used during SynthID's adversarial training. Stacked with the
    # paper's own worst-case geometric transforms (Section 4, "Combination-
    # Worst": small rotation + Cropresize + tiny translate) and an optional
    # residual-phase sweep, this closes three orthogonal evasion channels in
    # a single pipeline.
    REGEN_PRESETS: Dict[str, Dict[str, float]] = {
        # Pure VAE roundtrip. ≈ 25 dB PSNR on 1024-class images; still
        # visually very close to the source. No geometric or FFT stages.
        "regen_pure":  {"vae_strength": 1.0, "psnr_floor": 20.0,
                        "geo_rotation_deg": 0.0, "geo_zoom": 0.0,
                        "geo_translate_px": 0},
        # VAE + CombinationWorst geometry. Small rotation + 1 % centre crop
        # + 1-pixel shift. Paper Table 1 shows this is the regime where
        # SynthID-O itself only hits ~98 % TPR before we add re-generation.
        "regen_plus":  {"vae_strength": 1.0, "psnr_floor": 18.0,
                        "geo_rotation_deg": 0.4, "geo_zoom": 0.012,
                        "geo_translate_px": 1},
        # Full stack: VAE + geometry + residual-phase FFT subtraction at
        # blog carrier bins + JPEG-90 + mild luma noise. Defence in depth.
        "regen_combo": {"vae_strength": 1.0, "psnr_floor": 16.0,
                        "geo_rotation_deg": 0.5, "geo_zoom": 0.015,
                        "geo_translate_px": 1,
                        "do_residual": True, "residual_mag_cap": 0.35,
                        "residual_removal": 1.00,
                        "residual_harvest_k": 96,
                        "residual_consensus_floor": 0.50,
                        "residual_denoise_h": 10.0,
                        "post_jpeg_q": 90, "post_noise_sigma": 0.5,
                        "post_bilateral_d": 0},
    }

    def __init__(self, extractor=None):
        self.extractor = extractor

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def bypass_v4(
        self,
        image: np.ndarray,
        codebook: SpectralCodebookV4,
        strength: str = "moderate",
        model: Optional[str] = None,
        verify: bool = True,
    ) -> BypassResult:
        """Dissolve the SynthID watermark in ``image``.

        Args:
            image: RGB image, HxWx3, uint8 or float in [0, 255] / [0, 1].
            codebook: A loaded :class:`SpectralCodebookV4`.
            strength: One of ``gentle``, ``moderate``, ``aggressive``,
                ``maximum``. Controls ``tau``, removal fraction, and pass count.
            model: Optional model hint (e.g. ``gemini-3.1-flash-image-preview``).
                If ``None``, the best-matching profile across models is used.
            verify: Run the attached ``extractor`` on before/after for
                confidence deltas (optional).
        """
        cfg = self.STRENGTH_PRESETS.get(
            strength, self.STRENGTH_PRESETS["moderate"],
        )
        n_passes = int(cfg["passes"])

        original_uint8 = _to_uint8(image)
        work = original_uint8.astype(np.float64)
        h, w = work.shape[:2]

        profile, key, exact = codebook.get_profile(h, w, model=model)
        prof_h, prof_w = profile.shape

        stages: List[str] = []
        avg_luminance = float(np.mean(work)) / 255.0
        prev_state = work.copy()
        rolled_back_passes = 0

        for p_idx in range(n_passes):
            if exact:
                cleaned = self._subtract_fft_exact(
                    work, profile, cfg, avg_luminance,
                )
            else:
                cleaned = self._subtract_fft_fallback(
                    work, profile, cfg, avg_luminance,
                )

            # PSNR-floor rollback check
            psnr_so_far = _psnr(original_uint8, np.clip(cleaned, 0, 255))
            if psnr_so_far < cfg["psnr_floor"]:
                rolled_back_passes += 1
                stages.append(f"pass_{p_idx}_rolled_back(psnr={psnr_so_far:.1f})")
                work = prev_state
                break
            prev_state = cleaned.copy()
            work = cleaned
            stages.append(f"pass_{p_idx}(psnr={psnr_so_far:.1f})")

        work = cv2.GaussianBlur(work, (3, 3), 0.4)
        stages.append("anti_alias")

        cleaned_uint8 = np.clip(work, 0, 255).astype(np.uint8)

        # Orthogonal pixel-domain perturbations. These are designed to
        # defeat SynthID's spatial decoder on whatever residual energy the
        # FFT subtraction left behind. Every step is gated by PSNR vs the
        # original — if the combined output dips below the floor, the last
        # step is reverted.
        pre_post_state = cleaned_uint8.copy()
        cleaned_uint8, post_stages = _apply_post_processing(
            cleaned_uint8, original_uint8, cfg,
        )
        stages.extend(post_stages)
        post_psnr = _psnr(original_uint8, cleaned_uint8)
        if post_psnr < cfg.get("psnr_floor", 30.0):
            cleaned_uint8 = pre_post_state
            stages.append(f"post_rolled_back(psnr={post_psnr:.1f})")

        psnr = _psnr(original_uint8, cleaned_uint8)
        ssim = _ssim(original_uint8, cleaned_uint8)

        detection_before = detection_after = None
        if verify and self.extractor is not None:
            try:
                rb = self.extractor.detect_array(original_uint8)
                detection_before = dict(
                    is_watermarked=rb.is_watermarked,
                    confidence=rb.confidence,
                    phase_match=rb.phase_match,
                )
            except Exception:
                pass
            try:
                ra = self.extractor.detect_array(cleaned_uint8)
                detection_after = dict(
                    is_watermarked=ra.is_watermarked,
                    confidence=ra.confidence,
                    phase_match=ra.phase_match,
                )
            except Exception:
                pass

        success = psnr > 34.0 and ssim > 0.92
        if detection_before and detection_after:
            cd = detection_before["confidence"] - detection_after["confidence"]
            success = success and (
                cd > 0.10 or not detection_after["is_watermarked"]
            )

        return BypassResult(
            success=success,
            cleaned_image=cleaned_uint8,
            psnr=psnr,
            ssim=ssim,
            detection_before=detection_before,
            detection_after=detection_after,
            stages_applied=stages,
            details={
                "version": "v4_consensus",
                "strength": strength,
                "profile_key": f"{key[0]}/{key[1]}x{key[2]}",
                "exact_match": exact,
                "n_passes_applied": n_passes - rolled_back_passes,
                "n_passes_requested": n_passes,
                "n_passes_rolled_back": rolled_back_passes,
                "avg_luminance": avg_luminance,
                "colors_in_consensus": profile.color_names,
                "carrier_weight_mean": float(
                    np.mean(profile.carrier_weights[..., 1])
                ),
            },
        )

    def bypass_v4_file(
        self,
        input_path: str,
        output_path: str,
        codebook: SpectralCodebookV4,
        strength: str = "moderate",
        model: Optional[str] = None,
        verify: bool = True,
    ) -> BypassResult:
        """Read an image, dissolve, write result. Returns the BypassResult."""
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load: {input_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if strength in self.FINAL_PRESETS:
            result = self.bypass_v4_final(
                img_rgb, codebook, strength=strength, model=model,
            )
        elif strength in self.REGEN_PRESETS:
            result = self.bypass_v4_regen(
                img_rgb, codebook, strength=strength, model=model,
            )
        elif strength in self.UNIVERSAL_PRESETS:
            result = self.bypass_v4_universal(
                img_rgb, codebook, strength=strength, model=model,
            )
        else:
            result = self.bypass_v4(
                img_rgb, codebook, strength=strength,
                model=model, verify=verify,
            )
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(
            output_path,
            cv2.cvtColor(result.cleaned_image, cv2.COLOR_RGB2BGR),
        )
        return result

    # ------------------------------------------------------------------
    # Universal absolute-bin bypass (Round-03 / blog V3 recipe)
    # ------------------------------------------------------------------

    def bypass_v4_universal(
        self,
        image: np.ndarray,
        codebook: SpectralCodebookV4,
        strength: str = "blog_pure",
        model: Optional[str] = None,
    ) -> BypassResult:
        """Native-resolution absolute-bin subtraction.

        Targets a fixed set of ``(fy, fx)`` bin offsets (blog universals +
        codebook-harvested top carriers), using the codebook's phase at each
        bin and capping magnitude at a small fraction of the image's own
        spectrum. Complex-vector subtraction, not magnitude subtraction —
        the phase direction is what the detector keys on and what we rotate
        out of alignment.
        """
        cfg = self.UNIVERSAL_PRESETS.get(
            strength, self.UNIVERSAL_PRESETS["blog_pure"],
        )
        original_uint8 = _to_uint8(image)
        work = original_uint8.astype(np.float64)
        H, W = work.shape[:2]

        profile, key, exact = codebook.get_profile(H, W, model=model)
        pH, pW = profile.shape

        # Bin list: blog universals (scaled to (H, W)) ∪ top harvested from
        # the closest codebook profile (re-projected into native coordinates).
        harvested = harvest_codebook_carriers(
            profile,
            top_k=int(cfg.get("harvest_k", 64)),
            consensus_floor=float(cfg.get("consensus_floor", 0.55)),
            centred=True,
        )
        universal = scale_bins_to_shape(
            UNIVERSAL_CARRIER_BINS_1024, (H, W),
        )
        harvested_native = _project_bins(harvested, (pH, pW), (H, W))

        bins = _merge_bin_lists(universal, harvested_native, image_shape=(H, W))

        mag_cap = float(cfg["mag_cap"])
        removal = float(cfg["removal"])
        n_passes = int(cfg.get("passes", 1))

        prev = work.copy()
        phase_source = cfg.get("phase_source", "codebook")
        stages: List[str] = [
            f"bins={len(bins)}(universal={len(universal)},"
            f"harvested={len(harvested_native)}) phase={phase_source}",
        ]
        rolled_back = 0
        for p_idx in range(n_passes):
            if phase_source == "residual":
                cleaned = self._subtract_native_bins_residual(
                    work, original_uint8, bins,
                    removal=removal, mag_cap=mag_cap,
                    denoise_h=float(cfg.get("denoise_h", 10.0)),
                )
            else:
                cleaned = self._subtract_native_bins(
                    work, profile, bins, (pH, pW), removal, mag_cap,
                )
            psnr_so_far = _psnr(
                original_uint8, np.clip(cleaned, 0, 255),
            )
            if psnr_so_far < cfg.get("psnr_floor", 30.0):
                rolled_back += 1
                stages.append(
                    f"pass_{p_idx}_rolled_back(psnr={psnr_so_far:.1f})"
                )
                work = prev
                break
            prev = cleaned.copy()
            work = cleaned
            stages.append(f"pass_{p_idx}(psnr={psnr_so_far:.1f})")

        cleaned_uint8 = np.clip(work, 0, 255).astype(np.uint8)

        # Optional orthogonal post-processing (same helper as bypass_v4).
        pre_post = cleaned_uint8.copy()
        cleaned_uint8, post_stages = _apply_post_processing(
            cleaned_uint8, original_uint8, cfg,
        )
        stages.extend(post_stages)
        post_psnr = _psnr(original_uint8, cleaned_uint8)
        if post_psnr < cfg.get("psnr_floor", 30.0):
            cleaned_uint8 = pre_post
            stages.append(f"post_rolled_back(psnr={post_psnr:.1f})")

        psnr = _psnr(original_uint8, cleaned_uint8)
        ssim = _ssim(original_uint8, cleaned_uint8)

        return BypassResult(
            success=psnr > 34.0 and ssim > 0.92,
            cleaned_image=cleaned_uint8,
            psnr=psnr,
            ssim=ssim,
            detection_before=None,
            detection_after=None,
            stages_applied=stages,
            details={
                "version": "v4_universal",
                "strength": strength,
                "profile_key": f"{key[0]}/{key[1]}x{key[2]}",
                "exact_match": exact,
                "n_passes_applied": n_passes - rolled_back,
                "n_passes_requested": n_passes,
                "n_passes_rolled_back": rolled_back,
                "n_bins_targeted": len(bins),
                "n_bins_universal": len(universal),
                "n_bins_harvested": len(harvested_native),
            },
        )

    # ------------------------------------------------------------------
    # Round-05: VAE re-generation + geometric combo (paper-guided attack)
    # ------------------------------------------------------------------

    def bypass_v4_regen(
        self,
        image: np.ndarray,
        codebook: SpectralCodebookV4,
        strength: str = "regen_pure",
        model: Optional[str] = None,
    ) -> BypassResult:
        """Round-05 attack: diffusion-VAE re-generation + CombinationWorst geo.

        Pipeline (gated by PSNR floor at every step):

            1. VAE roundtrip (SD ft-mse). Encodes to 4-channel latent, decodes.
               The SynthID spatial watermark is projected off the VAE's natural
               image manifold and does not survive reconstruction. This is the
               attack the paper (Section 6.1) explicitly concedes.
            2. Geometric combo (optional): small rotation + centre zoom +
               single-pixel translate, all inverted with bilinear resample so
               the output is the same shape but the watermark's spatial grid
               is destroyed. Directly mirrors the Table 1 "CombinationWorst".
            3. Residual-phase FFT subtraction (optional) at the blog carrier
               bins, to scrub any residual watermark energy that survived 1-2.
            4. Optional post-processing (JPEG, luma noise) on top.
        """
        cfg = self.REGEN_PRESETS.get(strength, self.REGEN_PRESETS["regen_pure"])
        original_uint8 = _to_uint8(image)

        stages: List[str] = []
        floor = float(cfg.get("psnr_floor", 20.0))

        # Stage 1: VAE roundtrip
        try:
            from vae_regen import vae_roundtrip
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from vae_regen import vae_roundtrip
        work = vae_roundtrip(
            original_uint8,
            strength=float(cfg.get("vae_strength", 1.0)),
        )
        vae_psnr = _psnr(original_uint8, work)
        stages.append(f"vae_roundtrip(psnr={vae_psnr:.1f})")

        # Stage 2: Geometric combo — ran as a single composite warp so we
        # only resample once. Rotation + zoom + translate then resample-back
        # to the original shape.
        deg = float(cfg.get("geo_rotation_deg", 0.0))
        zoom = float(cfg.get("geo_zoom", 0.0))
        shift = int(cfg.get("geo_translate_px", 0))
        if abs(deg) > 1e-6 or zoom > 1e-6 or shift > 0:
            work = _apply_geo_combo(work, deg=deg, zoom=zoom, shift_px=shift)
            geo_psnr = _psnr(original_uint8, work)
            stages.append(
                f"geo(deg={deg:+.2f},zoom={zoom:.3f},shift={shift}px)"
                f"(psnr={geo_psnr:.1f})"
            )

        # Stage 3: Residual-phase FFT subtraction at blog carrier bins.
        if bool(cfg.get("do_residual", False)):
            H, W = work.shape[:2]
            profile, key, exact = codebook.get_profile(H, W, model=model)
            pH, pW = profile.shape
            harvested = harvest_codebook_carriers(
                profile,
                top_k=int(cfg.get("residual_harvest_k", 96)),
                consensus_floor=float(cfg.get("residual_consensus_floor", 0.50)),
                centred=True,
            )
            universal = scale_bins_to_shape(
                UNIVERSAL_CARRIER_BINS_1024, (H, W),
            )
            harvested_native = _project_bins(harvested, (pH, pW), (H, W))
            bins = _merge_bin_lists(universal, harvested_native, image_shape=(H, W))
            before = work.astype(np.float64)
            cleaned = self._subtract_native_bins_residual(
                before, original_uint8, bins,
                removal=float(cfg.get("residual_removal", 1.0)),
                mag_cap=float(cfg.get("residual_mag_cap", 0.35)),
                denoise_h=float(cfg.get("residual_denoise_h", 10.0)),
            )
            cand = np.clip(cleaned, 0, 255).astype(np.uint8)
            resid_psnr = _psnr(original_uint8, cand)
            if resid_psnr >= floor:
                work = cand
                stages.append(
                    f"residual_subtract(bins={len(bins)}"
                    f",psnr={resid_psnr:.1f})"
                )
            else:
                stages.append(
                    f"residual_subtract_skipped(psnr={resid_psnr:.1f}<{floor:.1f})"
                )

        # Stage 4: orthogonal post-processing
        pre_post = work.copy()
        work, post_stages = _apply_post_processing(
            work, original_uint8, cfg,
        )
        stages.extend(post_stages)
        post_psnr = _psnr(original_uint8, work)
        if post_psnr < floor:
            work = pre_post
            stages.append(f"post_rolled_back(psnr={post_psnr:.1f})")

        psnr = _psnr(original_uint8, work)
        ssim = _ssim(original_uint8, work)

        profile_key = ""
        exact_match = False
        try:
            profile, key, exact = codebook.get_profile(
                original_uint8.shape[0], original_uint8.shape[1], model=model,
            )
            profile_key = f"{key[0]}/{key[1]}x{key[2]}"
            exact_match = exact
        except Exception:
            pass

        return BypassResult(
            success=psnr > 18.0 and ssim > 0.60,
            cleaned_image=work,
            psnr=psnr,
            ssim=ssim,
            detection_before=None,
            detection_after=None,
            stages_applied=stages,
            details={
                "version": "v4_regen",
                "strength": strength,
                "profile_key": profile_key,
                "exact_match": exact_match,
                "n_passes_applied": 1,
                "n_passes_requested": 1,
                "n_passes_rolled_back": 0,
                "vae_psnr": vae_psnr,
            },
        )

    # ------------------------------------------------------------------
    # Round-06: unified all-in-one attack
    # ------------------------------------------------------------------

    def bypass_v4_final(
        self,
        image: np.ndarray,
        codebook: SpectralCodebookV4,
        strength: str = "final",
        model: Optional[str] = None,
    ) -> BypassResult:
        """Unified all-in-one attack targeting every Gemini-confirmed failure mode.

        Pipeline (each stage independently PSNR-gated via a soft floor):

            1. **VAE re-generation** (SD ft-mse), optionally *N* passes. Kills
               the mid-frequency watermark band by round-tripping through a
               natural-image manifold the SynthID decoder was never trained
               against (Gowal et al. 2026, §6.1).
            2. **Elastic deformation**. Smooth, low-frequency random warp
               field of amplitude ``elastic_alpha`` px, smoothed by
               ``elastic_sigma`` px. Every local neighbourhood gets its
               own sub-pixel offset — the *direct pixel-level analogue* of
               "collages and overlays", which Gemini's own app confirms
               defeats SynthID because "the digital signature can become
               fragmented".
            3. **Global geometric combo**. Small rotation + zoom, single
               affine warp. Catches any rigid-transform remnant.
            4. **Resize-squeeze**. Down-sample to ``squeeze_ratio`` × linear
               size with AREA, back up with LANCZOS. Erases sub-pixel info
               in the watermark band — directly targets Gemini's
               "significant resizing / shrinking" failure mode.
            5. **Color-contrast nudge**. Brightness/contrast/saturation/hue
               micro-shift. Targets Gemini's "color and contrast changes"
               failure mode.
            6. **Residual-phase FFT subtraction** (optional) at blog +
               harvested carrier bins, cap-limited.
            7. **JPEG chain + luma noise + bilateral + pixel shift** via the
               shared ``_apply_post_processing`` helper. Targets Gemini's
               "heavy compression / re-encoding" failure mode.

        Every stage is either additive to, or orthogonal to, the previous.
        Two presets are provided:

            * ``final`` — balanced fidelity, hits every mode once.
            * ``nuke``  — maximum strength, 2 VAE passes, tripled knobs.
        """
        cfg = self.FINAL_PRESETS.get(strength, self.FINAL_PRESETS["final"])
        original_uint8 = _to_uint8(image)
        stages: List[str] = []
        floor = float(cfg.get("psnr_floor", 12.0))

        # Stage 1: VAE roundtrip (possibly multi-pass).
        try:
            from vae_regen import vae_roundtrip
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from vae_regen import vae_roundtrip

        work = original_uint8
        n_vae = int(cfg.get("vae_passes", 1))
        vae_strength = float(cfg.get("vae_strength", 1.0))
        for pass_i in range(max(1, n_vae)):
            work = vae_roundtrip(work, strength=vae_strength)
            stages.append(
                f"vae_pass_{pass_i}(psnr={_psnr(original_uint8, work):.1f})"
            )

        # Stage 2: elastic deformation — the "collage fragmentation" blow.
        e_alpha = float(cfg.get("elastic_alpha", 0.0))
        e_sigma = float(cfg.get("elastic_sigma", 64.0))
        if e_alpha > 1e-6:
            work = _elastic_deform(work, alpha=e_alpha, sigma=e_sigma)
            stages.append(
                f"elastic(a={e_alpha:.2f},s={e_sigma:.0f})"
                f"(psnr={_psnr(original_uint8, work):.1f})"
            )

        # Stage 3: global geometric combo.
        deg = float(cfg.get("geo_rotation_deg", 0.0))
        zoom = float(cfg.get("geo_zoom", 0.0))
        shift = int(cfg.get("geo_translate_px", 0))
        if abs(deg) > 1e-6 or zoom > 1e-6 or shift > 0:
            work = _apply_geo_combo(work, deg=deg, zoom=zoom, shift_px=shift)
            stages.append(
                f"geo(deg={deg:+.2f},zoom={zoom:.3f},shift={shift}px)"
                f"(psnr={_psnr(original_uint8, work):.1f})"
            )

        # Stage 4: resize-squeeze (down-then-up sample).
        s_ratio = float(cfg.get("squeeze_ratio", 0.0))
        if 0.0 < s_ratio < 1.0:
            work = _resize_squeeze(work, squeeze=s_ratio)
            stages.append(
                f"squeeze(r={s_ratio:.2f})"
                f"(psnr={_psnr(original_uint8, work):.1f})"
            )

        # Stage 5: color-contrast nudge.
        b = float(cfg.get("brightness", 0.0))
        c = float(cfg.get("contrast", 0.0))
        sat = float(cfg.get("saturation", 0.0))
        hue = float(cfg.get("hue_deg", 0.0))
        if any(abs(x) > 1e-6 for x in (b, c, sat, hue)):
            work = _color_nudge(
                work, brightness=b, contrast=c, saturation=sat, hue_deg=hue,
            )
            stages.append(
                f"color(b={b:+.3f},c={c:+.3f},s={sat:+.3f},h={hue:+.1f})"
                f"(psnr={_psnr(original_uint8, work):.1f})"
            )

        # Stage 6: residual-phase FFT subtraction (optional polish).
        if bool(cfg.get("do_residual", False)):
            H, W = work.shape[:2]
            profile, key, exact = codebook.get_profile(H, W, model=model)
            pH, pW = profile.shape
            harvested = harvest_codebook_carriers(
                profile,
                top_k=int(cfg.get("residual_harvest_k", 96)),
                consensus_floor=float(cfg.get("residual_consensus_floor", 0.50)),
                centred=True,
            )
            universal = scale_bins_to_shape(
                UNIVERSAL_CARRIER_BINS_1024, (H, W),
            )
            harvested_native = _project_bins(harvested, (pH, pW), (H, W))
            bins = _merge_bin_lists(
                universal, harvested_native, image_shape=(H, W),
            )
            before = work.astype(np.float64)
            cleaned = self._subtract_native_bins_residual(
                before, original_uint8, bins,
                removal=float(cfg.get("residual_removal", 0.9)),
                mag_cap=float(cfg.get("residual_mag_cap", 0.30)),
                denoise_h=float(cfg.get("residual_denoise_h", 10.0)),
            )
            cand = np.clip(cleaned, 0, 255).astype(np.uint8)
            resid_psnr = _psnr(original_uint8, cand)
            if resid_psnr >= floor:
                work = cand
                stages.append(
                    f"residual_subtract(bins={len(bins)}"
                    f",psnr={resid_psnr:.1f})"
                )
            else:
                stages.append(
                    f"residual_subtract_skipped"
                    f"(psnr={resid_psnr:.1f}<{floor:.1f})"
                )

        # Stage 7: post-processing (JPEG chain, noise, bilateral, pixel shift).
        pre_post = work.copy()
        work, post_stages = _apply_post_processing(
            work, original_uint8, cfg,
        )
        stages.extend(post_stages)
        post_psnr = _psnr(original_uint8, work)
        if post_psnr < floor:
            work = pre_post
            stages.append(f"post_rolled_back(psnr={post_psnr:.1f})")

        psnr = _psnr(original_uint8, work)
        ssim = _ssim(original_uint8, work)

        profile_key = ""
        exact_match = False
        try:
            profile, key, exact = codebook.get_profile(
                original_uint8.shape[0], original_uint8.shape[1], model=model,
            )
            profile_key = f"{key[0]}/{key[1]}x{key[2]}"
            exact_match = exact
        except Exception:
            pass

        return BypassResult(
            success=psnr > 15.0 and ssim > 0.55,
            cleaned_image=work,
            psnr=psnr,
            ssim=ssim,
            detection_before=None,
            detection_after=None,
            stages_applied=stages,
            details={
                "version": "v4_final",
                "strength": strength,
                "profile_key": profile_key,
                "exact_match": exact_match,
                "n_passes_applied": 1,
                "n_passes_requested": 1,
                "n_passes_rolled_back": 0,
                "n_vae_passes": n_vae,
                "elastic_alpha": e_alpha,
                "elastic_sigma": e_sigma,
                "squeeze_ratio": s_ratio,
            },
        )

    @staticmethod
    def _subtract_native_bins(
        work: np.ndarray,
        profile: "ProfileV4",
        bins: List[Tuple[int, int]],
        profile_shape: Tuple[int, int],
        removal: float,
        mag_cap: float,
    ) -> np.ndarray:
        """Per-channel bin-wise complex subtraction at native resolution.

        For each ``(fy, fx)`` we look up ``phase`` and a scale reference from
        the profile at the same *proportional* position; the magnitude we
        actually subtract is capped at ``mag_cap * |image_fft|`` so we never
        damage content-dominant bins.
        """
        H, W = work.shape[:2]
        pH, pW = profile_shape
        cleaned = np.empty_like(work)
        for ch in range(3):
            fft = np.fft.fft2(work[:, :, ch])
            # Work on a copy so symmetric updates don't double-count.
            delta = np.zeros_like(fft)
            for fy, fx in bins:
                iy_im = fy % H
                ix_im = fx % W
                # Profile position via proportional remapping.
                py = int(round(fy * pH / H)) % pH
                px = int(round(fx * pW / W)) % pW
                phase = profile.consensus_phase[py, px, ch]
                cons = profile.consensus_coherence[py, px, ch]
                # Only subtract at bins with at least weak profile support;
                # skip the ones the codebook can't confidently phase-lock.
                if cons < 0.15:
                    continue
                img_val = fft[iy_im, ix_im]
                amp_cap = mag_cap * abs(img_val)
                if amp_cap <= 0:
                    continue
                # The magnitude to remove is purely decided by the cap — we
                # never claim to know the absolute watermark magnitude at
                # arbitrary resolutions, only its phase direction. ``removal``
                # scales below 1.0 to stay on the safe side of the cap.
                amp = amp_cap * removal
                delta[iy_im, ix_im] += amp * np.exp(1j * phase)
            fft_clean = fft - delta
            # Enforce conjugate symmetry for a real-valued result.
            fft_clean = _enforce_conjugate_symmetry(fft_clean)
            cleaned[:, :, ch] = np.real(np.fft.ifft2(fft_clean))
        return np.clip(cleaned, 0, 255)

    @staticmethod
    def _subtract_native_bins_residual(
        work: np.ndarray,
        original_uint8: np.ndarray,
        bins: List[Tuple[int, int]],
        removal: float,
        mag_cap: float,
        denoise_h: float = 10.0,
    ) -> np.ndarray:
        """Residual-phase bin subtraction — the blog's published recipe.

        The correct phase of the watermark at bin ``k`` on *this* image is
        not in any codebook; it's in the image's own denoise residual:

            residual = image − denoise(image)
            phase(residual_fft[k]) = phase of the watermark at bin k
            |residual_fft[k]|     ≈ magnitude of the watermark at bin k

        We subtract up to ``mag_cap × |image_fft[k]|`` of that residual
        complex value from the image FFT, which rotates the phase off the
        watermark's axis without damaging content at those bins.
        """
        H, W = work.shape[:2]
        # Denoise on uint8 (Non-Local Means works best). We always denoise
        # the *original* so the residual reflects the watermark at the
        # un-perturbed image, even on pass 2+.
        denoised = cv2.fastNlMeansDenoisingColored(
            original_uint8, None, float(denoise_h), float(denoise_h), 7, 21,
        ).astype(np.float64)
        residual = original_uint8.astype(np.float64) - denoised

        cleaned = np.empty_like(work)
        for ch in range(3):
            fft = np.fft.fft2(work[:, :, ch])
            res_fft = np.fft.fft2(residual[:, :, ch])
            delta = np.zeros_like(fft)
            for fy, fx in bins:
                iy = fy % H
                ix = fx % W
                res_val = res_fft[iy, ix]
                if abs(res_val) == 0:
                    continue
                amount = res_val * removal
                cap = mag_cap * abs(fft[iy, ix])
                if abs(amount) > cap and cap > 0:
                    amount = amount * (cap / abs(amount))
                delta[iy, ix] += amount
            fft_clean = fft - delta
            fft_clean = _enforce_conjugate_symmetry(fft_clean)
            cleaned[:, :, ch] = np.real(np.fft.ifft2(fft_clean))
        return np.clip(cleaned, 0, 255)

    # ------------------------------------------------------------------
    # Subtraction primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _subtract_fft_exact(
        work: np.ndarray,
        profile: ProfileV4,
        cfg: Dict[str, float],
        avg_luminance: float,
    ) -> np.ndarray:
        H, W = work.shape[:2]
        tau = float(cfg["tau"])
        dc_radius = float(cfg["dc_radius"])

        fy = np.arange(H).reshape(-1, 1).astype(np.float64)
        fx = np.arange(W).reshape(1, -1).astype(np.float64)
        fy = np.where(fy > H / 2, fy - H, fy)
        fx = np.where(fx > W / 2, fx - W, fx)
        dc_ramp = np.clip(np.sqrt(fy ** 2 + fx ** 2) / dc_radius, 0, 1)

        cleaned = np.empty_like(work)
        for ch in range(3):
            image_fft = np.fft.fft2(work[:, :, ch])
            wm_fft = _estimate_watermark_fft_v4(
                image_fft=image_fft,
                profile=profile,
                channel=ch,
                cfg=cfg,
                dc_ramp=dc_ramp,
                tau=tau,
                avg_luminance=avg_luminance,
            )
            cleaned[:, :, ch] = np.real(np.fft.ifft2(image_fft - wm_fft))
        return np.clip(cleaned, 0, 255)

    @staticmethod
    def _subtract_fft_fallback(
        work: np.ndarray,
        profile: ProfileV4,
        cfg: Dict[str, float],
        avg_luminance: float,
    ) -> np.ndarray:
        """Resolution mismatch: build the watermark at the profile's native
        shape, resize spatially, subtract. Lower quality than the exact path
        but still better than a noop."""
        H, W = work.shape[:2]
        pH, pW = profile.shape

        fy = np.arange(pH).reshape(-1, 1).astype(np.float64)
        fx = np.arange(pW).reshape(1, -1).astype(np.float64)
        fy = np.where(fy > pH / 2, fy - pH, fy)
        fx = np.where(fx > pW / 2, fx - pW, fx)
        dc_ramp = np.clip(
            np.sqrt(fy ** 2 + fx ** 2) / float(cfg["dc_radius"]),
            0, 1,
        )

        tau = float(cfg["tau"])
        cleaned = np.empty_like(work)
        for ch in range(3):
            # Synthesise a "carrier-only" FFT at native profile size.
            synth_mag = profile.avg_wm_magnitude[:, :, ch] * 10.0
            synth_fft = synth_mag * np.exp(1j * profile.consensus_phase[:, :, ch])
            wm_fft = _estimate_watermark_fft_v4(
                image_fft=synth_fft,
                profile=profile,
                channel=ch,
                cfg=cfg,
                dc_ramp=dc_ramp,
                tau=tau,
                avg_luminance=avg_luminance,
            )
            wm_spatial = np.real(np.fft.ifft2(wm_fft))
            wm_resized = cv2.resize(
                wm_spatial, (W, H), interpolation=cv2.INTER_LANCZOS4,
            )
            cleaned[:, :, ch] = work[:, :, ch] - wm_resized
        return np.clip(cleaned, 0, 255)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _estimate_watermark_fft_v4(
    image_fft: np.ndarray,
    profile: ProfileV4,
    channel: int,
    cfg: Dict[str, float],
    dc_ramp: np.ndarray,
    tau: float,
    avg_luminance: float,
) -> np.ndarray:
    """Compute the complex watermark FFT to subtract from ``image_fft``.

    Same shape as ``image_fft``. Zero where ``consensus_coherence < tau``.
    """
    consensus = profile.consensus_coherence[:, :, channel]
    phase = profile.consensus_phase[:, :, channel]
    carrier_weights = profile.carrier_weights[:, :, channel]

    # Hard mask: only bins that survived the cross-colour consensus get any
    # subtraction at all.
    mask = (consensus >= tau).astype(np.float64)
    if mask.sum() == 0:
        return np.zeros_like(image_fft)

    wm_mag_base = profile.avg_wm_magnitude[:, :, channel]
    # Luminance-aware blend using content baseline as a proxy for what will be
    # in non-solid images.
    content_mag = profile.content_baseline[:, :, channel]
    wm_mag = (
        wm_mag_base * (1.0 - avg_luminance)
        + content_mag * avg_luminance * 0.35
    )

    # Apply consensus + live calibration.
    wm_mag = wm_mag * mask * carrier_weights

    # Never touch DC; ramp up through the lowest-frequency ring.
    wm_mag = wm_mag * dc_ramp
    wm_mag[0, 0] = 0.0

    ch_w = float(CHANNEL_WEIGHTS[channel])
    removal = float(cfg["removal"])
    subtract_mag = wm_mag * removal * ch_w

    # Per-bin cap: never remove more than ``mag_cap`` of whatever is actually
    # at that bin in the image. Prevents dark ringing on smooth regions.
    subtract_mag = np.minimum(subtract_mag, np.abs(image_fft) * float(cfg["mag_cap"]))
    return subtract_mag * np.exp(1j * phase)


def _project_bins(
    bins: List[Tuple[int, int]],
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Proportionally project centred signed bin offsets between shapes.

    Filters out bins that fall outside the destination Nyquist envelope.
    """
    sH, sW = from_shape
    dH, dW = to_shape
    nyq_y, nyq_x = dH // 2, dW // 2
    out: List[Tuple[int, int]] = []
    for fy, fx in bins:
        ny = int(round(fy * dH / sH))
        nx = int(round(fx * dW / sW))
        if ny == 0 and nx == 0:
            continue
        if abs(ny) >= nyq_y or abs(nx) >= nyq_x:
            continue
        out.append((ny, nx))
    return out


def _merge_bin_lists(
    *lists: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Union a sequence of signed-offset lists; deduplicate at native res."""
    H, W = image_shape
    nyq_y, nyq_x = H // 2, W // 2
    seen: set = set()
    out: List[Tuple[int, int]] = []
    for lst in lists:
        for fy, fx in lst:
            if fy == 0 and fx == 0:
                continue
            if abs(fy) >= nyq_y or abs(fx) >= nyq_x:
                continue
            if (fy, fx) in seen:
                continue
            seen.add((fy, fx))
            out.append((fy, fx))
    return out


def _apply_geo_combo(
    img_uint8: np.ndarray,
    deg: float = 0.4,
    zoom: float = 0.012,
    shift_px: int = 1,
) -> np.ndarray:
    """CombinationWorst-style geometric attack in a single resample.

    Applies rotation (``deg``), centre zoom-in (fraction of linear dimension
    cropped on each side, then resized back up), and pixel-translate in one
    affine warp so we resample only once and don't compound aliasing. The
    output is the same shape as the input; watermark carriers no longer lie
    on the grid the detector expects, which is precisely the Table 1
    "CombinationWorst" failure mode for every evaluated method.
    """
    H, W = img_uint8.shape[:2]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    scale = 1.0 + max(zoom, 0.0)
    M = cv2.getRotationMatrix2D((cx, cy), deg, scale)
    if shift_px:
        M[0, 2] += shift_px
        M[1, 2] += shift_px

    warped = cv2.warpAffine(
        img_uint8, M, (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def _elastic_deform(
    img_uint8: np.ndarray,
    alpha: float = 2.0,
    sigma: float = 48.0,
    seed: int = 0xC0FFEE,
) -> np.ndarray:
    """Spatially-varying sub-pixel warp — the "collage fragmentation" attack.

    The Gemini app itself confirms (in its own user-facing error response)
    that collages / fragmented spatial signals defeat SynthID because
    "the digital signature can become fragmented". A literal collage would
    be visible; instead we simulate the same effect by warping with a
    smooth low-frequency random vector field: every ~``sigma``-pixel
    neighbourhood gets its own independent sub-pixel shift of amplitude
    up to ``alpha``. Globally imperceptible (natural perspective-like
    flow); locally, every carrier's (fy, fx) position is displaced by a
    different amount, so phase-consensus detectors cannot aggregate.

    Parameters:
        alpha: max warp amplitude in pixels (1.5-3.0 is the useful range).
        sigma: Gaussian smoothing kernel for the vector field. Smaller
            ``sigma`` → higher-frequency fragmentation (kills short-
            wavelength carriers harder) but more visible rippling.
    """
    H, W = img_uint8.shape[:2]
    rng = np.random.default_rng(seed)
    dx = rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32)
    dy = rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32)
    k = int(2 * round(3 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    dx = cv2.GaussianBlur(dx, (k, k), sigma) * float(alpha) / 0.30
    dy = cv2.GaussianBlur(dy, (k, k), sigma) * float(alpha) / 0.30
    # Renormalise amplitude — GaussianBlur of uniform noise has an
    # RMS that depends on sigma; rescale to the requested max amplitude.
    max_abs = max(float(np.abs(dx).max()), float(np.abs(dy).max()), 1e-6)
    dx *= alpha / max_abs
    dy *= alpha / max_abs

    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    map_x = (xs + dx).astype(np.float32)
    map_y = (ys + dy).astype(np.float32)
    return cv2.remap(
        img_uint8, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )


def _resize_squeeze(
    img_uint8: np.ndarray,
    squeeze: float = 0.85,
) -> np.ndarray:
    """Downsample to ``squeeze × squeeze`` area ratio, then upsample back.

    The Gemini app's own failure-mode list calls out "shrinking to very
    small dimensions" as a detection-breaker. A single down-up roundtrip
    through AREA-then-LANCZOS irreversibly erases sub-pixel information
    in the watermark band while visually preserving the image down to
    the limits of LANCZOS reconstruction.
    """
    H, W = img_uint8.shape[:2]
    s = max(0.4, min(0.999, float(squeeze)))
    nh = max(8, int(round(H * s)))
    nw = max(8, int(round(W * s)))
    small = cv2.resize(img_uint8, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LANCZOS4)


def _color_nudge(
    img_uint8: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue_deg: float = 0.0,
) -> np.ndarray:
    """Small but non-trivial color/contrast nudge in HSV.

    Gemini's own guidance lists "extreme changes to the color balance /
    brightness / heavy filters" as detection-breakers. We stay well
    below "extreme" — typical operating point is ±1-2 % brightness,
    ±1-2 % contrast, ±1 % saturation, ±0.5° hue — but that's enough to
    shift the pixel statistics SynthID keys on.
    """
    f = img_uint8.astype(np.float32) / 255.0
    if abs(contrast) > 1e-6:
        f = 0.5 + (1.0 + contrast) * (f - 0.5)
    if abs(brightness) > 1e-6:
        f = f + brightness
    f = np.clip(f, 0.0, 1.0)
    if abs(saturation) > 1e-6 or abs(hue_deg) > 1e-6:
        rgb8 = (f * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
        if abs(hue_deg) > 1e-6:
            hsv[..., 0] = (hsv[..., 0] + hue_deg / 2.0) % 180.0
        if abs(saturation) > 1e-6:
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + saturation), 0, 255)
        rgb8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        f = rgb8.astype(np.float32) / 255.0
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def _enforce_conjugate_symmetry(fft: np.ndarray) -> np.ndarray:
    """Average ``F(k)`` with ``conj(F(-k))`` so ``ifft2(F).imag ≈ 0``.

    Needed after modifying a handful of bins by hand — the ifft of an
    asymmetric complex array would have a non-zero imaginary component
    that ``np.real`` silently discards, losing signal.
    """
    H, W = fft.shape
    out = fft.copy()
    # Handle the centre-square pair mapping: (y, x) <-> (-y mod H, -x mod W)
    out = 0.5 * (out + np.conj(out[(-np.arange(H)) % H][:, (-np.arange(W)) % W]))
    return out


def _apply_post_processing(
    cleaned: np.ndarray,
    original: np.ndarray,
    cfg: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
    """Orthogonal pixel-domain stages applied after the FFT subtraction.

    Each stage is independently gated: if a stage drops PSNR below
    ``cfg['psnr_floor']`` we skip it and continue with earlier stages applied.
    Order is chosen so that destructive stages (JPEG, noise) run first and
    cosmetic ones (gamma/saturation, pixel shift) run last.
    """
    floor = float(cfg.get("psnr_floor", 30.0))
    stages: List[str] = []
    out = cleaned.copy()

    def _try(candidate: np.ndarray, name: str) -> None:
        nonlocal out
        p = _psnr(original, candidate)
        if p >= floor:
            out = candidate
            stages.append(f"{name}(psnr={p:.1f})")
        else:
            stages.append(f"{name}_skipped(psnr={p:.1f}<{floor:.1f})")

    # 1. JPEG roundtrip(s). A single moderate-quality JPEG cycle is the
    # canonical disruptor of spatial watermarks because it quantizes
    # mid-frequency DCT coefficients — exactly the band SynthID lives in.
    jpeg_chain: List[int] = []
    if "post_jpeg_q_chain" in cfg:
        jpeg_chain = list(cfg["post_jpeg_q_chain"])
    elif "post_jpeg_q" in cfg:
        jpeg_chain = [int(cfg["post_jpeg_q"])]
    for i, q in enumerate(jpeg_chain):
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        if not ok:
            stages.append(f"jpeg_q{q}_failed")
            continue
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if dec is None:
            stages.append(f"jpeg_q{q}_decode_failed")
            continue
        _try(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB), f"jpeg{i}_q{q}")

    # 2. Luminance-only Gaussian noise. SynthID's decoder is noise-robust by
    # design, but additive perturbations in luma break the per-bin statistics
    # that phase-consensus detectors key on.
    sigma = float(cfg.get("post_noise_sigma", 0.0))
    if sigma > 0.0:
        yuv = cv2.cvtColor(out, cv2.COLOR_RGB2YUV).astype(np.float64)
        rng = np.random.default_rng(0xA55A)
        yuv[..., 0] += rng.normal(0.0, sigma, yuv[..., 0].shape)
        yuv = np.clip(yuv, 0, 255).astype(np.uint8)
        _try(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB), f"luma_noise_s{sigma:g}")

    # 3. Bilateral filter. Edge-preserving denoise that crushes micro-texture
    # patterns while leaving macro content untouched.
    d = int(cfg.get("post_bilateral_d", 0))
    if d > 0:
        sc = float(cfg.get("post_bilateral_sigma", 10.0))
        filt = cv2.bilateralFilter(out, d=d, sigmaColor=sc, sigmaSpace=sc)
        _try(filt, f"bilateral_d{d}")

    # 4. Tiny global gamma/saturation nudge in HSV — perceptually invisible
    # but shifts pixel statistics enough to disrupt spatial watermarks that
    # encode in lightness/saturation.
    gamma_delta = float(cfg.get("post_gamma", 0.0))
    sat_delta = float(cfg.get("post_saturation", 0.0))
    if abs(gamma_delta) > 1e-6 or abs(sat_delta) > 1e-6:
        f = out.astype(np.float64) / 255.0
        if abs(gamma_delta) > 1e-6:
            gamma = 1.0 + gamma_delta
            f = np.clip(f, 0, 1) ** gamma
        if abs(sat_delta) > 1e-6:
            hsv = cv2.cvtColor(
                (f * 255).astype(np.uint8), cv2.COLOR_RGB2HSV,
            ).astype(np.float64)
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat_delta), 0, 255)
            f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0
        _try(np.clip(f * 255, 0, 255).astype(np.uint8),
             f"tonemap_g{gamma_delta:+.3f}_s{sat_delta:+.3f}")

    # 5. Single-pixel shift per channel — breaks per-pixel spatial lookup
    # patterns with negligible perceptual cost at natural content.
    shift = int(cfg.get("post_pixel_shift", 0))
    if shift > 0:
        shifted = out.copy()
        for c, (dy, dx) in enumerate([(shift, 0), (0, shift), (-shift, 0)]):
            shifted[..., c] = np.roll(out[..., c], shift=(dy, dx), axis=(0, 1))
        _try(shifted, f"pixel_shift_{shift}px")

    return out, stages


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    arr = np.asarray(image)
    if np.max(arr) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64) / 255.0
    bf = b.astype(np.float64) / 255.0
    mse = float(np.mean((af - bf) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight block-SSIM on luminance (mirrors v3)."""
    af = a.astype(np.float64) / 255.0
    bf = b.astype(np.float64) / 255.0
    ga = 0.299 * af[:, :, 0] + 0.587 * af[:, :, 1] + 0.114 * af[:, :, 2]
    gb = 0.299 * bf[:, :, 0] + 0.587 * bf[:, :, 1] + 0.114 * bf[:, :, 2]
    block = 8
    rc = (ga.shape[0] // block) * block
    cc = (ga.shape[1] // block) * block
    ao = ga[:rc, :cc].reshape(rc // block, block, cc // block, block) \
        .transpose(0, 2, 1, 3).reshape(-1, block, block)
    am = gb[:rc, :cc].reshape(rc // block, block, cc // block, block) \
        .transpose(0, 2, 1, 3).reshape(-1, block, block)
    ma = ao.mean(axis=(1, 2))
    mb = am.mean(axis=(1, 2))
    va = ao.var(axis=(1, 2))
    vb = am.var(axis=(1, 2))
    cv = ((ao - ma[:, None, None]) * (am - mb[:, None, None])).mean(axis=(1, 2))
    return float(np.mean(
        (2 * ma * mb + 1e-4) * (2 * cv + 9e-4)
        / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4))
    ))


# ---------------------------------------------------------------------------
# Dataset path helpers
# ---------------------------------------------------------------------------

def _parse_res_name(name: str) -> Optional[Tuple[int, int]]:
    """Parse ``'1440x720'`` → ``(H=720, W=1440)``.

    Dataset directory names use Gemini's ``WIDTHxHEIGHT`` convention (e.g.
    ``1440x720`` for a 1440-wide, 720-tall landscape image). We return
    ``(H, W)`` to match numpy/cv2 shape ordering used everywhere else in the
    codebase. Returns ``None`` for non-resolution directories (e.g.
    ``24000hz_1ch``) so they are silently skipped during dataset walks.
    """
    parts = name.lower().split("x")
    if len(parts) != 2:
        return None
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if h <= 0 or w <= 0 or h > 16384 or w > 16384:
        return None
    return (h, w)


def _count_images(directory: str, cap: Optional[int] = None) -> int:
    n = 0
    for f in os.listdir(directory):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            n += 1
            if cap is not None and n >= cap:
                return n
    return n


def _list_images(directory: str) -> List[str]:
    out = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            out.append(os.path.join(directory, f))
    return out


# ``_accumulate_color`` derives the res_dir from (model, color, shape). The
# builder needs to know the dataset root. We store it at build time on the
# codebook instance instead of threading it through every call. The wrapper
# below keeps the API clean.
_ACTIVE_DATASET_ROOT: Optional[str] = None


def _find_res_dir(
    model: str, color: str, shape: Tuple[int, int],
) -> Optional[str]:
    """Resolve the disk path for a ``(model, color, shape)`` bucket.

    ``shape`` is ``(H, W)``; the on-disk folder name is ``WxH`` (Gemini's
    convention — e.g. ``1440x720`` for a 1440-wide, 720-tall image). We also
    fall back to ``HxW`` for forward-compat with any dataset that uses the
    flipped convention. Returns ``None`` if neither path exists, which is a
    valid case — not every model has every colour at every resolution.
    """
    if _ACTIVE_DATASET_ROOT is None:
        return None
    H, W = shape
    for name in (f"{W}x{H}", f"{H}x{W}"):
        candidate = os.path.join(_ACTIVE_DATASET_ROOT, model, color, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def _bind_root(root: str) -> None:
    """Set the process-wide dataset root used by :func:`_find_res_dir`."""
    global _ACTIVE_DATASET_ROOT
    _ACTIVE_DATASET_ROOT = os.path.abspath(root)


# Expose the binder on the class for convenience.
SpectralCodebookV4._bind_root = staticmethod(_bind_root)  # type: ignore[attr-defined]


__all__ = [
    "ALL_COLORS",
    "CONSENSUS_COLORS",
    "CONTENT_COLORS",
    "DEFAULT_MODEL",
    "UNION_MODEL",
    "CHANNEL_WEIGHTS",
    "ProfileV4",
    "SpectralCodebookV4",
    "SynthIDBypassV4",
]
