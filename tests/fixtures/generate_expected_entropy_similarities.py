#!/usr/bin/env -S uv run --no-config --isolated --with ms_entropy --with "numpy<2"
"""Generate ms_entropy baselines for LinearEntropy reference validation.

This script auto-discovers all reference spectra from
src/traits/reference_spectra/*.rs and generates two CSV files:

  - expected_entropy_similarities.csv  — pairwise similarity scores
  - expected_cleaned_spectra.csv       — per-spectrum cleaned peak lists

Both use `min_ms2_difference_in_da = 2 * tolerance` for each tolerance,
matching the Rust test configuration.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ms_entropy import (
    __version__ as ms_entropy_version,
    calculate_entropy_similarity,
    calculate_unweighted_entropy_similarity,
    clean_spectrum,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "src" / "traits" / "reference_spectra"
OUTPUT_DIR = Path(__file__).resolve().parent
SIMILARITY_CSV = OUTPUT_DIR / "expected_entropy_similarities.csv"
CLEANED_CSV = OUTPUT_DIR / "expected_cleaned_spectra.csv"

TOLERANCES = [0.01, 0.05, 0.1, 0.5, 2.0]

NUMBER_RE = re.compile(r"[-+]?\d[\d_]*(?:\.[\d_]+)?(?:[eE][-+]?\d+)?")
PREFIX_RE = re.compile(r"pub const (\w+)_MZ:\s*\[f32;")


@dataclass(frozen=True)
class SpectrumSource:
    name: str  # canonical name (from filename)
    filename: str
    mz_var: str  # e.g. ASPIRIN_MZ
    int_var: str  # e.g. ASPIRIN_INTENSITIES


def discover_spectra() -> list[SpectrumSource]:
    """Auto-discover all reference spectra from .rs files."""
    spectra = []
    for rs_file in sorted(REFERENCE_DIR.glob("*.rs")):
        if rs_file.name == "mod.rs":
            continue
        src = rs_file.read_text(encoding="utf-8")
        match = PREFIX_RE.search(src)
        if match is None:
            print(f"  Skipping {rs_file.name}: no MZ constant found", file=sys.stderr)
            continue
        prefix = match.group(1)
        canonical_name = rs_file.stem  # filename sans .rs
        spectra.append(
            SpectrumSource(
                name=canonical_name,
                filename=rs_file.name,
                mz_var=f"{prefix}_MZ",
                int_var=f"{prefix}_INTENSITIES",
            )
        )
    return spectra


def _parse_array(src: str, var_name: str) -> np.ndarray:
    pattern = rf"{var_name}\s*:\s*\[f32;\s*\d+\]\s*=\s*\[(.*?)\];"
    match = re.search(pattern, src, re.S)
    if match is None:
        raise ValueError(f"Unable to locate array for {var_name}")
    values = [float(tok.replace("_", "")) for tok in NUMBER_RE.findall(match.group(1))]
    return np.asarray(values, dtype=np.float32)


def _load_peaks(spec: SpectrumSource) -> np.ndarray:
    src = (REFERENCE_DIR / spec.filename).read_text(encoding="utf-8")
    mz = _parse_array(src, spec.mz_var)
    intensities = _parse_array(src, spec.int_var)
    if mz.shape != intensities.shape:
        raise ValueError(f"{spec.name}: mz/intensity length mismatch")
    return np.column_stack((mz, intensities)).astype(np.float32)


def format_peaks(peaks: np.ndarray) -> str:
    """Format peaks as semicolon-separated mz:intensity pairs."""
    if peaks is None or len(peaks) == 0:
        return ""
    return ";".join(f"{mz:.12f}:{intensity:.12f}" for mz, intensity in peaks)


def main() -> None:
    spectra = discover_spectra()
    print(f"Discovered {len(spectra)} reference spectra")

    peaks_by_name: dict[str, np.ndarray] = {}
    for spec in spectra:
        try:
            peaks_by_name[spec.name] = _load_peaks(spec)
        except ValueError as e:
            print(f"  Error loading {spec.name}: {e}", file=sys.stderr)

    names = [spec.name for spec in spectra if spec.name in peaks_by_name]
    print(f"Loaded {len(names)} spectra successfully")

    n_pairs = len(names) * (len(names) + 1) // 2

    # --- Pre-clean spectra per tolerance ---
    # calculate_entropy_similarity doesn't accept min_ms2_difference_in_da,
    # so we pre-clean and pass clean_spectra=False.
    cleaned_cache: dict[tuple[str, float], np.ndarray] = {}
    for tolerance in TOLERANCES:
        min_diff = 2.0 * tolerance
        for name in names:
            cleaned = clean_spectrum(
                peaks_by_name[name],
                min_ms2_difference_in_da=min_diff,
                noise_threshold=0.01,
            )
            cleaned_cache[(name, tolerance)] = cleaned
        print(f"  Pre-cleaned {len(names)} spectra for tolerance={tolerance}")

    # --- Generate similarity CSV ---
    sim_rows: list[dict[str, str]] = []
    for tolerance in TOLERANCES:
        min_diff = 2.0 * tolerance
        for weighted in (True, False):
            count = 0
            for i, left_name in enumerate(names):
                for right_name in names[i:]:
                    left = cleaned_cache[(left_name, tolerance)]
                    right = cleaned_cache[(right_name, tolerance)]
                    if weighted:
                        score = calculate_entropy_similarity(
                            left,
                            right,
                            ms2_tolerance_in_da=tolerance,
                            clean_spectra=False,
                        )
                    else:
                        score = calculate_unweighted_entropy_similarity(
                            left,
                            right,
                            ms2_tolerance_in_da=tolerance,
                            clean_spectra=False,
                        )
                    sim_rows.append(
                        {
                            "left_name": left_name,
                            "right_name": right_name,
                            "tolerance": f"{tolerance:.6f}",
                            "min_ms2_difference_in_da": f"{min_diff:.6f}",
                            "weighted": str(weighted).lower(),
                            "expected_score": f"{float(score):.12f}",
                            "ms_entropy_version": ms_entropy_version,
                            "numpy_version": np.__version__,
                        }
                    )
                    count += 1
            label = "weighted" if weighted else "unweighted"
            print(f"  tolerance={tolerance}, {label}: {count}/{n_pairs} pairs")

    with SIMILARITY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "left_name",
                "right_name",
                "tolerance",
                "min_ms2_difference_in_da",
                "weighted",
                "expected_score",
                "ms_entropy_version",
                "numpy_version",
            ],
        )
        writer.writeheader()
        writer.writerows(sim_rows)
    print(f"Wrote {len(sim_rows)} similarity rows to {SIMILARITY_CSV}")

    # --- Generate cleaned spectra CSV ---
    clean_rows: list[dict[str, str]] = []
    for tolerance in TOLERANCES:
        min_diff = 2.0 * tolerance
        for name in names:
            cleaned = cleaned_cache[(name, tolerance)]
            num_peaks = len(cleaned) if cleaned is not None else 0
            clean_rows.append(
                {
                    "name": name,
                    "min_ms2_difference_in_da": f"{min_diff:.6f}",
                    "num_peaks": str(num_peaks),
                    "peaks": format_peaks(cleaned),
                    "ms_entropy_version": ms_entropy_version,
                    "numpy_version": np.__version__,
                }
            )
        print(f"  tolerance={tolerance}: cleaned {len(names)} spectra")

    with CLEANED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "min_ms2_difference_in_da",
                "num_peaks",
                "peaks",
                "ms_entropy_version",
                "numpy_version",
            ],
        )
        writer.writeheader()
        writer.writerows(clean_rows)
    print(f"Wrote {len(clean_rows)} cleaning rows to {CLEANED_CSV}")


if __name__ == "__main__":
    main()
