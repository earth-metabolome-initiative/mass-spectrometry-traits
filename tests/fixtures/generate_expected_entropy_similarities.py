#!/usr/bin/env -S uv run --no-config --isolated --with ms_entropy --with "numpy<2"
"""Generate ms_entropy baselines for LinearEntropy reference validation.

This script emits `expected_entropy_similarities.csv` for a curated set of
reference spectra that satisfy LinearEntropy's strict spacing precondition at
`tolerance = 0.1` (consecutive peaks must be > 0.2 apart).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ms_entropy import (
    __version__ as ms_entropy_version,
    calculate_unweighted_entropy_similarity,
    apply_weight_to_intensity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "src" / "traits" / "reference_spectra"
OUTPUT_CSV = Path(__file__).resolve().parent / "expected_entropy_similarities.csv"

TOLERANCE_DA = 0.1
MIN_GAP_REQUIRED = 2.0 * TOLERANCE_DA

NUMBER_RE = re.compile(r"[-+]?\d[\d_]*(?:\.[\d_]+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class SpectrumSource:
    name: str
    filename: str
    mz_var: str
    int_var: str


SPECTRA: tuple[SpectrumSource, ...] = (
    SpectrumSource("cocaine", "cocaine.rs", "COCAINE_MZ", "COCAINE_INTENSITIES"),
    SpectrumSource("glucose", "glucose.rs", "GLUCOSE_MZ", "GLUCOSE_INTENSITIES"),
    SpectrumSource("hydroxy_cholesterol", "hydroxycholesterol.rs", "HYDROXY_CHOLESTEROL_MZ", "HYDROXY_CHOLESTEROL_INTENSITIES"),
    SpectrumSource("salicin", "salicin.rs", "SALICIN_MZ", "SALICIN_INTENSITIES"),
    SpectrumSource("phenylalanine", "phenylalanine.rs", "PHENYLANINE_MZ", "PHENYLANINE_INTENSITIES"),
)


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
    peaks = np.column_stack((mz, intensities)).astype(np.float32)
    if peaks.shape[0] > 1:
        min_gap = float(np.min(np.diff(peaks[:, 0])))
        if not min_gap > MIN_GAP_REQUIRED:
            raise ValueError(
                f"{spec.name}: min m/z gap {min_gap:.6f} does not satisfy "
                f"> {MIN_GAP_REQUIRED:.6f} required for LinearEntropy"
            )
    return peaks


def _normalize_peaks(peaks: np.ndarray) -> np.ndarray:
    normalized = np.array(peaks, dtype=np.float32, copy=True)
    intensity_sum = float(np.sum(normalized[:, 1], dtype=np.float64))
    if intensity_sum <= 0.0:
        raise ValueError("Spectrum has non-positive total intensity")
    normalized[:, 1] /= intensity_sum
    return normalized


def main() -> None:
    normalized_peaks_by_name = {
        spec.name: _normalize_peaks(_load_peaks(spec)) for spec in SPECTRA
    }
    names = [spec.name for spec in SPECTRA]

    rows: list[dict[str, str]] = []
    for weighted in (True, False):
        for i, left_name in enumerate(names):
            for right_name in names[i:]:
                left = normalized_peaks_by_name[left_name]
                right = normalized_peaks_by_name[right_name]
                if weighted:
                    score = calculate_unweighted_entropy_similarity(
                        apply_weight_to_intensity(np.array(left, copy=True)),
                        apply_weight_to_intensity(np.array(right, copy=True)),
                        ms2_tolerance_in_da=TOLERANCE_DA,
                        clean_spectra=False,
                    )
                else:
                    score = calculate_unweighted_entropy_similarity(
                        left,
                        right,
                        ms2_tolerance_in_da=TOLERANCE_DA,
                        clean_spectra=False,
                    )

                rows.append(
                    {
                        "left_name": left_name,
                        "right_name": right_name,
                        "tolerance": f"{TOLERANCE_DA:.6f}",
                        "weighted": str(weighted).lower(),
                        "expected_score": f"{float(score):.12f}",
                        "ms_entropy_version": ms_entropy_version,
                        "numpy_version": np.__version__,
                    }
                )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "left_name",
                "right_name",
                "tolerance",
                "weighted",
                "expected_score",
                "ms_entropy_version",
                "numpy_version",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
