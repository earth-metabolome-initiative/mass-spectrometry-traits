#!/usr/bin/env -S uv run --no-config --isolated --with ms_entropy --with "numpy<2"
"""Generate ms_entropy baselines for LinearEntropy reference validation.

This script emits `expected_entropy_similarities.csv` using upstream
`ms_entropy` with `clean_spectra=True`, so cleaning/centroiding behavior is
part of the reference.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ms_entropy import (
    __version__ as ms_entropy_version,
    calculate_entropy_similarity,
    calculate_unweighted_entropy_similarity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "src" / "traits" / "reference_spectra"
OUTPUT_CSV = Path(__file__).resolve().parent / "expected_entropy_similarities.csv"

TOLERANCE_DA = 0.1

NUMBER_RE = re.compile(r"[-+]?\d[\d_]*(?:\.[\d_]+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class SpectrumSource:
    name: str
    filename: str
    mz_var: str
    int_var: str


SPECTRA: tuple[SpectrumSource, ...] = (
    SpectrumSource("aspirin", "aspirin.rs", "ASPIRIN_MZ", "ASPIRIN_INTENSITIES"),
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
    return np.column_stack((mz, intensities)).astype(np.float32)


def main() -> None:
    peaks_by_name = {spec.name: _load_peaks(spec) for spec in SPECTRA}
    names = [spec.name for spec in SPECTRA]

    rows: list[dict[str, str]] = []
    for weighted in (True, False):
        for i, left_name in enumerate(names):
            for right_name in names[i:]:
                left = peaks_by_name[left_name]
                right = peaks_by_name[right_name]
                if weighted:
                    score = calculate_entropy_similarity(
                        left,
                        right,
                        ms2_tolerance_in_da=TOLERANCE_DA,
                        clean_spectra=True,
                    )
                else:
                    score = calculate_unweighted_entropy_similarity(
                        left,
                        right,
                        ms2_tolerance_in_da=TOLERANCE_DA,
                        clean_spectra=True,
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
