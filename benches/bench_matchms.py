#!/usr/bin/env -S uv run --no-config --isolated --with matchms --with "numpy<2"
"""Benchmark matchms ModifiedCosine on the same spectra used by the Rust criterion
benchmarks, so we can compare wall-clock times directly.

Spectra are extracted at import time from the Rust source files that live next to
this script.  The three reference spectra are:

    salicin             (21 peaks,  precursor_mz=321.075)
    hydroxy_cholesterol (131 peaks, precursor_mz=385.345)
    epimeloscine        (5379 peaks, precursor_mz=747.832)

Each pair is timed with timeit; we report the median of 5 rounds.
"""

from __future__ import annotations

import re
import timeit
from pathlib import Path

import numpy as np
from matchms import Spectrum
from matchms.similarity import CosineGreedy, CosineHungarian, ModifiedCosine

# ---------------------------------------------------------------------------
# Helpers to parse Rust source arrays
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src" / "traits" / "reference_spectra"

_NUM_RE = re.compile(r"[-+]?\d[\d_]*(?:\.[\d_]+)?(?:[eE][-+]?\d+)?")


def _parse_rust_array(text: str) -> list[float]:
    """Extract all numeric literals from a Rust array literal."""
    return [float(tok.replace("_", "")) for tok in _NUM_RE.findall(text)]


def _load_spectrum(filename: str, mz_var: str, int_var: str, prec_var: str) -> Spectrum:
    src = (_SRC / filename).read_text()

    # Extract precursor_mz constant value
    m = re.search(rf"{prec_var}:\s*f32\s*=\s*([\d._]+)", src)
    assert m, f"cannot find {prec_var}"
    precursor_mz = float(m.group(1).replace("_", ""))

    # Extract mz array — body between '[' and '];' after the var declaration
    mz_start = src.index(mz_var)
    mz_bracket = src.index("[", mz_start)
    mz_body = src[mz_bracket + 1 : src.index("];", mz_bracket)]
    # Skip the first token if it looks like an array length (e.g. "f32; 21")
    mz_body = re.sub(r"^[^]]*\]\s*=\s*\[", "", src[mz_start : src.index("];", mz_start)])
    mz = np.array(_parse_rust_array(mz_body), dtype=np.float64)

    int_start = src.index(int_var)
    int_body = re.sub(r"^[^]]*\]\s*=\s*\[", "", src[int_start : src.index("];", int_start)])
    intensities = np.array(_parse_rust_array(int_body), dtype=np.float64)

    assert len(mz) == len(intensities), f"{filename}: mz/int length mismatch"

    return Spectrum(
        mz=mz,
        intensities=intensities,
        metadata={"precursor_mz": precursor_mz},
    )


# ---------------------------------------------------------------------------
# Load spectra
# ---------------------------------------------------------------------------

salicin = _load_spectrum(
    "salicin.rs", "SALICIN_MZ", "SALICIN_INTENSITIES", "SALICIN_PRECURSOR_MZ"
)
hydroxy_cholesterol = _load_spectrum(
    "hydroxycholesterol.rs",
    "HYDROXY_CHOLESTEROL_MZ",
    "HYDROXY_CHOLESTEROL_INTENSITIES",
    "HYDROXY_CHOLESTEROL_PRECURSOR_MZ",
)
epimeloscine = _load_spectrum(
    "epimeloscine.rs",
    "EPIMELOSCINE_MZ",
    "EPIMELOSCINE_INTENSITIES",
    "EPIMELOSCINE_PRECURSOR_MZ",
)

print(f"salicin:             {len(salicin.peaks.mz):>5} peaks")
print(f"hydroxy_cholesterol: {len(hydroxy_cholesterol.peaks.mz):>5} peaks")
print(f"epimeloscine:        {len(epimeloscine.peaks.mz):>5} peaks")
print()

# ---------------------------------------------------------------------------
# Benchmark pairs
# ---------------------------------------------------------------------------

SCORERS = {
    "CosineGreedy": CosineGreedy(tolerance=0.1),
    "ModifiedCosine (greedy)": ModifiedCosine(tolerance=0.1),
    "CosineHungarian": CosineHungarian(tolerance=0.1),
}

PAIRS: list[tuple[str, Spectrum, Spectrum]] = [
    ("hydroxy_cholesterol vs salicin", hydroxy_cholesterol, salicin),
    ("hydroxy_cholesterol vs self", hydroxy_cholesterol, hydroxy_cholesterol),
    ("salicin vs self", salicin, salicin),
    ("salicin vs epimeloscine", salicin, epimeloscine),
    ("hydroxy_cholesterol vs epimeloscine", hydroxy_cholesterol, epimeloscine),
    ("epimeloscine vs self", epimeloscine, epimeloscine),
]


def _format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:>10.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:>10.2f} ms"
    return f"{seconds:>10.2f}  s"


for scorer_name, scorer in SCORERS.items():
    print(f"\n=== {scorer_name} ===")
    print(f"{'Pair':<42} {'median':>12}  {'min':>12}  iters")
    print("-" * 90)

    def _bench_pair(left: Spectrum, right: Spectrum) -> None:
        scorer.pair(left, right)

    for label, left, right in PAIRS:
        # Warmup
        for _ in range(3):
            _bench_pair(left, right)
        # Adaptive iteration count: aim for ~2 s total per benchmark
        single = timeit.timeit(lambda l=left, r=right: _bench_pair(l, r), number=10) / 10
        n = max(10, int(2.0 / single))

        rounds = 5
        times = []
        for _ in range(rounds):
            t = timeit.timeit(lambda l=left, r=right: _bench_pair(l, r), number=n) / n
            times.append(t)
        times.sort()
        median = times[len(times) // 2]
        minimum = times[0]
        print(f"{label:<42} {_format_time(median)}  {_format_time(minimum)}  n={n}")
