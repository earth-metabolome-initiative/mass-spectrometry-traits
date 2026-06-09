from collections import defaultdict
from pathlib import Path

import pytest

import mass_spectrometry


FIXTURES = Path(__file__).parent / "fixtures"


def parse_mgf(path):
    spectra = []
    current = None
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", ";", "!")):
            continue

        upper = line.upper()
        if upper == "BEGIN IONS":
            if current is not None:
                raise ValueError(f"nested BEGIN IONS at line {line_no}")
            current = {"metadata": {}, "peaks": []}
            continue

        if upper == "END IONS":
            if current is None:
                raise ValueError(f"END IONS without BEGIN IONS at line {line_no}")
            spectra.append(current)
            current = None
            continue

        if current is None:
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            current["metadata"][key.strip().upper()] = value.strip()
            continue

        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"invalid peak line {line_no}: {line}")
        current["peaks"].append((float(fields[0]), float(fields[1])))

    if current is not None:
        raise ValueError("unclosed BEGIN IONS block")

    return spectra


def test_splash_readme_example():
    assert (
        mass_spectrometry.splash([(100.0, 10.0), (200.0, 20.0)])
        == "splash10-0udi-0490000000-4425acda10ed7d4709bd"
    )


@pytest.mark.parametrize(
    ("spectrum_type", "prefix"),
    [
        ("mass", "splash10-"),
        ("nmr", "splash20-"),
        ("uv", "splash30-"),
        ("ir", "splash40-"),
        ("raman", "splash50-"),
    ],
)
def test_splash_spectrum_types(spectrum_type, prefix):
    assert mass_spectrometry.splash([(100.0, 10.0)], spectrum_type).startswith(prefix)


def test_splash_rejects_empty_peaks():
    with pytest.raises(ValueError, match="empty spectrum"):
        mass_spectrometry.splash([])


def test_splash_rejects_unknown_spectrum_type():
    with pytest.raises(ValueError, match="unknown spectrum_type"):
        mass_spectrometry.splash([(100.0, 10.0)], "unknown")


def test_splash_rejects_malformed_peaks():
    with pytest.raises(ValueError, match="sequence of \\(mz, intensity\\) pairs"):
        mass_spectrometry.splash([(100.0, 10.0, 5.0)])


def test_mgf_fixture_reports_duplicate_splashes():
    spectra = parse_mgf(FIXTURES / "duplicates.MGF")
    splashes = defaultdict(list)
    for spectrum in spectra:
        splash = mass_spectrometry.splash(spectrum["peaks"])
        splashes[splash].append(spectrum)

    duplicate_groups = {
        splash: group for splash, group in splashes.items() if len(group) > 1
    }

    assert len(spectra) == 6
    assert len(splashes) == 5
    assert duplicate_groups.keys() == {
        "splash10-0udi-0490000000-4425acda10ed7d4709bd"
    }
    assert [
        spectrum["metadata"]["FEATURE_ID"]
        for spectrum in duplicate_groups["splash10-0udi-0490000000-4425acda10ed7d4709bd"]
    ] == ["1", "6"]
