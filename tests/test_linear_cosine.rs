//! Tests for the LinearCosine similarity implementation.
//!
//! LinearCosine must produce identical results to HungarianCosine on
//! well-separated spectra (min peak gap > 2 * tolerance).

use mass_spectrometry::prelude::{
    CocaineSpectrum, GenericSpectrum, GlucoseSpectrum, HungarianCosine, HydroxyCholesterolSpectrum,
    LinearCosine, PhenylalanineSpectrum, SalicinSpectrum, ScalarSimilarity,
    SimilarityComputationError, Spectrum, SpectrumAlloc, SpectrumMut,
};

fn cosine() -> LinearCosine<f32, f32> {
    LinearCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn hungarian() -> HungarianCosine<f32, f32> {
    HungarianCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn make_spectrum(precursor: f32, peaks: &[(f32, f32)]) -> GenericSpectrum<f32, f32> {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor, peaks.len()).expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum.add_peak(mz, intensity).expect("valid sorted peak");
    }
    spectrum
}

fn assert_self_similarity(name: &str, spectrum: &GenericSpectrum<f32, f32>) {
    let (sim, peaks) = cosine()
        .similarity(spectrum, spectrum)
        .expect("similarity computation should succeed");
    assert!(
        (1.0_f32 - sim).abs() < 1e-6,
        "{name} self-similarity: expected ~1.0, got {sim}"
    );
    assert_eq!(peaks, spectrum.len());
}

// ---------- self-similarity (score must be ~1.0, matches = #peaks) ----------

#[test]
fn self_similarity_cocaine() {
    assert_self_similarity(
        "cocaine",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
    );
}

#[test]
fn self_similarity_glucose() {
    assert_self_similarity(
        "glucose",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
    );
}

#[test]
fn self_similarity_hydroxy_cholesterol() {
    assert_self_similarity(
        "hydroxy_cholesterol",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
    );
}

#[test]
fn self_similarity_salicin() {
    assert_self_similarity(
        "salicin",
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
    );
}

#[test]
fn self_similarity_phenylalanine() {
    assert_self_similarity(
        "phenylalanine",
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

// ---------- cross-similarity equivalence with HungarianCosine ----------

fn assert_matches_hungarian(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let (linear_score, linear_matches) = cosine()
        .similarity(left, right)
        .expect("LinearCosine similarity should succeed");
    let (hungarian_score, hungarian_matches) = hungarian()
        .similarity(left, right)
        .expect("HungarianCosine similarity should succeed");

    assert!(
        (linear_score - hungarian_score).abs() < 1e-6,
        "{name}: LinearCosine={linear_score} vs HungarianCosine={hungarian_score}"
    );
    assert_eq!(
        linear_matches, hungarian_matches,
        "{name}: LinearCosine matches={linear_matches} vs HungarianCosine matches={hungarian_matches}"
    );
}

#[test]
fn equivalence_cocaine_glucose() {
    assert_matches_hungarian(
        "cocaine_glucose",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_cocaine_hydroxy_cholesterol() {
    assert_matches_hungarian(
        "cocaine_hc",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_cocaine_salicin() {
    assert_matches_hungarian(
        "cocaine_salicin",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_cocaine_phenylalanine() {
    assert_matches_hungarian(
        "cocaine_phe",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_glucose_hydroxy_cholesterol() {
    assert_matches_hungarian(
        "glucose_hc",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_glucose_salicin() {
    assert_matches_hungarian(
        "glucose_salicin",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_glucose_phenylalanine() {
    assert_matches_hungarian(
        "glucose_phe",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_hydroxy_cholesterol_salicin() {
    assert_matches_hungarian(
        "hc_salicin",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_hydroxy_cholesterol_phenylalanine() {
    assert_matches_hungarian(
        "hc_phe",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

#[test]
fn equivalence_salicin_phenylalanine() {
    assert_matches_hungarian(
        "salicin_phe",
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

// ---------- symmetry: sim(A, B) == sim(B, A) ----------

fn assert_symmetry(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let c = cosine();
    let (score_ab, matches_ab) = c
        .similarity(left, right)
        .expect("similarity computation should succeed");
    let (score_ba, matches_ba) = c
        .similarity(right, left)
        .expect("similarity computation should succeed");
    assert!(
        (score_ab - score_ba).abs() < 1e-6,
        "{name}: sim(A,B)={score_ab} != sim(B,A)={score_ba}"
    );
    assert_eq!(
        matches_ab, matches_ba,
        "{name}: matches(A,B)={matches_ab} != matches(B,A)={matches_ba}"
    );
}

#[test]
fn symmetry_cocaine_glucose() {
    assert_symmetry(
        "cocaine_glucose",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
    );
}

#[test]
fn symmetry_hydroxy_cholesterol_phenylalanine() {
    assert_symmetry(
        "hc_phe",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

#[test]
fn boundary_gap_equal_2x_tolerance_returns_error() {
    // Binary-exact boundary: 2 * 0.125 == 0.25 exactly in f32/f64.
    let left = make_spectrum(200.0, &[(100.0, 10.0), (100.25, 8.0)]);
    let right = make_spectrum(200.0, &[(100.0, 10.0), (100.25, 8.0)]);
    let linear = LinearCosine::new(1.0_f32, 1.0_f32, 0.125_f32).expect("valid scorer config");

    let error = linear
        .similarity(&left, &right)
        .expect_err("boundary-equal spacing should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::InvalidPeakSpacing("left spectrum")
    );
}

#[test]
fn boundary_gap_strictly_above_2x_tolerance_succeeds() {
    let left = make_spectrum(200.0, &[(100.0, 10.0), (100.2501, 8.0)]);
    let right = make_spectrum(200.0, &[(100.0, 10.0), (100.2501, 8.0)]);
    let linear = LinearCosine::new(1.0_f32, 1.0_f32, 0.125_f32).expect("valid scorer config");

    let (score, matches) = linear
        .similarity(&left, &right)
        .expect("strictly separated spectra should be accepted");
    assert!((1.0_f32 - score).abs() < 1e-6);
    assert_eq!(matches, 2);
}
