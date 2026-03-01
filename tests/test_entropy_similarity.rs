//! Tests for the EntropySimilarity implementation.
//!
//! Reference values are computed with a manual Python implementation of the
//! spectral entropy algorithm (Li et al., Nature Methods 2021), validated
//! against the `ms_entropy` package (with `clean_spectra=False`).

use mass_spectrometry::prelude::{
    AspirinSpectrum, CocaineSpectrum, EntropySimilarity, GenericSpectrum, GlucoseSpectrum,
    HydroxyCholesterolSpectrum, PhenylalanineSpectrum, SalicinSpectrum, ScalarSimilarity, Spectrum,
};

fn weighted() -> EntropySimilarity<f32> {
    EntropySimilarity::weighted(0.1).expect("valid scorer config")
}

fn unweighted() -> EntropySimilarity<f32> {
    EntropySimilarity::unweighted(0.1).expect("valid scorer config")
}

fn assert_self_similarity(
    name: &str,
    spectrum: &GenericSpectrum<f32, f32>,
    scorer: &EntropySimilarity<f32>,
) {
    let (sim, peaks) = scorer
        .similarity(spectrum, spectrum)
        .expect("similarity computation should succeed");
    assert!(
        (1.0_f32 - sim).abs() < 1e-5,
        "{name} self-similarity: expected ~1.0, got {sim}"
    );
    assert_eq!(peaks, spectrum.len());
}

fn assert_cross(
    name: &str,
    a: &GenericSpectrum<f32, f32>,
    b: &GenericSpectrum<f32, f32>,
    scorer: &EntropySimilarity<f32>,
    expected_sim: f32,
    expected_matches: usize,
    tol: f32,
) {
    let (sim, peaks) = scorer
        .similarity(a, b)
        .expect("similarity computation should succeed");
    assert!(
        (sim - expected_sim).abs() < tol,
        "{name}: expected {expected_sim}, got {sim} (diff={})",
        (sim - expected_sim).abs()
    );
    assert_eq!(peaks, expected_matches, "{name}: wrong match count");
}

fn assert_symmetry(
    name: &str,
    a: &GenericSpectrum<f32, f32>,
    b: &GenericSpectrum<f32, f32>,
    scorer: &EntropySimilarity<f32>,
) {
    let (sim_ab, peaks_ab) = scorer
        .similarity(a, b)
        .expect("similarity computation should succeed");
    let (sim_ba, peaks_ba) = scorer
        .similarity(b, a)
        .expect("similarity computation should succeed");
    assert!(
        (sim_ab - sim_ba).abs() < 1e-6,
        "{name} symmetry: {sim_ab} vs {sim_ba}"
    );
    assert_eq!(peaks_ab, peaks_ba, "{name} symmetry: peak count mismatch");
}

// ========== Weighted self-similarity ==========

#[test]
fn weighted_self_cocaine() {
    assert_self_similarity(
        "cocaine",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &weighted(),
    );
}

#[test]
fn weighted_self_glucose() {
    assert_self_similarity(
        "glucose",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &weighted(),
    );
}

#[test]
fn weighted_self_aspirin() {
    assert_self_similarity(
        "aspirin",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &weighted(),
    );
}

#[test]
fn weighted_self_hydroxy_cholesterol() {
    assert_self_similarity(
        "hydroxy_cholesterol",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &weighted(),
    );
}

#[test]
fn weighted_self_salicin() {
    assert_self_similarity(
        "salicin",
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &weighted(),
    );
}

#[test]
fn weighted_self_phenylalanine() {
    assert_self_similarity(
        "phenylalanine",
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &weighted(),
    );
}

// ========== Unweighted self-similarity ==========

#[test]
fn unweighted_self_cocaine() {
    assert_self_similarity(
        "cocaine",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &unweighted(),
    );
}

#[test]
fn unweighted_self_glucose() {
    assert_self_similarity(
        "glucose",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &unweighted(),
    );
}

#[test]
fn unweighted_self_aspirin() {
    assert_self_similarity(
        "aspirin",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &unweighted(),
    );
}

#[test]
fn unweighted_self_hydroxy_cholesterol() {
    assert_self_similarity(
        "hydroxy_cholesterol",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &unweighted(),
    );
}

#[test]
fn unweighted_self_salicin() {
    assert_self_similarity(
        "salicin",
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &unweighted(),
    );
}

#[test]
fn unweighted_self_phenylalanine() {
    assert_self_similarity(
        "phenylalanine",
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &unweighted(),
    );
}

// ========== Weighted cross-similarity (Python reference, no cleaning) ==========

#[test]
fn weighted_cocaine_vs_glucose() {
    assert_cross(
        "cocaine_vs_glucose",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &weighted(),
        0.0,
        0,
        1e-6,
    );
}

#[test]
fn weighted_cocaine_vs_aspirin() {
    assert_cross(
        "cocaine_vs_aspirin",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &weighted(),
        0.026_026_2,
        1,
        1e-4,
    );
}

#[test]
fn weighted_cocaine_vs_hydroxy_cholesterol() {
    assert_cross(
        "cocaine_vs_hydroxy_cholesterol",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &weighted(),
        0.0,
        0,
        1e-6,
    );
}

#[test]
fn weighted_cocaine_vs_salicin() {
    assert_cross(
        "cocaine_vs_salicin",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &weighted(),
        0.0,
        0,
        1e-6,
    );
}

#[test]
fn weighted_cocaine_vs_phenylalanine() {
    assert_cross(
        "cocaine_vs_phenylalanine",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &weighted(),
        0.0,
        0,
        1e-6,
    );
}

#[test]
fn weighted_glucose_vs_aspirin() {
    assert_cross(
        "glucose_vs_aspirin",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &weighted(),
        0.043_315_06,
        1,
        1e-4,
    );
}

#[test]
fn weighted_glucose_vs_hydroxy_cholesterol() {
    assert_cross(
        "glucose_vs_hydroxy_cholesterol",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &weighted(),
        0.037_349_56,
        6,
        1e-4,
    );
}

#[test]
fn weighted_glucose_vs_salicin() {
    assert_cross(
        "glucose_vs_salicin",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &weighted(),
        0.0,
        0,
        1e-6,
    );
}

#[test]
fn weighted_glucose_vs_phenylalanine() {
    assert_cross(
        "glucose_vs_phenylalanine",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &weighted(),
        0.055_385_83,
        2,
        1e-4,
    );
}

#[test]
fn weighted_aspirin_vs_hydroxy_cholesterol() {
    assert_cross(
        "aspirin_vs_hydroxy_cholesterol",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &weighted(),
        0.053_310_06,
        8,
        1e-4,
    );
}

#[test]
fn weighted_aspirin_vs_salicin() {
    assert_cross(
        "aspirin_vs_salicin",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &weighted(),
        0.005_232_96,
        1,
        1e-4,
    );
}

#[test]
fn weighted_aspirin_vs_phenylalanine() {
    assert_cross(
        "aspirin_vs_phenylalanine",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &weighted(),
        0.229_214_96,
        3,
        1e-4,
    );
}

#[test]
fn weighted_hydroxy_cholesterol_vs_salicin() {
    assert_cross(
        "hydroxy_cholesterol_vs_salicin",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &weighted(),
        0.0,
        0,
        1e-6,
    );
}

#[test]
fn weighted_hydroxy_cholesterol_vs_phenylalanine() {
    assert_cross(
        "hydroxy_cholesterol_vs_phenylalanine",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &weighted(),
        0.028_153_34,
        3,
        1e-4,
    );
}

#[test]
fn weighted_salicin_vs_phenylalanine() {
    assert_cross(
        "salicin_vs_phenylalanine",
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &weighted(),
        0.021_864_35,
        1,
        1e-4,
    );
}

// ========== Unweighted cross-similarity ==========

#[test]
fn unweighted_cocaine_vs_aspirin() {
    assert_cross(
        "cocaine_vs_aspirin",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &unweighted(),
        0.010_192_22,
        1,
        1e-4,
    );
}

#[test]
fn unweighted_glucose_vs_aspirin() {
    assert_cross(
        "glucose_vs_aspirin",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &unweighted(),
        0.027_876_1,
        1,
        1e-4,
    );
}

#[test]
fn unweighted_glucose_vs_phenylalanine() {
    assert_cross(
        "glucose_vs_phenylalanine",
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &unweighted(),
        0.007_554_79,
        2,
        1e-4,
    );
}

#[test]
fn unweighted_aspirin_vs_hydroxy_cholesterol() {
    assert_cross(
        "aspirin_vs_hydroxy_cholesterol",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &unweighted(),
        0.038_041_51,
        8,
        1e-4,
    );
}

#[test]
fn unweighted_aspirin_vs_phenylalanine() {
    assert_cross(
        "aspirin_vs_phenylalanine",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
        &unweighted(),
        0.098_100_87,
        3,
        1e-4,
    );
}

// ========== Symmetry ==========

#[test]
fn weighted_symmetry() {
    let scorer = weighted();
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let hc = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let phe = GenericSpectrum::phenylalanine().expect("reference spectrum should build");
    let salicin = GenericSpectrum::salicin().expect("reference spectrum should build");

    assert_symmetry("cocaine_aspirin", &cocaine, &aspirin, &scorer);
    assert_symmetry("glucose_aspirin", &glucose, &aspirin, &scorer);
    assert_symmetry("aspirin_hc", &aspirin, &hc, &scorer);
    assert_symmetry("aspirin_phe", &aspirin, &phe, &scorer);
    assert_symmetry("glucose_hc", &glucose, &hc, &scorer);
    assert_symmetry("glucose_phe", &glucose, &phe, &scorer);
    assert_symmetry("hc_phe", &hc, &phe, &scorer);
    assert_symmetry("salicin_phe", &salicin, &phe, &scorer);
}

#[test]
fn unweighted_symmetry() {
    let scorer = unweighted();
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let hc = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let phe = GenericSpectrum::phenylalanine().expect("reference spectrum should build");

    assert_symmetry("cocaine_aspirin", &cocaine, &aspirin, &scorer);
    assert_symmetry("glucose_aspirin", &glucose, &aspirin, &scorer);
    assert_symmetry("aspirin_hc", &aspirin, &hc, &scorer);
    assert_symmetry("aspirin_phe", &aspirin, &phe, &scorer);
    assert_symmetry("glucose_phe", &glucose, &phe, &scorer);
}
