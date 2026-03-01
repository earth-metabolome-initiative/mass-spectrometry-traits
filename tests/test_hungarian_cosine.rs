//! Tests for the HungarianCosine similarity implementation.
//!
//! Reference values are computed with matchms CosineHungarian
//! (tolerance=0.1, mz_power=1.0, intensity_power=1.0).

use mass_spectrometry::prelude::{
    AspirinSpectrum, CocaineSpectrum, GenericSpectrum, GlucoseSpectrum, HungarianCosine,
    HydroxyCholesterolSpectrum, PhenylalanineSpectrum, SalicinSpectrum, ScalarSimilarity, Spectrum,
};

fn cosine() -> HungarianCosine<f32, f32> {
    HungarianCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn assert_self_similarity(name: &str, spectrum: &GenericSpectrum<f32, f32>) {
    let (sim, peaks) = cosine()
        .similarity(spectrum, spectrum)
        .expect("similarity computation should succeed");
    // f32 self-similarity may not be exactly 1.0 because sqrt(x)*sqrt(x) != x
    // in floating point; 1 - sim should be within f32 machine epsilon (~1.2e-7).
    assert!(
        (1.0_f32 - sim).abs() < 1e-6,
        "{name} self-similarity: expected ~1.0, got {sim}"
    );
    assert_eq!(peaks, spectrum.len());
}

// ---------- self-similarity (score must be ~1.0, matches = #peaks) ----------

#[test]
fn self_similarity_cocaine() {
    assert_self_similarity("cocaine", &GenericSpectrum::cocaine().expect("reference spectrum should build"));
}

#[test]
fn self_similarity_glucose() {
    assert_self_similarity("glucose", &GenericSpectrum::glucose().expect("reference spectrum should build"));
}

#[test]
fn self_similarity_aspirin() {
    assert_self_similarity("aspirin", &GenericSpectrum::aspirin().expect("reference spectrum should build"));
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
    assert_self_similarity("salicin", &GenericSpectrum::salicin().expect("reference spectrum should build"));
}

#[test]
fn self_similarity_phenylalanine() {
    assert_self_similarity("phenylalanine", &GenericSpectrum::phenylalanine().expect("reference spectrum should build"));
}

// ---------- cross-similarity (matchms CosineHungarian reference) ----------

#[test]
fn cocaine_vs_glucose() {
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&cocaine, &glucose)
        .expect("similarity computation should succeed");
    assert!(sim < 1e-9, "cocaine vs glucose: {sim}");
    assert_eq!(peaks, 0);
}

#[test]
fn cocaine_vs_aspirin() {
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&cocaine, &aspirin)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.000_183_533_03).abs() < 1e-6,
        "cocaine vs aspirin: {sim}"
    );
    assert_eq!(peaks, 1);
}

#[test]
fn cocaine_vs_hydroxy_cholesterol() {
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let hc = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&cocaine, &hc)
        .expect("similarity computation should succeed");
    assert!(sim < 1e-9, "cocaine vs hydroxycholesterol: {sim}");
    assert_eq!(peaks, 0);
}

#[test]
fn cocaine_vs_salicin() {
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let salicin = GenericSpectrum::salicin().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&cocaine, &salicin)
        .expect("similarity computation should succeed");
    assert!(sim < 1e-9, "cocaine vs salicin: {sim}");
    assert_eq!(peaks, 0);
}

#[test]
fn cocaine_vs_phenylalanine() {
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let phe = GenericSpectrum::phenylalanine().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&cocaine, &phe)
        .expect("similarity computation should succeed");
    assert!(sim < 1e-9, "cocaine vs phenylalanine: {sim}");
    assert_eq!(peaks, 0);
}

#[test]
fn glucose_vs_aspirin() {
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&glucose, &aspirin)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.003_525_189).abs() < 1e-6,
        "glucose vs aspirin: {sim}"
    );
    assert_eq!(peaks, 1);
}

#[test]
fn glucose_vs_hydroxy_cholesterol() {
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let hc = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&glucose, &hc)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.001_299_261_3).abs() < 1e-6,
        "glucose vs hydroxycholesterol: {sim}"
    );
    assert_eq!(peaks, 6);
}

#[test]
fn glucose_vs_salicin() {
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let salicin = GenericSpectrum::salicin().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&glucose, &salicin)
        .expect("similarity computation should succeed");
    assert!(sim < 1e-9, "glucose vs salicin: {sim}");
    assert_eq!(peaks, 0);
}

#[test]
fn glucose_vs_phenylalanine() {
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let phe = GenericSpectrum::phenylalanine().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&glucose, &phe)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.000_116_421_776).abs() < 1e-6,
        "glucose vs phenylalanine: {sim}"
    );
    assert_eq!(peaks, 2);
}

#[test]
fn aspirin_vs_hydroxy_cholesterol() {
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
    let hc = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&aspirin, &hc)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.000_772_552_33).abs() < 1e-6,
        "aspirin vs hydroxycholesterol: {sim}"
    );
    assert_eq!(peaks, 8);
}

#[test]
fn aspirin_vs_salicin() {
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
    let salicin = GenericSpectrum::salicin().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&aspirin, &salicin)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.000_015_375_54).abs() < 1e-6,
        "aspirin vs salicin: {sim}"
    );
    assert_eq!(peaks, 1);
}

#[test]
fn aspirin_vs_phenylalanine() {
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
    let phe = GenericSpectrum::phenylalanine().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&aspirin, &phe)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.044_286_36).abs() < 1e-6,
        "aspirin vs phenylalanine: {sim}"
    );
    assert_eq!(peaks, 3);
}

#[test]
fn hydroxy_cholesterol_vs_salicin() {
    let hc = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let salicin = GenericSpectrum::salicin().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&hc, &salicin)
        .expect("similarity computation should succeed");
    assert!(sim < 1e-9, "hydroxycholesterol vs salicin: {sim}");
    assert_eq!(peaks, 0);
}

#[test]
fn hydroxy_cholesterol_vs_phenylalanine() {
    let hc = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let phe = GenericSpectrum::phenylalanine().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&hc, &phe)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.000_014_375_056).abs() < 1e-6,
        "hydroxycholesterol vs phenylalanine: {sim}"
    );
    assert_eq!(peaks, 3);
}

#[test]
fn salicin_vs_phenylalanine() {
    let salicin = GenericSpectrum::salicin().expect("reference spectrum should build");
    let phe = GenericSpectrum::phenylalanine().expect("reference spectrum should build");
    let (sim, peaks) = cosine()
        .similarity(&salicin, &phe)
        .expect("similarity computation should succeed");
    assert!(
        (sim - 0.000_002_544_149_5).abs() < 1e-6,
        "salicin vs phenylalanine: {sim}"
    );
    assert_eq!(peaks, 1);
}
