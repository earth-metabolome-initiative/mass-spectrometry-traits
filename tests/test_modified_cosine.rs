//! Tests for the ModifiedCosine similarity implementation.

use mass_spectrometry::prelude::{
    AspirinSpectrum, CocaineSpectrum, ExactCosine, GenericSpectrum, GlucoseSpectrum,
    HydroxyCholesterolSpectrum, ModifiedCosine, PhenylalanineSpectrum, SalicinSpectrum,
    ScalarSimilarity, Spectrum, SpectrumAlloc, SpectrumMut,
};

fn modified_cosine() -> ModifiedCosine<f32, f32> {
    ModifiedCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn assert_self_similarity(name: &str, spectrum: &GenericSpectrum<f32, f32>) {
    let (sim, peaks) = modified_cosine()
        .similarity(spectrum, spectrum)
        .expect("similarity computation should succeed");
    assert!(
        (1.0_f32 - sim).abs() < 1e-6,
        "{name} self-similarity: expected ~1.0, got {sim}"
    );
    assert_eq!(
        peaks,
        spectrum.len(),
        "{name} self-similarity: expected {0} matches, got {peaks}",
        spectrum.len()
    );
}

// ---------- self-similarity (score must be ~1.0, matches = #peaks) ----------

#[test]
fn self_similarity_cocaine() {
    assert_self_similarity("cocaine", &GenericSpectrum::cocaine());
}

#[test]
fn self_similarity_glucose() {
    assert_self_similarity("glucose", &GenericSpectrum::glucose());
}

#[test]
fn self_similarity_aspirin() {
    assert_self_similarity("aspirin", &GenericSpectrum::aspirin());
}

#[test]
fn self_similarity_hydroxy_cholesterol() {
    assert_self_similarity(
        "hydroxy_cholesterol",
        &GenericSpectrum::hydroxy_cholesterol(),
    );
}

#[test]
fn self_similarity_salicin() {
    assert_self_similarity("salicin", &GenericSpectrum::salicin());
}

#[test]
fn self_similarity_phenylalanine() {
    assert_self_similarity("phenylalanine", &GenericSpectrum::phenylalanine());
}

// ---------- shift=0 equivalence with ExactCosine ----------

/// When both spectra have the same precursor m/z (shift = 0), ModifiedCosine
/// should produce identical scores to ExactCosine.
fn assert_shift0_equivalence(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let exact = ExactCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified = ModifiedCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    // Force same precursor by wrapping — but our test spectra already have
    // distinct precursors. Instead, when shift=0 (same precursor), the shifted
    // window duplicates the direct window, so results must match ExactCosine.
    // We test this via self-similarity (precursor difference = 0).
    let (exact_score, exact_matches) = exact
        .similarity(left, right)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(left, right)
        .expect("similarity computation should succeed");

    // When precursors differ, modified may find additional matches, so
    // score >= exact_score. But when precursors are equal (self-similarity),
    // they should be identical.
    if left.precursor_mz() == right.precursor_mz() {
        assert!(
            (exact_score - mod_score).abs() < 1e-6,
            "{name}: ExactCosine={exact_score} vs ModifiedCosine={mod_score}"
        );
        assert_eq!(
            exact_matches, mod_matches,
            "{name}: ExactCosine matches={exact_matches} vs ModifiedCosine matches={mod_matches}"
        );
    } else {
        // With different precursors, modified cosine should find at least as
        // many matches (shifted window adds edges).
        assert!(
            mod_score >= exact_score - 1e-6,
            "{name}: ModifiedCosine ({mod_score}) < ExactCosine ({exact_score})"
        );
    }
}

#[test]
fn shift0_equivalence_self() {
    let cocaine = GenericSpectrum::cocaine();
    assert_shift0_equivalence("cocaine_self", &cocaine, &cocaine);

    let glucose = GenericSpectrum::glucose();
    assert_shift0_equivalence("glucose_self", &glucose, &glucose);

    let salicin = GenericSpectrum::salicin();
    assert_shift0_equivalence("salicin_self", &salicin, &salicin);
}

#[test]
fn shift0_equivalence_cross() {
    // Cross-similarity between spectra with different precursors.
    let cocaine = GenericSpectrum::cocaine();
    let glucose = GenericSpectrum::glucose();
    let aspirin = GenericSpectrum::aspirin();
    assert_shift0_equivalence("cocaine_vs_glucose", &cocaine, &glucose);
    assert_shift0_equivalence("cocaine_vs_aspirin", &cocaine, &aspirin);
    assert_shift0_equivalence("glucose_vs_aspirin", &glucose, &aspirin);
}

// ---------- symmetry: sim(A, B) == sim(B, A) ----------

fn assert_symmetry(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let mc = modified_cosine();
    let (score_ab, matches_ab) = mc
        .similarity(left, right)
        .expect("similarity computation should succeed");
    let (score_ba, matches_ba) = mc
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
        &GenericSpectrum::cocaine(),
        &GenericSpectrum::glucose(),
    );
}

#[test]
fn symmetry_aspirin_salicin() {
    assert_symmetry(
        "aspirin_salicin",
        &GenericSpectrum::aspirin(),
        &GenericSpectrum::salicin(),
    );
}

#[test]
fn symmetry_hydroxy_cholesterol_phenylalanine() {
    assert_symmetry(
        "hc_phe",
        &GenericSpectrum::hydroxy_cholesterol(),
        &GenericSpectrum::phenylalanine(),
    );
}

// ---------- synthetic shifted-match case ----------

/// Build two spectra where a peak only matches through the shifted window.
///
/// Spectrum A: precursor=100, peaks at mz=[50, 80]
/// Spectrum B: precursor=110, peaks at mz=[50, 90]
///
/// shift = 100 - 110 = -10
/// Direct matches: mz=50 ↔ mz=50 (within tol=0.1)
/// Shifted matches: mz=80 matches mz=90 because 80 - (-10) = 90
///
/// With ExactCosine, only 1 match (mz=50). With ModifiedCosine, 2 matches.
#[test]
fn synthetic_shifted_match() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 2);
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(110.0_f32, 2);
    b.add_peak(50.0, 1000.0).unwrap();
    b.add_peak(90.0, 500.0).unwrap();

    let exact = ExactCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified = ModifiedCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (exact_score, exact_matches) = exact
        .similarity(&a, &b)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert_eq!(exact_matches, 1, "ExactCosine should find 1 match");
    assert_eq!(mod_matches, 2, "ModifiedCosine should find 2 matches");
    assert!(
        mod_score > exact_score,
        "ModifiedCosine score ({mod_score}) should exceed ExactCosine ({exact_score})"
    );

    // Verify symmetry on the synthetic case.
    let (mod_score_ba, mod_matches_ba) = modified
        .similarity(&b, &a)
        .expect("similarity computation should succeed");
    assert!(
        (mod_score - mod_score_ba).abs() < 1e-6,
        "Synthetic symmetry: {mod_score} != {mod_score_ba}"
    );
    assert_eq!(mod_matches, mod_matches_ba);
}

/// Another synthetic case: shift exactly equals direct match offset.
/// When |shift| < 2*tol, windows overlap. Ensure no panics from duplicates.
#[test]
fn synthetic_overlapping_windows() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 2);
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();

    // shift = 100 - 100.05 = -0.05, which is less than 2*tol=0.2
    let mut b = GenericSpectrum::with_capacity(100.05_f32, 2);
    b.add_peak(50.0, 1000.0).unwrap();
    b.add_peak(80.0, 500.0).unwrap();

    let modified = ModifiedCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let (score, matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    // Both peaks should match directly; the shifted window nearly overlaps
    // but should not cause panics.
    assert!(score > 0.99, "Near-identical spectra: score={score}");
    assert_eq!(matches, 2);
}

/// Synthetic: no matches at all (peaks far apart, shift doesn't help).
#[test]
fn synthetic_no_matches() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 1);
    a.add_peak(50.0, 1000.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(200.0_f32, 1);
    b.add_peak(300.0, 1000.0).unwrap();

    let modified = ModifiedCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let (score, matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert!(
        score.abs() < 1e-9,
        "No-match score should be ~0, got {score}"
    );
    assert_eq!(matches, 0);
}

/// Verify that ModifiedCosine with shift=0 yields exact same score as ExactCosine
/// by using spectra with identical precursor masses.
#[test]
fn exact_equivalence_same_precursor() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 3);
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();
    a.add_peak(120.0, 200.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(100.0_f32, 3);
    b.add_peak(50.05, 800.0).unwrap();
    b.add_peak(80.03, 600.0).unwrap();
    b.add_peak(150.0, 300.0).unwrap();

    let exact = ExactCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified = ModifiedCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (exact_score, exact_matches) = exact
        .similarity(&a, &b)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert!(
        (exact_score - mod_score).abs() < 1e-6,
        "Same precursor: ExactCosine={exact_score} vs ModifiedCosine={mod_score}"
    );
    assert_eq!(exact_matches, mod_matches);
}
