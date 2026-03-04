//! Tests for the ModifiedGreedyCosine similarity implementation.

use mass_spectrometry::prelude::{
    AspirinSpectrum, CocaineSpectrum, GenericSpectrum, GlucoseSpectrum, GreedyCosine,
    HydroxyCholesterolSpectrum, ModifiedGreedyCosine, PhenylalanineSpectrum, SalicinSpectrum,
    ScalarSimilarity, Spectrum, SpectrumAlloc, SpectrumMut,
};

fn modified_greedy_cosine() -> ModifiedGreedyCosine<f32, f32> {
    ModifiedGreedyCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn assert_self_similarity(name: &str, spectrum: &GenericSpectrum<f32, f32>) {
    let (sim, peaks) = modified_greedy_cosine()
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
fn self_similarity_aspirin() {
    assert_self_similarity(
        "aspirin",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
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

// ---------- shift=0 equivalence with GreedyCosine ----------

/// When both spectra have the same precursor m/z (shift = 0), ModifiedGreedyCosine
/// should produce identical scores to GreedyCosine.
fn assert_shift0_equivalence(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let greedy = GreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (greedy_score, greedy_matches) = greedy
        .similarity(left, right)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(left, right)
        .expect("similarity computation should succeed");

    if left.precursor_mz() == right.precursor_mz() {
        assert!(
            (greedy_score - mod_score).abs() < 1e-6,
            "{name}: GreedyCosine={greedy_score} vs ModifiedGreedyCosine={mod_score}"
        );
        assert_eq!(
            greedy_matches, mod_matches,
            "{name}: GreedyCosine matches={greedy_matches} vs ModifiedGreedyCosine matches={mod_matches}"
        );
    } else {
        // With different precursors, modified cosine should find at least as
        // many matches (shifted window adds edges).
        assert!(
            mod_score >= greedy_score - 1e-6,
            "{name}: ModifiedGreedyCosine ({mod_score}) < GreedyCosine ({greedy_score})"
        );
    }
}

#[test]
fn shift0_equivalence_self() {
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    assert_shift0_equivalence("cocaine_self", &cocaine, &cocaine);

    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    assert_shift0_equivalence("glucose_self", &glucose, &glucose);

    let salicin = GenericSpectrum::salicin().expect("reference spectrum should build");
    assert_shift0_equivalence("salicin_self", &salicin, &salicin);
}

#[test]
fn shift0_equivalence_cross() {
    let cocaine = GenericSpectrum::cocaine().expect("reference spectrum should build");
    let glucose = GenericSpectrum::glucose().expect("reference spectrum should build");
    let aspirin = GenericSpectrum::aspirin().expect("reference spectrum should build");
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
    let mc = modified_greedy_cosine();
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
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
    );
}

#[test]
fn symmetry_aspirin_salicin() {
    assert_symmetry(
        "aspirin_salicin",
        &GenericSpectrum::aspirin().expect("reference spectrum should build"),
        &GenericSpectrum::salicin().expect("reference spectrum should build"),
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

// ---------- synthetic shifted-match case ----------

/// Build two spectra where a peak only matches through the shifted window.
///
/// Spectrum A: precursor=100, peaks at mz=[50, 80]
/// Spectrum B: precursor=110, peaks at mz=[50, 90]
///
/// shift = 100 - 110 = -10
/// Direct matches: mz=50 <-> mz=50 (within tol=0.1)
/// Shifted matches: mz=80 matches mz=90 because 80 - (-10) = 90
///
/// With GreedyCosine, only 1 match (mz=50). With ModifiedGreedyCosine, 2 matches.
#[test]
fn synthetic_shifted_match() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(110.0_f32, 2).expect("valid spectrum allocation");
    b.add_peak(50.0, 1000.0).unwrap();
    b.add_peak(90.0, 500.0).unwrap();

    let greedy = GreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (greedy_score, greedy_matches) = greedy
        .similarity(&a, &b)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert_eq!(greedy_matches, 1, "GreedyCosine should find 1 match");
    assert_eq!(mod_matches, 2, "ModifiedGreedyCosine should find 2 matches");
    assert!(
        mod_score > greedy_score,
        "ModifiedGreedyCosine score ({mod_score}) should exceed GreedyCosine ({greedy_score})"
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
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();

    // shift = 100 - 100.05 = -0.05, which is less than 2*tol=0.2
    let mut b = GenericSpectrum::with_capacity(100.05_f32, 2).expect("valid spectrum allocation");
    b.add_peak(50.0, 1000.0).unwrap();
    b.add_peak(80.0, 500.0).unwrap();

    let modified =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
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
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(200.0_f32, 1).expect("valid spectrum allocation");
    b.add_peak(300.0, 1000.0).unwrap();

    let modified =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let (score, matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert!(
        score.abs() < 1e-9,
        "No-match score should be ~0, got {score}"
    );
    assert_eq!(matches, 0);
}

/// Verify that ModifiedGreedyCosine with shift=0 yields exact same score as GreedyCosine
/// by using spectra with identical precursor masses.
#[test]
fn exact_equivalence_same_precursor() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 3).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();
    a.add_peak(120.0, 200.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(100.0_f32, 3).expect("valid spectrum allocation");
    b.add_peak(50.05, 800.0).unwrap();
    b.add_peak(80.03, 600.0).unwrap();
    b.add_peak(150.0, 300.0).unwrap();

    let greedy = GreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (greedy_score, greedy_matches) = greedy
        .similarity(&a, &b)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert!(
        (greedy_score - mod_score).abs() < 1e-6,
        "Same precursor: GreedyCosine={greedy_score} vs ModifiedGreedyCosine={mod_score}"
    );
    assert_eq!(greedy_matches, mod_matches);
}

#[test]
fn equivalence_within_precursor_tolerance() {
    let tolerance = 0.1_f32;

    let mut a = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();

    // precursor shift = 100.0 - 99.91 = 0.09, within tolerance.
    // The peak at 79.85 would only match via shifted matching; this verifies
    // modified behaves like non-modified inside the precursor-tolerance window.
    let mut b = GenericSpectrum::with_capacity(99.91_f32, 2).expect("valid spectrum allocation");
    b.add_peak(50.0, 900.0).unwrap();
    b.add_peak(79.85, 600.0).unwrap();

    let greedy = GreedyCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");
    let modified =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");

    let precursor_delta = (a.precursor_mz() - b.precursor_mz()).abs();
    assert!(
        precursor_delta <= tolerance,
        "test setup must satisfy within-tolerance precursor delta"
    );

    let (greedy_score, greedy_matches) = greedy
        .similarity(&a, &b)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert!(
        (greedy_score - mod_score).abs() < 1e-6,
        "Within precursor tolerance: GreedyCosine={greedy_score} vs ModifiedGreedyCosine={mod_score}"
    );
    assert_eq!(greedy_matches, mod_matches);
}

#[test]
fn equivalence_at_precursor_tolerance_boundary() {
    let tolerance = 0.125_f32;

    let mut a = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();

    // precursor shift = 100.0 - 99.875 = 0.125, exactly tolerance.
    let mut b = GenericSpectrum::with_capacity(99.875_f32, 2).expect("valid spectrum allocation");
    b.add_peak(50.0, 900.0).unwrap();
    b.add_peak(79.8, 600.0).unwrap();

    let greedy = GreedyCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");
    let modified =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");

    let precursor_delta = (a.precursor_mz() - b.precursor_mz()).abs();
    assert!(
        (precursor_delta - tolerance).abs() < 1e-6,
        "test setup must hit the tolerance boundary"
    );

    let (greedy_score, greedy_matches) = greedy
        .similarity(&a, &b)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert!(
        (greedy_score - mod_score).abs() < 1e-6,
        "Boundary precursor tolerance: GreedyCosine={greedy_score} vs ModifiedGreedyCosine={mod_score}"
    );
    assert_eq!(greedy_matches, mod_matches);
}
