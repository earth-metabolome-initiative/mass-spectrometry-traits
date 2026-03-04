//! Tests for the ModifiedLinearCosine similarity implementation.

use mass_spectrometry::prelude::{
    CocaineSpectrum, GenericSpectrum, GlucoseSpectrum, HydroxyCholesterolSpectrum, LinearCosine,
    ModifiedGreedyCosine, ModifiedHungarianCosine, ModifiedLinearCosine, PhenylalanineSpectrum,
    SalicinSpectrum, ScalarSimilarity, SimilarityComputationError, Spectrum, SpectrumAlloc,
    SpectrumMut,
};

fn modified_linear_cosine() -> ModifiedLinearCosine<f32, f32> {
    ModifiedLinearCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn assert_self_similarity(name: &str, spectrum: &GenericSpectrum<f32, f32>) {
    let (sim, peaks) = modified_linear_cosine()
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

// ---------- shift=0 equivalence with LinearCosine ----------

fn assert_shift0_equivalence(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let linear = LinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (linear_score, linear_matches) = linear
        .similarity(left, right)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(left, right)
        .expect("similarity computation should succeed");

    if left.precursor_mz() == right.precursor_mz() {
        assert!(
            (linear_score - mod_score).abs() < 1e-6,
            "{name}: LinearCosine={linear_score} vs ModifiedLinearCosine={mod_score}"
        );
        assert_eq!(
            linear_matches, mod_matches,
            "{name}: LinearCosine matches={linear_matches} vs ModifiedLinearCosine matches={mod_matches}"
        );
    } else {
        assert!(
            mod_score >= linear_score - 1e-6,
            "{name}: ModifiedLinearCosine ({mod_score}) < LinearCosine ({linear_score})"
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
    assert_shift0_equivalence("cocaine_vs_glucose", &cocaine, &glucose);
}

// ---------- strict parity with ModifiedGreedyCosine on reference spectra ----------

fn assert_matches_modified_greedy(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let modified_greedy =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified_linear =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (greedy_score, greedy_matches) = modified_greedy
        .similarity(left, right)
        .expect("similarity computation should succeed");
    let (linear_score, linear_matches) = modified_linear
        .similarity(left, right)
        .expect("similarity computation should succeed");

    assert!(
        (greedy_score - linear_score).abs() < 1e-6,
        "{name}: ModifiedGreedyCosine={greedy_score} vs ModifiedLinearCosine={linear_score}"
    );
    assert_eq!(
        greedy_matches, linear_matches,
        "{name}: ModifiedGreedyCosine matches={greedy_matches} vs ModifiedLinearCosine matches={linear_matches}"
    );
}

#[test]
fn parity_with_modified_greedy_cocaine_glucose() {
    assert_matches_modified_greedy(
        "cocaine_glucose",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
    );
}

#[test]
fn parity_with_modified_greedy_hydroxy_cholesterol_phenylalanine() {
    assert_matches_modified_greedy(
        "hc_phe",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

// ---------- non-strict regression check vs ModifiedHungarianCosine ----------

fn assert_no_better_than_modified_hungarian(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let modified_hungarian =
        ModifiedHungarianCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified_linear =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (hungarian_score, hungarian_matches) = modified_hungarian
        .similarity(left, right)
        .expect("similarity computation should succeed");
    let (linear_score, linear_matches) = modified_linear
        .similarity(left, right)
        .expect("similarity computation should succeed");

    assert!(
        linear_score <= hungarian_score + 1e-6,
        "{name}: ModifiedLinearCosine score={linear_score} exceeded ModifiedHungarianCosine score={hungarian_score}"
    );
    assert!(
        linear_matches <= hungarian_matches,
        "{name}: ModifiedLinearCosine matches={linear_matches} exceeded ModifiedHungarianCosine matches={hungarian_matches}"
    );
}

#[test]
fn regression_vs_modified_hungarian_cocaine_glucose() {
    assert_no_better_than_modified_hungarian(
        "cocaine_glucose",
        &GenericSpectrum::cocaine().expect("reference spectrum should build"),
        &GenericSpectrum::glucose().expect("reference spectrum should build"),
    );
}

#[test]
fn regression_vs_modified_hungarian_hydroxy_cholesterol_phenylalanine() {
    assert_no_better_than_modified_hungarian(
        "hc_phe",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

// ---------- symmetry: sim(A, B) == sim(B, A) ----------

fn assert_symmetry(
    name: &str,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let mc = modified_linear_cosine();
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
fn symmetry_hydroxy_cholesterol_phenylalanine() {
    assert_symmetry(
        "hc_phe",
        &GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build"),
        &GenericSpectrum::phenylalanine().expect("reference spectrum should build"),
    );
}

// ---------- synthetic shifted-match case ----------

#[test]
fn synthetic_shifted_match() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();
    a.add_peak(80.0, 500.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(110.0_f32, 2).expect("valid spectrum allocation");
    b.add_peak(50.0, 1000.0).unwrap();
    b.add_peak(90.0, 500.0).unwrap();

    let linear = LinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (linear_score, linear_matches) = linear
        .similarity(&a, &b)
        .expect("similarity computation should succeed");
    let (mod_score, mod_matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert_eq!(linear_matches, 1, "LinearCosine should find 1 match");
    assert_eq!(mod_matches, 2, "ModifiedLinearCosine should find 2 matches");
    assert!(
        mod_score > linear_score,
        "ModifiedLinearCosine score ({mod_score}) should exceed LinearCosine ({linear_score})"
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

#[test]
fn overlapping_direct_and_shifted_windows_do_not_double_count() {
    let mut left = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    left.add_peak(50.0, 1000.0).unwrap();
    left.add_peak(80.0, 500.0).unwrap();

    let mut right =
        GenericSpectrum::with_capacity(100.05_f32, 2).expect("valid spectrum allocation");
    right.add_peak(50.03, 1000.0).unwrap();
    right.add_peak(80.03, 500.0).unwrap();

    let modified_greedy =
        ModifiedGreedyCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let modified_linear =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let (greedy_score, greedy_matches) = modified_greedy
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (linear_score, linear_matches) = modified_linear
        .similarity(&left, &right)
        .expect("similarity computation should succeed");

    assert!(
        (greedy_score - linear_score).abs() < 1e-6,
        "ModifiedGreedyCosine={greedy_score} vs ModifiedLinearCosine={linear_score}"
    );
    assert_eq!(
        greedy_matches, linear_matches,
        "ModifiedGreedyCosine matches={greedy_matches} vs ModifiedLinearCosine matches={linear_matches}"
    );
    assert_eq!(linear_matches, 2);
}

#[test]
fn synthetic_no_matches() {
    let mut a = GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    a.add_peak(50.0, 1000.0).unwrap();

    let mut b = GenericSpectrum::with_capacity(200.0_f32, 1).expect("valid spectrum allocation");
    b.add_peak(300.0, 1000.0).unwrap();

    let modified =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");
    let (score, matches) = modified
        .similarity(&a, &b)
        .expect("similarity computation should succeed");

    assert!(
        score.abs() < 1e-9,
        "No-match score should be ~0, got {score}"
    );
    assert_eq!(matches, 0);
}

#[test]
fn boundary_gap_equal_2x_tolerance_returns_error() {
    let mut left = GenericSpectrum::with_capacity(200.0_f32, 2).expect("valid spectrum allocation");
    left.add_peak(100.0, 10.0).unwrap();
    left.add_peak(100.25, 8.0).unwrap();

    let mut right =
        GenericSpectrum::with_capacity(200.0_f32, 2).expect("valid spectrum allocation");
    right.add_peak(100.0, 10.0).unwrap();
    right.add_peak(100.25, 8.0).unwrap();

    let modified =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.125_f32).expect("valid scorer config");
    let error = modified
        .similarity(&left, &right)
        .expect_err("boundary-equal spacing should be rejected");

    assert_eq!(
        error,
        SimilarityComputationError::InvalidPeakSpacing("left spectrum")
    );
}

#[test]
fn equivalence_within_precursor_tolerance() {
    let tolerance = 0.1_f32;

    let mut left = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    left.add_peak(50.0, 1000.0).unwrap();
    left.add_peak(80.0, 500.0).unwrap();

    // precursor shift = 100.0 - 99.91 = 0.09, within tolerance.
    // The peak at 79.85 would only match via shifted matching; this verifies
    // modified behaves like non-modified inside the precursor-tolerance window.
    let mut right =
        GenericSpectrum::with_capacity(99.91_f32, 2).expect("valid spectrum allocation");
    right.add_peak(50.0, 900.0).unwrap();
    right.add_peak(79.85, 600.0).unwrap();

    let linear = LinearCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");
    let modified =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");

    let precursor_delta = (left.precursor_mz() - right.precursor_mz()).abs();
    assert!(
        precursor_delta <= tolerance,
        "test setup must satisfy within-tolerance precursor delta"
    );

    let (linear_score, linear_matches) = linear
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (modified_score, modified_matches) = modified
        .similarity(&left, &right)
        .expect("similarity computation should succeed");

    assert!(
        (linear_score - modified_score).abs() < 1e-6,
        "Within precursor tolerance: LinearCosine={linear_score} vs ModifiedLinearCosine={modified_score}"
    );
    assert_eq!(linear_matches, modified_matches);
}

#[test]
fn equivalence_at_precursor_tolerance_boundary() {
    let tolerance = 0.125_f32;

    let mut left = GenericSpectrum::with_capacity(100.0_f32, 2).expect("valid spectrum allocation");
    left.add_peak(50.0, 1000.0).unwrap();
    left.add_peak(80.0, 500.0).unwrap();

    // precursor shift = 100.0 - 99.875 = 0.125, exactly tolerance.
    let mut right =
        GenericSpectrum::with_capacity(99.875_f32, 2).expect("valid spectrum allocation");
    right.add_peak(50.0, 900.0).unwrap();
    right.add_peak(79.8, 600.0).unwrap();

    let linear = LinearCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");
    let modified =
        ModifiedLinearCosine::new(1.0_f32, 1.0_f32, tolerance).expect("valid scorer config");

    let precursor_delta = (left.precursor_mz() - right.precursor_mz()).abs();
    assert!(
        (precursor_delta - tolerance).abs() < 1e-6,
        "test setup must hit the tolerance boundary"
    );

    let (linear_score, linear_matches) = linear
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (modified_score, modified_matches) = modified
        .similarity(&left, &right)
        .expect("similarity computation should succeed");

    assert!(
        (linear_score - modified_score).abs() < 1e-6,
        "Boundary precursor tolerance: LinearCosine={linear_score} vs ModifiedLinearCosine={modified_score}"
    );
    assert_eq!(linear_matches, modified_matches);
}
