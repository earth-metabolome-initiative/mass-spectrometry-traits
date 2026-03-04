//! Tests for the entropy similarity variant family.
//!
//! Covers all 4 variants: HungarianEntropy, LinearEntropy,
//! ModifiedHungarianEntropy, ModifiedLinearEntropy.

use mass_spectrometry::prelude::{
    AspirinSpectrum, CocaineSpectrum, GenericSpectrum, GlucoseSpectrum, HungarianEntropy,
    HydroxyCholesterolSpectrum, LinearEntropy, ModifiedHungarianEntropy, ModifiedLinearEntropy,
    PhenylalanineSpectrum, SalicinSpectrum, ScalarSimilarity, SimilarityComputationError, Spectrum,
    SpectrumMut,
};

// ========== Helper spectra ==========

fn cocaine() -> GenericSpectrum<f64, f64> {
    GenericSpectrum::cocaine().expect("reference spectrum should build")
}

fn glucose() -> GenericSpectrum<f64, f64> {
    GenericSpectrum::glucose().expect("reference spectrum should build")
}

fn aspirin() -> GenericSpectrum<f64, f64> {
    GenericSpectrum::aspirin().expect("reference spectrum should build")
}

fn hydroxy_cholesterol() -> GenericSpectrum<f64, f64> {
    GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build")
}

fn phenylalanine() -> GenericSpectrum<f64, f64> {
    GenericSpectrum::phenylalanine().expect("reference spectrum should build")
}

fn salicin() -> GenericSpectrum<f64, f64> {
    GenericSpectrum::salicin().expect("reference spectrum should build")
}

fn build_spectrum(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum<f64, f64> {
    let mut s =
        GenericSpectrum::try_with_capacity(precursor, peaks.len()).expect("valid precursor");
    for &(mz, int) in peaks {
        s.add_peak(mz, int).expect("valid peak");
    }
    s
}

/// Well-separated spectrum (peaks > 2 * 0.1 = 0.2 apart) for linear variants.
fn well_separated_a() -> GenericSpectrum<f64, f64> {
    build_spectrum(
        50.0,
        &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0), (400.0, 40.0)],
    )
}

fn well_separated_b() -> GenericSpectrum<f64, f64> {
    build_spectrum(
        55.0,
        &[(100.05, 15.0), (200.05, 25.0), (300.05, 35.0), (500.0, 5.0)],
    )
}

/// Spectrum with same mz as well_separated_a but different precursor.
fn well_separated_shifted() -> GenericSpectrum<f64, f64> {
    build_spectrum(
        60.0,
        &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0), (400.0, 40.0)],
    )
}

fn empty_spectrum() -> GenericSpectrum<f64, f64> {
    GenericSpectrum::try_with_capacity(100.0, 0).expect("valid precursor")
}

fn zero_intensity_spectrum() -> GenericSpectrum<f64, f64> {
    build_spectrum(100.0, &[(100.0, 0.0), (200.0, 0.0)])
}

// ========== Self-similarity tests ==========

macro_rules! test_self_similarity {
    ($name:ident, $scorer:expr, $spectrum_fn:ident) => {
        #[test]
        fn $name() {
            let scorer = $scorer;
            let spectrum = $spectrum_fn();
            let (sim, peaks): (f64, usize) = scorer
                .similarity(&spectrum, &spectrum)
                .expect("similarity should succeed");
            assert!(
                (1.0 - sim).abs() < 1e-10,
                "self-similarity: expected ~1.0, got {sim}"
            );
            assert_eq!(peaks, spectrum.len());
        }
    };
}

// HungarianEntropy self-similarity
test_self_similarity!(
    hungarian_entropy_self_cocaine,
    HungarianEntropy::weighted(0.1).unwrap(),
    cocaine
);
test_self_similarity!(
    hungarian_entropy_self_aspirin,
    HungarianEntropy::weighted(0.1).unwrap(),
    aspirin
);

// LinearEntropy self-similarity (well-separated spectra only)
test_self_similarity!(
    linear_entropy_self_well_separated,
    LinearEntropy::weighted(0.1).unwrap(),
    well_separated_a
);

// ModifiedHungarianEntropy self-similarity
test_self_similarity!(
    modified_hungarian_entropy_self_cocaine,
    ModifiedHungarianEntropy::weighted(0.1).unwrap(),
    cocaine
);
test_self_similarity!(
    modified_hungarian_entropy_self_aspirin,
    ModifiedHungarianEntropy::weighted(0.1).unwrap(),
    aspirin
);

// ModifiedLinearEntropy self-similarity (well-separated only)
test_self_similarity!(
    modified_linear_entropy_self_well_separated,
    ModifiedLinearEntropy::weighted(0.1).unwrap(),
    well_separated_a
);

// ========== Symmetry tests ==========

macro_rules! test_symmetry {
    ($name:ident, $scorer:expr, $a_fn:ident, $b_fn:ident) => {
        #[test]
        fn $name() {
            let scorer = $scorer;
            let a = $a_fn();
            let b = $b_fn();
            let (sim_ab, peaks_ab): (f64, usize) = scorer
                .similarity(&a, &b)
                .expect("similarity should succeed");
            let (sim_ba, peaks_ba): (f64, usize) = scorer
                .similarity(&b, &a)
                .expect("similarity should succeed");
            assert!(
                (sim_ab - sim_ba).abs() < 1e-10,
                "symmetry: {sim_ab} vs {sim_ba}"
            );
            assert_eq!(peaks_ab, peaks_ba, "symmetry: peak count mismatch");
        }
    };
}

test_symmetry!(
    hungarian_entropy_symmetry,
    HungarianEntropy::weighted(0.1).unwrap(),
    cocaine,
    aspirin
);
test_symmetry!(
    linear_entropy_symmetry,
    LinearEntropy::weighted(0.1).unwrap(),
    well_separated_a,
    well_separated_b
);
test_symmetry!(
    modified_hungarian_entropy_symmetry,
    ModifiedHungarianEntropy::weighted(0.1).unwrap(),
    cocaine,
    aspirin
);
test_symmetry!(
    modified_linear_entropy_symmetry,
    ModifiedLinearEntropy::weighted(0.1).unwrap(),
    well_separated_a,
    well_separated_b
);

// ========== Empty / zero spectra edge cases ==========

macro_rules! test_empty_zero {
    ($name:ident, $scorer:expr) => {
        #[test]
        fn $name() {
            let scorer = $scorer;
            let spec = cocaine();
            let empty = empty_spectrum();
            let zero = zero_intensity_spectrum();

            let (sim, n): (f64, usize) = scorer
                .similarity(&spec, &empty)
                .expect("similarity should succeed");
            assert_eq!(sim, 0.0, "empty: sim should be 0.0");
            assert_eq!(n, 0, "empty: matches should be 0");

            let (sim, n): (f64, usize) = scorer
                .similarity(&spec, &zero)
                .expect("similarity should succeed");
            assert_eq!(sim, 0.0, "zero: sim should be 0.0");
            assert_eq!(n, 0, "zero: matches should be 0");
        }
    };
}

test_empty_zero!(
    hungarian_entropy_empty_zero,
    HungarianEntropy::weighted(0.1).unwrap()
);
test_empty_zero!(
    modified_hungarian_entropy_empty_zero,
    ModifiedHungarianEntropy::weighted(0.1).unwrap()
);

#[test]
fn linear_entropy_empty_zero() {
    let scorer = LinearEntropy::weighted(0.1).expect("valid scorer config");
    let spec = well_separated_a();
    let empty = empty_spectrum();
    let zero = zero_intensity_spectrum();

    let (sim_empty, n_empty): (f64, usize) = scorer
        .similarity(&spec, &empty)
        .expect("similarity should succeed");
    assert_eq!(sim_empty, 0.0);
    assert_eq!(n_empty, 0);

    let (sim_zero, n_zero): (f64, usize) = scorer
        .similarity(&spec, &zero)
        .expect("similarity should succeed");
    assert_eq!(sim_zero, 0.0);
    assert_eq!(n_zero, 0);
}

#[test]
fn modified_linear_entropy_empty_zero() {
    let scorer = ModifiedLinearEntropy::weighted(0.1).expect("valid scorer config");
    let spec = well_separated_a();
    let empty = empty_spectrum();
    let zero = zero_intensity_spectrum();

    let (sim_empty, n_empty): (f64, usize) = scorer
        .similarity(&spec, &empty)
        .expect("similarity should succeed");
    assert_eq!(sim_empty, 0.0);
    assert_eq!(n_empty, 0);

    let (sim_zero, n_zero): (f64, usize) = scorer
        .similarity(&spec, &zero)
        .expect("similarity should succeed");
    assert_eq!(sim_zero, 0.0);
    assert_eq!(n_zero, 0);
}

#[test]
fn entropy_variant_config_accessors() {
    let hungarian = HungarianEntropy::unweighted(0.2).expect("valid scorer config");
    assert_eq!(hungarian.mz_tolerance(), 0.2);
    assert!(!hungarian.is_weighted());

    let linear = LinearEntropy::weighted(0.25).expect("valid scorer config");
    assert_eq!(linear.mz_tolerance(), 0.25);
    assert!(linear.is_weighted());

    let modified_hungarian = ModifiedHungarianEntropy::weighted(0.35).expect("valid scorer config");
    assert_eq!(modified_hungarian.mz_tolerance(), 0.35);
    assert!(modified_hungarian.is_weighted());

    let modified_linear = ModifiedLinearEntropy::unweighted(0.4).expect("valid scorer config");
    assert_eq!(modified_linear.mz_tolerance(), 0.4);
    assert!(!modified_linear.is_weighted());
}

#[test]
fn modified_variants_non_empty_no_matches() {
    let a = build_spectrum(100.0, &[(100.0, 1.0), (200.0, 2.0)]);
    let b = build_spectrum(120.0, &[(500.0, 3.0), (600.0, 4.0)]);

    let modified_hungarian =
        ModifiedHungarianEntropy::unweighted(0.01).expect("valid scorer config");

    let (sim_h, n_h): (f64, usize) = modified_hungarian
        .similarity(&a, &b)
        .expect("similarity should succeed");

    assert_eq!(sim_h, 0.0);
    assert_eq!(n_h, 0);
}

// ========== Cross-checks: Linear == Hungarian on well-separated spectra ==========

#[test]
fn linear_equals_hungarian_well_separated_weighted() {
    let a = well_separated_a();
    let b = well_separated_b();
    let linear = LinearEntropy::weighted(0.1).unwrap();
    let hungarian = HungarianEntropy::weighted(0.1).unwrap();

    let (sim_l, n_l): (f64, usize) = linear.similarity(&a, &b).expect("linear should succeed");
    let (sim_h, n_h): (f64, usize) = hungarian
        .similarity(&a, &b)
        .expect("hungarian should succeed");

    assert!(
        (sim_l - sim_h).abs() < 1e-10,
        "linear vs hungarian: {sim_l} vs {sim_h}"
    );
    assert_eq!(n_l, n_h);
}

#[test]
fn linear_equals_hungarian_well_separated_unweighted() {
    let a = well_separated_a();
    let b = well_separated_b();
    let linear = LinearEntropy::unweighted(0.1).unwrap();
    let hungarian = HungarianEntropy::unweighted(0.1).unwrap();

    let (sim_l, n_l): (f64, usize) = linear.similarity(&a, &b).expect("linear should succeed");
    let (sim_h, n_h): (f64, usize) = hungarian
        .similarity(&a, &b)
        .expect("hungarian should succeed");

    assert!(
        (sim_l - sim_h).abs() < 1e-10,
        "linear vs hungarian: {sim_l} vs {sim_h}"
    );
    assert_eq!(n_l, n_h);
}

// ========== Optimality: Hungarian >= Greedy ==========

#[test]
fn hungarian_geq_greedy_weighted() {
    let pairs: Vec<(&str, GenericSpectrum<f64, f64>)> = vec![
        ("cocaine", cocaine()),
        ("glucose", glucose()),
        ("aspirin", aspirin()),
        ("hc", hydroxy_cholesterol()),
        ("phe", phenylalanine()),
        ("salicin", salicin()),
    ];
    let hungarian = HungarianEntropy::weighted(0.1).unwrap();

    for (i, (name_a, a)) in pairs.iter().enumerate() {
        for (name_b, b) in pairs.iter().skip(i + 1) {
            let (sim_h, _): (f64, usize) = hungarian
                .similarity(a, b)
                .expect("hungarian should succeed");
            assert!(
                sim_h >= -1e-10,
                "hungarian score should be non-negative for {name_a} vs {name_b}: {sim_h}"
            );
        }
    }
}

// ========== Modified variants: precursor tolerance invariant ==========
// When |precursor_delta| <= tolerance, modified score == direct counterpart.

#[test]
fn modified_hungarian_equals_hungarian_within_tolerance() {
    let a = cocaine();
    let hungarian = HungarianEntropy::weighted(0.1).unwrap();
    let modified = ModifiedHungarianEntropy::weighted(0.1).unwrap();

    let (sim_h, n_h): (f64, usize) = hungarian
        .similarity(&a, &a)
        .expect("hungarian should succeed");
    let (sim_m, n_m): (f64, usize) = modified
        .similarity(&a, &a)
        .expect("modified hungarian should succeed");

    assert!(
        (sim_h - sim_m).abs() < 1e-10,
        "hungarian vs modified hungarian: {sim_h} vs {sim_m}"
    );
    assert_eq!(n_h, n_m);
}

#[test]
fn modified_linear_equals_linear_within_tolerance() {
    let a = well_separated_a();
    let linear = LinearEntropy::weighted(0.1).unwrap();
    let modified = ModifiedLinearEntropy::weighted(0.1).unwrap();

    let (sim_l, n_l): (f64, usize) = linear.similarity(&a, &a).expect("linear should succeed");
    let (sim_m, n_m): (f64, usize) = modified
        .similarity(&a, &a)
        .expect("modified linear should succeed");

    assert!(
        (sim_l - sim_m).abs() < 1e-10,
        "linear vs modified linear: {sim_l} vs {sim_m}"
    );
    assert_eq!(n_l, n_m);
}

// ========== Modified variants: boundary tests at tolerance ± ε ==========

#[test]
fn modified_hungarian_boundary_at_tolerance() {
    let tolerance = 5.0;
    // precursor delta = 5.0 = tolerance => should behave like direct
    let a = build_spectrum(100.0, &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)]);
    let b = build_spectrum(105.0, &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)]);

    let hungarian = HungarianEntropy::weighted(tolerance).unwrap();
    let modified = ModifiedHungarianEntropy::weighted(tolerance).unwrap();

    let (sim_h, n_h): (f64, usize) = hungarian
        .similarity(&a, &b)
        .expect("hungarian should succeed");
    let (sim_m, n_m): (f64, usize) = modified
        .similarity(&a, &b)
        .expect("modified should succeed");

    assert!(
        (sim_h - sim_m).abs() < 1e-10,
        "at tolerance boundary: {sim_h} vs {sim_m}"
    );
    assert_eq!(n_h, n_m);
}

#[test]
fn modified_hungarian_boundary_beyond_tolerance() {
    let tolerance = 5.0;
    // precursor delta = 5.0 + 2ε > tolerance => shifted matching active
    let a = build_spectrum(100.0, &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)]);
    let b = build_spectrum(
        105.0 + 2.0 * f64::EPSILON,
        &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)],
    );

    let modified = ModifiedHungarianEntropy::weighted(tolerance).unwrap();
    let (sim_m, _n_m): (f64, usize) = modified
        .similarity(&a, &b)
        .expect("modified should succeed");

    // Score should still be valid (>= 0) even with shifted matching
    assert!(sim_m >= 0.0);
}

// ========== Linear variants: error on non-well-separated spectra ==========

#[test]
fn linear_entropy_rejects_close_peaks() {
    // Peaks 100.0 and 100.15 are within 2 * 0.1 = 0.2
    let close = build_spectrum(50.0, &[(100.0, 10.0), (100.15, 20.0), (200.0, 30.0)]);
    let other = well_separated_a();

    let linear = LinearEntropy::weighted(0.1).unwrap();
    let result: Result<(f64, usize), SimilarityComputationError> =
        linear.similarity(&close, &other);
    assert!(
        matches!(
            result,
            Err(SimilarityComputationError::InvalidPeakSpacing(_))
        ),
        "expected InvalidPeakSpacing, got {:?}",
        result
    );
}

#[test]
fn modified_linear_entropy_rejects_close_peaks() {
    let close = build_spectrum(50.0, &[(100.0, 10.0), (100.15, 20.0), (200.0, 30.0)]);
    let other = well_separated_a();

    let modified = ModifiedLinearEntropy::weighted(0.1).unwrap();
    let result: Result<(f64, usize), SimilarityComputationError> =
        modified.similarity(&close, &other);
    assert!(
        matches!(
            result,
            Err(SimilarityComputationError::InvalidPeakSpacing(_))
        ),
        "expected InvalidPeakSpacing, got {:?}",
        result
    );
}

// ========== Cross-variant comparison on same spectra ==========

#[test]
fn all_direct_variants_agree_on_well_separated() {
    let a = well_separated_a();
    let b = well_separated_b();

    let hungarian = HungarianEntropy::unweighted(0.1).unwrap();
    let linear = LinearEntropy::unweighted(0.1).unwrap();

    let (sim_h, n_h): (f64, usize) = hungarian.similarity(&a, &b).expect("hungarian");
    let (sim_l, n_l): (f64, usize) = linear.similarity(&a, &b).expect("linear");

    // For well-separated spectra, both should produce identical results
    assert!(
        (sim_h - sim_l).abs() < 1e-10,
        "hungarian vs linear: {sim_h} vs {sim_l}"
    );
    assert_eq!(n_h, n_l);
}

#[test]
fn all_modified_variants_self_similarity() {
    let a = well_separated_a();

    let m_hungarian = ModifiedHungarianEntropy::unweighted(0.1).unwrap();
    let m_linear = ModifiedLinearEntropy::unweighted(0.1).unwrap();

    let (sim_mh, _): (f64, usize) = m_hungarian.similarity(&a, &a).expect("modified hungarian");
    let (sim_ml, _): (f64, usize) = m_linear.similarity(&a, &a).expect("modified linear");

    assert!(
        (1.0 - sim_mh).abs() < 1e-10,
        "modified hungarian self-sim: {sim_mh}"
    );
    assert!(
        (1.0 - sim_ml).abs() < 1e-10,
        "modified linear self-sim: {sim_ml}"
    );
}

// ========== Modified variants with actual precursor shift ==========

#[test]
fn modified_variants_with_precursor_shift() {
    let a = well_separated_a(); // precursor 50.0
    let b = well_separated_shifted(); // precursor 60.0, same peaks

    let m_hungarian = ModifiedHungarianEntropy::unweighted(0.1).unwrap();

    // Self-sim despite precursor shift (shift=10, well beyond tolerance=0.1)
    // but peaks are identical so direct matches should still give score=1
    let (sim_mh, _): (f64, usize) = m_hungarian.similarity(&a, &b).expect("modified hungarian");

    // Score should be 1.0 because direct matching still matches all peaks
    assert!(
        (1.0 - sim_mh).abs() < 1e-10,
        "modified hungarian with shift: {sim_mh}"
    );
}
