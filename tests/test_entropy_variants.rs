//! Tests for the entropy similarity variant family.
//!
//! Covers LinearEntropy and ModifiedLinearEntropy.

use mass_spectrometry::prelude::{
    GenericSpectrum, LinearEntropy, ModifiedLinearEntropy, ScalarSimilarity,
    SimilarityComputationError, Spectrum, SpectrumMut,
};

// ========== Helper spectra ==========

fn build_spectrum(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut s =
        GenericSpectrum::try_with_capacity(precursor, peaks.len()).expect("valid precursor");
    for &(mz, int) in peaks {
        s.add_peak(mz, int).expect("valid peak");
    }
    s
}

/// Well-separated spectrum (peaks > 2 * 0.1 = 0.2 apart) for linear variants.
fn well_separated_a() -> GenericSpectrum {
    build_spectrum(
        50.0,
        &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0), (400.0, 40.0)],
    )
}

fn well_separated_b() -> GenericSpectrum {
    build_spectrum(
        55.0,
        &[(100.05, 15.0), (200.05, 25.0), (300.05, 35.0), (500.0, 5.0)],
    )
}

/// Spectrum with same mz as well_separated_a but different precursor.
fn well_separated_shifted() -> GenericSpectrum {
    build_spectrum(
        60.0,
        &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0), (400.0, 40.0)],
    )
}

fn empty_spectrum() -> GenericSpectrum {
    GenericSpectrum::try_with_capacity(100.0, 0).expect("valid precursor")
}

fn zero_intensity_spectrum() -> GenericSpectrum {
    // Zero-intensity peaks are rejected; return an empty spectrum.
    GenericSpectrum::try_with_capacity(100.0, 0).expect("valid precursor")
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

// LinearEntropy self-similarity (well-separated spectra only)
test_self_similarity!(
    linear_entropy_self_well_separated,
    LinearEntropy::weighted(0.1).unwrap(),
    well_separated_a
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
    linear_entropy_symmetry,
    LinearEntropy::weighted(0.1).unwrap(),
    well_separated_a,
    well_separated_b
);
test_symmetry!(
    modified_linear_entropy_symmetry,
    ModifiedLinearEntropy::weighted(0.1).unwrap(),
    well_separated_a,
    well_separated_b
);

// ========== Empty / zero spectra edge cases ==========

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
    let linear = LinearEntropy::weighted(0.25).expect("valid scorer config");
    assert_eq!(linear.mz_tolerance(), 0.25);
    assert!(linear.is_weighted());

    let modified_linear = ModifiedLinearEntropy::unweighted(0.4).expect("valid scorer config");
    assert_eq!(modified_linear.mz_tolerance(), 0.4);
    assert!(!modified_linear.is_weighted());
}

// ========== Modified variants: precursor tolerance invariant ==========
// When |precursor_delta| <= tolerance, modified score == direct counterpart.

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
fn all_modified_variants_self_similarity() {
    let a = well_separated_a();

    let m_linear = ModifiedLinearEntropy::unweighted(0.1).unwrap();

    let (sim_ml, _): (f64, usize) = m_linear.similarity(&a, &a).expect("modified linear");

    assert!(
        (1.0 - sim_ml).abs() < 1e-10,
        "modified linear self-sim: {sim_ml}"
    );
}

// ========== Modified variants with actual precursor shift ==========

#[test]
fn modified_linear_variants_with_precursor_shift() {
    let a = well_separated_a(); // precursor 50.0
    let b = well_separated_shifted(); // precursor 60.0, same peaks

    let m_linear = ModifiedLinearEntropy::unweighted(0.1).unwrap();

    // Self-sim despite precursor shift (shift=10, well beyond tolerance=0.1)
    // but peaks are identical so direct matches should still give score=1
    let (sim_ml, _): (f64, usize) = m_linear.similarity(&a, &b).expect("modified linear");

    assert!(
        (1.0 - sim_ml).abs() < 1e-10,
        "modified linear with shift: {sim_ml}"
    );
}
