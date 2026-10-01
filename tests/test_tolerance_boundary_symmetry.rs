//! Regression tests for tolerance-boundary symmetry in cosine matchers.

use mass_spectrometry::prelude::{
    GenericSpectrum, HungarianCosine, ModifiedHungarianCosine, ScalarSimilarity, SpectrumAlloc,
    SpectrumMut,
};

fn one_peak_spectrum(mz: f64, precursor_mz: f64) -> GenericSpectrum {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor_mz, 1).expect("valid spectrum allocation");
    spectrum
        .add_peak(mz, 1.0)
        .expect("single-peak spectrum must stay sorted");
    spectrum
}

#[test]
fn hungarian_is_symmetric_on_tolerance_boundary_single_peak() {
    let left = one_peak_spectrum(3.96, 500.0);
    let right = one_peak_spectrum(4.01, 500.0);
    let scorer = HungarianCosine::new(1.0, 1.0, 0.05).expect("valid scorer config");

    let (score_ab, matches_ab) = scorer
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (score_ba, matches_ba) = scorer
        .similarity(&right, &left)
        .expect("similarity computation should succeed");

    assert!(
        (score_ab - score_ba).abs() < 1e-7,
        "boundary asymmetry: sim(A,B)={score_ab} vs sim(B,A)={score_ba}"
    );
    assert_eq!(
        matches_ab, matches_ba,
        "boundary asymmetry: matches(A,B)={matches_ab} vs matches(B,A)={matches_ba}"
    );
}

#[test]
fn modified_is_symmetric_on_tolerance_boundary_single_peak() {
    let left = one_peak_spectrum(3.96, 500.0);
    let right = one_peak_spectrum(4.01, 500.0);
    let scorer = ModifiedHungarianCosine::new(1.0, 1.0, 0.05).expect("valid scorer config");

    let (score_ab, matches_ab) = scorer
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (score_ba, matches_ba) = scorer
        .similarity(&right, &left)
        .expect("similarity computation should succeed");

    assert!(
        (score_ab - score_ba).abs() < 1e-7,
        "boundary asymmetry: sim(A,B)={score_ab} vs sim(B,A)={score_ba}"
    );
    assert_eq!(
        matches_ab, matches_ba,
        "boundary asymmetry: matches(A,B)={matches_ab} vs matches(B,A)={matches_ba}"
    );
}

#[test]
fn hungarian_matches_at_exact_fp_tolerance_boundary() {
    // Regression: 71.0605 - 71.0505 = 0.010000000000005116 in f64,
    // which exceeds tolerance=0.01 under abs(diff) <= tol but should
    // match (matchms reference accepts it via bound-based comparison).
    let left = one_peak_spectrum(71.0605, 500.0);
    let right = one_peak_spectrum(71.0505, 500.0);
    let scorer = HungarianCosine::new(1.0, 1.0, 0.01).expect("valid scorer config");

    let (_score, matches) = scorer
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    assert_eq!(
        matches, 1,
        "peaks 0.01 Da apart in decimal must match at tolerance=0.01"
    );
}

#[test]
fn modified_matches_at_exact_fp_tolerance_boundary() {
    let left = one_peak_spectrum(71.0605, 500.0);
    let right = one_peak_spectrum(71.0505, 500.0);
    let scorer = ModifiedHungarianCosine::new(1.0, 1.0, 0.01).expect("valid scorer config");

    let (_score, matches) = scorer
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    assert_eq!(
        matches, 1,
        "peaks 0.01 Da apart in decimal must match at tolerance=0.01"
    );
}

#[test]
fn modified_is_symmetric_on_shifted_boundary_single_peak() {
    let left = one_peak_spectrum(80.0, 500.0);
    let right = one_peak_spectrum(90.05, 510.0);
    let scorer = ModifiedHungarianCosine::new(1.0, 1.0, 0.05).expect("valid scorer config");

    let (score_ab, matches_ab) = scorer
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (score_ba, matches_ba) = scorer
        .similarity(&right, &left)
        .expect("similarity computation should succeed");

    assert!(
        (score_ab - score_ba).abs() < 1e-7,
        "shifted boundary asymmetry: sim(A,B)={score_ab} vs sim(B,A)={score_ba}"
    );
    assert_eq!(
        matches_ab, matches_ba,
        "shifted boundary asymmetry: matches(A,B)={matches_ab} vs matches(B,A)={matches_ba}"
    );
}
