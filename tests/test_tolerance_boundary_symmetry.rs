//! Regression tests for tolerance-boundary symmetry in cosine matchers.

use mass_spectrometry::prelude::{
    GenericSpectrum, HungarianCosine, ModifiedHungarianCosine, ScalarSimilarity, SpectrumAlloc,
    SpectrumMut,
};

fn one_peak_spectrum(mz: f32, precursor_mz: f32) -> GenericSpectrum<f32, f32> {
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
    let scorer = HungarianCosine::new(1.0_f32, 1.0_f32, 0.05_f32).expect("valid scorer config");

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
    let scorer =
        ModifiedHungarianCosine::new(1.0_f32, 1.0_f32, 0.05_f32).expect("valid scorer config");

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
fn modified_is_symmetric_on_shifted_boundary_single_peak() {
    let left = one_peak_spectrum(80.0, 500.0);
    let right = one_peak_spectrum(90.05, 510.0);
    let scorer =
        ModifiedHungarianCosine::new(1.0_f32, 1.0_f32, 0.05_f32).expect("valid scorer config");

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
