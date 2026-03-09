//! Tests for the ms_entropy-style cleaning spectral processor.

use mass_spectrometry::prelude::{
    GenericSpectrum, MsEntropyCleanSpectrum, SimilarityConfigError, SpectralProcessor, Spectrum,
    SpectrumMut,
};

fn make_spectrum(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut s =
        GenericSpectrum::try_with_capacity(precursor, peaks.len()).expect("valid precursor");
    for &(mz, intensity) in peaks {
        s.add_peak(mz, intensity).expect("valid sorted peak");
    }
    s
}

#[test]
fn default_cleaning_centroids_with_weighted_mz_and_normalizes() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .build()
        .expect("valid default builder config");

    let input = make_spectrum(500.0, &[(100.0, 1.0), (100.04, 3.0), (150.0, 0.5)]);
    let out = cleaner.process(&input);

    assert_eq!(out.len(), 2);

    let (mz0, int0) = out.peak_nth(0);
    let (mz1, int1) = out.peak_nth(1);

    // Weighted centroid of first cluster: (100*1 + 100.04*3) / 4 = 100.03
    assert!(
        (mz0 - 100.03).abs() < 1e-4,
        "expected weighted centroid m/z"
    );
    assert!(
        (int0 - (4.0 / 4.5)).abs() < 1e-6,
        "expected normalized intensity"
    );

    assert!((mz1 - 150.0).abs() < 1e-6);
    assert!((int1 - (0.5 / 4.5)).abs() < 1e-6);
}

#[test]
fn noise_filter_removes_weak_peaks() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .build()
        .expect("valid default builder config");

    let input = make_spectrum(500.0, &[(100.0, 10.0), (200.0, 0.05)]);
    let out = cleaner.process(&input);

    assert_eq!(out.len(), 1);
    let (mz, intensity) = out.peak_nth(0);
    assert!((mz - 100.0).abs() < 1e-6);
    assert!((intensity - 1.0).abs() < 1e-6);
}

#[test]
fn top_n_keeps_most_intense_then_sorts_by_mz() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .noise_threshold(None)
        .expect("valid noise threshold")
        .max_peak_num(Some(2))
        .expect("valid max_peak_num")
        .build()
        .expect("valid builder config");

    let input = make_spectrum(500.0, &[(100.0, 5.0), (200.0, 3.0), (300.0, 1.0)]);
    let out = cleaner.process(&input);

    assert_eq!(out.len(), 2);
    let (mz0, int0) = out.peak_nth(0);
    let (mz1, int1) = out.peak_nth(1);

    assert!((mz0 - 100.0).abs() < 1e-6);
    assert!((mz1 - 200.0).abs() < 1e-6);
    assert!((int0 - (5.0 / 8.0)).abs() < 1e-6);
    assert!((int1 - (3.0 / 8.0)).abs() < 1e-6);
}

#[test]
fn ppm_centroiding_merges_close_peaks_when_da_disabled() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .noise_threshold(None)
        .expect("valid noise threshold")
        .normalize_intensity(false)
        .expect("normalize_intensity is always valid")
        .min_ms2_difference_in_da(-1.0)
        .expect("finite min_ms2_difference_in_da")
        .min_ms2_difference_in_ppm(Some(20.0))
        .expect("finite min_ms2_difference_in_ppm")
        .build()
        .expect("valid ppm builder config");

    let input = make_spectrum(500.0, &[(100.0, 1.0), (100.001, 1.0), (120.0, 2.0)]);
    let out = cleaner.process(&input);

    assert_eq!(out.len(), 2);

    let (mz0, int0) = out.peak_nth(0);
    let (mz1, int1) = out.peak_nth(1);

    // (100 + 100.001)/2
    assert!((mz0 - 100.0005).abs() < 1e-4);
    assert!((int0 - 2.0).abs() < 1e-6);

    assert!((mz1 - 120.0).abs() < 1e-6);
    assert!((int1 - 2.0).abs() < 1e-6);
}

#[test]
fn min_and_max_mz_filters_are_applied() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .noise_threshold(None)
        .expect("valid noise threshold")
        .normalize_intensity(false)
        .expect("normalize_intensity is always valid")
        .min_mz(Some(110.0))
        .expect("finite min_mz")
        .max_mz(Some(210.0))
        .expect("finite max_mz")
        .build()
        .expect("valid builder config");

    let input = make_spectrum(500.0, &[(100.0, 1.0), (150.0, 2.0), (250.0, 3.0)]);
    let out = cleaner.process(&input);

    assert_eq!(out.len(), 1);
    let (mz, intensity) = out.peak_nth(0);
    assert!((mz - 150.0).abs() < 1e-6);
    assert!((intensity - 2.0).abs() < 1e-6);
}

#[test]
fn builder_rejects_invalid_centroid_configuration() {
    let result = MsEntropyCleanSpectrum::builder()
        .min_ms2_difference_in_da(-1.0)
        .expect("finite min_ms2_difference_in_da")
        .min_ms2_difference_in_ppm(Some(-1.0))
        .expect("finite min_ms2_difference_in_ppm")
        .build();

    assert!(matches!(
        result,
        Err(SimilarityConfigError::InvalidParameter(
            "min_ms2_difference_in_da/min_ms2_difference_in_ppm"
        ))
    ));
}

#[test]
fn builder_defaults_and_getters_round_trip() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .build()
        .expect("valid default builder config");

    assert_eq!(cleaner.min_mz(), None);
    assert_eq!(cleaner.max_mz(), None);
    assert_eq!(cleaner.noise_threshold(), Some(0.01));
    assert_eq!(cleaner.min_ms2_difference_in_da(), 0.05);
    assert_eq!(cleaner.min_ms2_difference_in_ppm(), None);
    assert_eq!(cleaner.max_peak_num(), None);
    assert!(cleaner.normalize_intensity());
}

#[test]
fn cleaning_can_return_empty_after_filters_remove_everything() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .noise_threshold(Some(1.1))
        .expect("finite noise threshold")
        .build()
        .expect("valid builder config");

    let input = make_spectrum(500.0, &[(100.0, 1.0), (200.0, 0.5)]);
    let out = cleaner.process(&input);

    assert_eq!(out.len(), 0);
}

#[test]
fn getter_round_trip_preserves_custom_configuration() {
    let cleaner = MsEntropyCleanSpectrum::builder()
        .min_mz(Some(50.0))
        .expect("finite min_mz")
        .max_mz(Some(500.0))
        .expect("finite max_mz")
        .noise_threshold(None)
        .expect("noise threshold can be disabled")
        .min_ms2_difference_in_da(0.2)
        .expect("finite min_ms2_difference_in_da")
        .min_ms2_difference_in_ppm(Some(15.0))
        .expect("finite min_ms2_difference_in_ppm")
        .max_peak_num(Some(5))
        .expect("non-zero max_peak_num")
        .normalize_intensity(false)
        .expect("normalize_intensity is always valid")
        .build()
        .expect("valid builder config");

    assert_eq!(cleaner.min_mz(), Some(50.0));
    assert_eq!(cleaner.max_mz(), Some(500.0));
    assert_eq!(cleaner.noise_threshold(), None);
    assert_eq!(cleaner.min_ms2_difference_in_da(), 0.2);
    assert_eq!(cleaner.min_ms2_difference_in_ppm(), Some(15.0));
    assert_eq!(cleaner.max_peak_num(), Some(5));
    assert!(!cleaner.normalize_intensity());
}
