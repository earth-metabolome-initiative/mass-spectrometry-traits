//! Regression tests for `Spectrum::matching_peaks` input validation.

use mass_spectrometry::prelude::{
    GenericSpectrum, SimilarityComputationError, Spectrum, SpectrumAlloc, SpectrumMut,
};

fn spectrum_from_peaks(precursor_mz: f32, peaks: &[(f32, f32)]) -> GenericSpectrum<f32, f32> {
    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz, peaks.len());
    for &(mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .expect("test peaks must be sorted by m/z");
    }
    spectrum
}

#[test]
fn matching_peaks_rejects_nan_tolerance() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .matching_peaks(&right, f32::NAN)
        .expect_err("NaN tolerance should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("mz_tolerance")
    );
}

#[test]
fn matching_peaks_rejects_infinite_tolerance() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .matching_peaks(&right, f32::INFINITY)
        .expect_err("infinite tolerance should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("mz_tolerance")
    );
}

#[test]
fn matching_peaks_rejects_non_finite_left_mz() {
    let left = spectrum_from_peaks(100.0, &[(f32::INFINITY, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .matching_peaks(&right, 0.1)
        .expect_err("non-finite left mz should be rejected");
    assert_eq!(error, SimilarityComputationError::NonFiniteValue("mz"));
}

#[test]
fn matching_peaks_rejects_non_finite_right_mz() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(f32::INFINITY, 1.0)]);

    let error = left
        .matching_peaks(&right, 0.1)
        .expect_err("non-finite right mz should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("other_mz")
    );
}
