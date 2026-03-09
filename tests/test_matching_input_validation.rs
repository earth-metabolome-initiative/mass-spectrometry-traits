//! Regression tests for `Spectrum::matching_peaks` input validation.

use geometric_traits::prelude::*;
use mass_spectrometry::prelude::{
    ELECTRON_MASS, GenericSpectrum, GenericSpectrumMutationError, SimilarityComputationError,
    Spectrum, SpectrumAlloc, SpectrumMut,
};

#[derive(Clone)]
struct RawSpectrum {
    precursor_mz: f64,
    peaks: Vec<(f64, f64)>,
}

impl Spectrum for RawSpectrum {
    type SortedIntensitiesIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, (f64, f64)>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.peaks.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.peaks.iter().map(|peak| peak.1)
    }

    fn intensity_nth(&self, n: usize) -> f64 {
        self.peaks[n].1
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.peaks.iter().map(|peak| peak.0)
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.peaks[index..].iter().map(|peak| peak.0)
    }

    fn mz_nth(&self, n: usize) -> f64 {
        self.peaks[n].0
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.peaks.iter().copied()
    }

    fn peak_nth(&self, n: usize) -> (f64, f64) {
        self.peaks[n]
    }

    fn precursor_mz(&self) -> f64 {
        self.precursor_mz
    }
}

fn spectrum_from_peaks(precursor_mz: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz, peaks.len())
        .expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .expect("test peaks must be sorted by m/z");
    }
    spectrum
}

fn raw_spectrum_from_peaks(precursor_mz: f64, peaks: &[(f64, f64)]) -> RawSpectrum {
    RawSpectrum {
        precursor_mz,
        peaks: peaks.to_vec(),
    }
}

#[test]
fn matching_peaks_rejects_nan_tolerance() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .matching_peaks(&right, f64::NAN)
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
        .matching_peaks(&right, f64::INFINITY)
        .expect_err("infinite tolerance should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("mz_tolerance")
    );
}

#[test]
fn matching_peaks_rejects_negative_tolerance() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .matching_peaks(&right, -0.1)
        .expect_err("negative tolerance should be rejected");
    assert_eq!(error, SimilarityComputationError::NegativeTolerance);
}

#[test]
fn modified_matching_peaks_rejects_negative_tolerance() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .modified_matching_peaks(&right, -0.1, 100.0, 100.0)
        .expect_err("negative tolerance should be rejected");
    assert_eq!(error, SimilarityComputationError::NegativeTolerance);
}

#[test]
fn matching_peaks_rejects_non_finite_left_mz() {
    let left = raw_spectrum_from_peaks(100.0, &[(f64::INFINITY, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .matching_peaks(&right, 0.1)
        .expect_err("non-finite left mz should be rejected");
    assert_eq!(error, SimilarityComputationError::NonFiniteValue("left_mz"));
}

#[test]
fn matching_peaks_rejects_non_finite_right_mz() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum_from_peaks(100.0, &[(f64::INFINITY, 1.0)]);

    let error = left
        .matching_peaks(&right, 0.1)
        .expect_err("non-finite right mz should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("right_mz")
    );
}

#[test]
fn modified_matching_peaks_rejects_non_finite_left_mz() {
    let left = raw_spectrum_from_peaks(100.0, &[(f64::INFINITY, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .modified_matching_peaks(&right, 0.1, 100.0, 100.0)
        .expect_err("non-finite left mz should be rejected");
    assert_eq!(error, SimilarityComputationError::NonFiniteValue("left_mz"));
}

#[test]
fn modified_matching_peaks_rejects_non_finite_right_mz() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum_from_peaks(100.0, &[(f64::INFINITY, 1.0)]);

    let error = left
        .modified_matching_peaks(&right, 0.1, 100.0, 100.0)
        .expect_err("non-finite right mz should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("right_mz")
    );
}

#[test]
fn matching_peaks_empty_inputs_preserve_shape() {
    let empty = spectrum_from_peaks(100.0, &[]);
    let nonempty = spectrum_from_peaks(100.0, &[(100.0, 1.0), (200.0, 2.0)]);

    let direct_empty_left = empty
        .matching_peaks(&nonempty, 0.1)
        .expect("matching graph should build");
    assert_eq!(direct_empty_left.number_of_rows(), 0);
    assert_eq!(direct_empty_left.number_of_columns(), nonempty.len() as u32);
    assert_eq!(direct_empty_left.number_of_defined_values(), 0);

    let direct_empty_right = nonempty
        .matching_peaks(&empty, 0.1)
        .expect("matching graph should build");
    assert_eq!(direct_empty_right.number_of_rows(), nonempty.len() as u32);
    assert_eq!(direct_empty_right.number_of_columns(), 0);
    assert_eq!(direct_empty_right.number_of_defined_values(), 0);

    let modified_empty_left = empty
        .modified_matching_peaks(&nonempty, 0.1, 100.0, 100.3)
        .expect("modified matching graph should build");
    assert_eq!(modified_empty_left.number_of_rows(), 0);
    assert_eq!(
        modified_empty_left.number_of_columns(),
        nonempty.len() as u32
    );
    assert_eq!(modified_empty_left.number_of_defined_values(), 0);

    let modified_empty_right = nonempty
        .modified_matching_peaks(&empty, 0.1, 100.3, 100.0)
        .expect("modified matching graph should build");
    assert_eq!(modified_empty_right.number_of_rows(), nonempty.len() as u32);
    assert_eq!(modified_empty_right.number_of_columns(), 0);
    assert_eq!(modified_empty_right.number_of_defined_values(), 0);
}

#[test]
fn modified_matching_peaks_deduplicates_direct_and_shifted_overlap() {
    let left = spectrum_from_peaks(200.15, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(200.0, &[(99.95, 1.0)]);

    let graph = left
        .modified_matching_peaks(&right, 0.1, 200.15, 200.0)
        .expect("modified matching graph should build");
    let cols: Vec<u32> = graph.sparse_row(0).collect();

    assert_eq!(cols, vec![0]);
    assert_eq!(graph.number_of_defined_values(), 1);
}

#[test]
fn modified_matching_peaks_reports_shifted_non_finite_right_value() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum_from_peaks(-f64::MAX, &[(f64::MAX, 1.0)]);

    let error = left
        .modified_matching_peaks(&right, 0.1, 100.2, -f64::MAX)
        .expect_err("shifted non-finite right mz should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("shifted_other_mz")
    );
}

#[test]
fn generic_spectrum_rejects_negative_intensity() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(100.0, -1.0)
        .expect_err("negative intensity should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NonPositiveIntensity);
}

#[test]
fn generic_spectrum_rejects_non_finite_intensity() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(100.0, f64::NAN)
        .expect_err("non-finite intensity should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NonFiniteIntensity);
}

#[test]
fn generic_spectrum_rejects_non_finite_mz() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(f64::INFINITY, 1.0)
        .expect_err("non-finite mz should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NonFiniteMz);
}

#[test]
fn generic_spectrum_rejects_duplicate_mz() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 2).expect("valid spectrum allocation");
    spectrum
        .add_peak(10.0, 1.0)
        .expect("first peak should be accepted");
    let error = spectrum
        .add_peak(10.0, 2.0)
        .expect_err("duplicate mz should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::DuplicateMz);
}

#[test]
fn generic_spectrum_rejects_descending_mz() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 2).expect("valid spectrum allocation");
    spectrum
        .add_peak(10.0, 1.0)
        .expect("first peak should be accepted");
    let error = spectrum
        .add_peak(9.0, 2.0)
        .expect_err("descending mz should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::UnsortedMz);
}

#[test]
fn generic_spectrum_rejects_negative_mz() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(-1.0, 1.0)
        .expect_err("negative mz should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::MzBelowMinimum);
}

#[test]
fn generic_spectrum_try_with_capacity_rejects_non_finite_precursor_mz() {
    let error = match GenericSpectrum::try_with_capacity(f64::INFINITY, 1) {
        Err(error) => error,
        Ok(_) => panic!("non-finite precursor_mz should be rejected"),
    };
    assert_eq!(error, GenericSpectrumMutationError::NonFinitePrecursorMz);
}

#[test]
fn generic_spectrum_with_capacity_rejects_negative_precursor_mz() {
    let error = match GenericSpectrum::with_capacity(-1.0, 1) {
        Err(error) => error,
        Ok(_) => panic!("negative precursor_mz should be rejected"),
    };
    assert_eq!(error, GenericSpectrumMutationError::PrecursorMzBelowMinimum);
}

#[test]
fn add_peak_rejects_mz_below_electron_mass() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(0.0, 1.0)
        .expect_err("mz below electron mass should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::MzBelowMinimum);
}

#[test]
fn add_peak_rejects_mz_above_max() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(3_000_000.0, 1.0)
        .expect_err("mz above MAX_MZ should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::MzAboveMaximum);
}

#[test]
fn add_peak_rejects_zero_intensity() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(100.0, 0.0)
        .expect_err("zero intensity should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NonPositiveIntensity);
}

#[test]
fn add_peak_accepts_mz_at_electron_mass_boundary() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    spectrum
        .add_peak(ELECTRON_MASS, 1.0)
        .expect("mz at ELECTRON_MASS boundary should be accepted");
}

#[test]
fn add_peak_accepts_mz_at_max_mz_boundary() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    spectrum
        .add_peak(2_000_000.0, 1.0)
        .expect("mz at MAX_MZ boundary should be accepted");
}

#[test]
fn add_peak_accepts_min_positive_intensity() {
    let mut spectrum = GenericSpectrum::with_capacity(100.0, 1).expect("valid spectrum allocation");
    spectrum
        .add_peak(100.0, f64::MIN_POSITIVE)
        .expect("f64::MIN_POSITIVE intensity should be accepted");
}

#[test]
fn try_with_capacity_rejects_precursor_below_electron_mass() {
    let error = match GenericSpectrum::try_with_capacity(0.0, 1) {
        Err(error) => error,
        Ok(_) => panic!("precursor below ELECTRON_MASS should be rejected"),
    };
    assert_eq!(error, GenericSpectrumMutationError::PrecursorMzBelowMinimum);
}

#[test]
fn try_with_capacity_rejects_precursor_above_max_mz() {
    let error = match GenericSpectrum::try_with_capacity(3_000_000.0, 1) {
        Err(error) => error,
        Ok(_) => panic!("precursor above MAX_MZ should be rejected"),
    };
    assert_eq!(error, GenericSpectrumMutationError::PrecursorMzAboveMaximum);
}
