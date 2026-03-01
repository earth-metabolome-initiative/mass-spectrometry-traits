//! Regression tests for `Spectrum::matching_peaks` input validation.

use mass_spectrometry::prelude::{
    GenericSpectrum, GenericSpectrumMutationError, SimilarityComputationError, Spectrum,
    SpectrumAlloc, SpectrumMut,
};

#[derive(Clone)]
struct RawSpectrum {
    precursor_mz: f32,
    peaks: Vec<(f32, f32)>,
}

impl Spectrum for RawSpectrum {
    type Intensity = f32;
    type Mz = f32;
    type SortedIntensitiesIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f32, f32)>, fn(&(f32, f32)) -> f32>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f32, f32)>, fn(&(f32, f32)) -> f32>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, (f32, f32)>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.peaks.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.peaks.iter().map(|peak| peak.1)
    }

    fn intensity_nth(&self, n: usize) -> Self::Intensity {
        self.peaks[n].1
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.peaks.iter().map(|peak| peak.0)
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.peaks[index..].iter().map(|peak| peak.0)
    }

    fn mz_nth(&self, n: usize) -> Self::Mz {
        self.peaks[n].0
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.peaks.iter().copied()
    }

    fn peak_nth(&self, n: usize) -> (Self::Mz, Self::Intensity) {
        self.peaks[n]
    }

    fn precursor_mz(&self) -> Self::Mz {
        self.precursor_mz
    }
}

fn spectrum_from_peaks(precursor_mz: f32, peaks: &[(f32, f32)]) -> GenericSpectrum<f32, f32> {
    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz, peaks.len())
        .expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .expect("test peaks must be sorted by m/z");
    }
    spectrum
}

fn raw_spectrum_from_peaks(precursor_mz: f32, peaks: &[(f32, f32)]) -> RawSpectrum {
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
        .modified_matching_peaks(&right, -0.1, 0.0)
        .expect_err("negative tolerance should be rejected");
    assert_eq!(error, SimilarityComputationError::NegativeTolerance);
}

#[test]
fn matching_peaks_rejects_non_finite_left_mz() {
    let left = raw_spectrum_from_peaks(100.0, &[(f32::INFINITY, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = left
        .matching_peaks(&right, 0.1)
        .expect_err("non-finite left mz should be rejected");
    assert_eq!(error, SimilarityComputationError::NonFiniteValue("mz"));
}

#[test]
fn matching_peaks_rejects_non_finite_right_mz() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum_from_peaks(100.0, &[(f32::INFINITY, 1.0)]);

    let error = left
        .matching_peaks(&right, 0.1)
        .expect_err("non-finite right mz should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("other_mz")
    );
}

#[test]
fn generic_spectrum_rejects_negative_intensity() {
    let mut spectrum =
        GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(100.0_f32, -1.0_f32)
        .expect_err("negative intensity should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NegativeIntensity);
}

#[test]
fn generic_spectrum_rejects_non_finite_intensity() {
    let mut spectrum =
        GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(100.0_f32, f32::NAN)
        .expect_err("non-finite intensity should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NonFiniteIntensity);
}

#[test]
fn generic_spectrum_rejects_non_finite_mz() {
    let mut spectrum =
        GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(f32::INFINITY, 1.0_f32)
        .expect_err("non-finite mz should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NonFiniteMz);
}

#[test]
fn generic_spectrum_rejects_negative_mz() {
    let mut spectrum =
        GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    let error = spectrum
        .add_peak(-1.0_f32, 1.0_f32)
        .expect_err("negative mz should be rejected");
    assert_eq!(error, GenericSpectrumMutationError::NegativeMz);
}

#[test]
fn generic_spectrum_try_with_capacity_rejects_non_finite_precursor_mz() {
    let error = match GenericSpectrum::<f32, f32>::try_with_capacity(f32::INFINITY, 1) {
        Err(error) => error,
        Ok(_) => panic!("non-finite precursor_mz should be rejected"),
    };
    assert_eq!(error, GenericSpectrumMutationError::NonFinitePrecursorMz);
}

#[test]
fn generic_spectrum_with_capacity_rejects_negative_precursor_mz() {
    let error = match GenericSpectrum::<f32, f32>::with_capacity(-1.0_f32, 1) {
        Err(error) => error,
        Ok(_) => panic!("negative precursor_mz should be rejected"),
    };
    assert_eq!(error, GenericSpectrumMutationError::NegativePrecursorMz);
}
