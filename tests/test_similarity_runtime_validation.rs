use geometric_traits::prelude::ScalarSimilarity;
use mass_spectrometry::prelude::{
    EntropySimilarity, GenericSpectrum, GreedyCosine, HungarianCosine, ModifiedHungarianCosine,
    ModifiedLinearCosine, SimilarityComputationError, SimilarityConfigError, Spectrum,
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

fn raw_spectrum(precursor_mz: f32, peaks: &[(f32, f32)]) -> RawSpectrum {
    RawSpectrum {
        precursor_mz,
        peaks: peaks.to_vec(),
    }
}

#[test]
fn hungarian_cosine_rejects_negative_tolerance_at_construction() {
    let result = HungarianCosine::new(1.0_f32, 1.0_f32, -0.1_f32);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn modified_hungarian_cosine_rejects_negative_tolerance_at_construction() {
    let result = ModifiedHungarianCosine::new(1.0_f32, 1.0_f32, -0.1_f32);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

fn large_peak_spectra() -> (GenericSpectrum<f32, f32>, GenericSpectrum<f32, f32>) {
    let mut left = GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    let mut right =
        GenericSpectrum::with_capacity(100.0_f32, 1).expect("valid spectrum allocation");
    left.add_peak(1.0e20_f32, 1.0e20_f32)
        .expect("finite values should be accepted");
    right
        .add_peak(1.0e20_f32, 1.0e20_f32)
        .expect("finite values should be accepted");
    (left, right)
}

#[test]
fn greedy_cosine_rejects_non_finite_peak_product_at_runtime() {
    let (left, right) = large_peak_spectra();
    let scorer = GreedyCosine::new(3.0_f32, 3.0_f32, 0.1_f32).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("overflowed peak product should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("peak_product")
    );
}

#[test]
fn hungarian_cosine_rejects_non_finite_peak_product_at_runtime() {
    let (left, right) = large_peak_spectra();
    let scorer = HungarianCosine::new(3.0_f32, 3.0_f32, 0.1_f32).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("overflowed peak product should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("peak_product")
    );
}

#[test]
fn modified_linear_cosine_rejects_non_finite_left_precursor_at_runtime() {
    let left = raw_spectrum(f32::NAN, &[(100.0, 1.0)]);
    let right = raw_spectrum(100.0, &[(100.0, 1.0)]);
    let scorer = ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("non-finite left precursor should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("left_precursor_mz")
    );
}

#[test]
fn modified_linear_cosine_rejects_non_finite_right_precursor_at_runtime() {
    let left = raw_spectrum(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum(f32::NAN, &[(100.0, 1.0)]);
    let scorer = ModifiedLinearCosine::new(1.0_f32, 1.0_f32, 0.1_f32).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("non-finite right precursor should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("right_precursor_mz")
    );
}

#[test]
fn entropy_similarity_rejects_non_finite_left_intensity_at_runtime() {
    let left = raw_spectrum(100.0, &[(100.0, f32::NAN)]);
    let right = raw_spectrum(100.0, &[(100.0, 1.0)]);
    let scorer = EntropySimilarity::unweighted(0.1_f32).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("non-finite left intensity should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("left_intensity")
    );
}

#[test]
fn entropy_similarity_rejects_non_finite_right_intensity_at_runtime() {
    let left = raw_spectrum(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum(100.0, &[(100.0, f32::INFINITY)]);
    let scorer = EntropySimilarity::unweighted(0.1_f32).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("non-finite right intensity should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("right_intensity")
    );
}
