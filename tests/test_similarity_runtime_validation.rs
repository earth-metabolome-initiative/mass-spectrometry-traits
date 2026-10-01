use geometric_traits::prelude::ScalarSimilarity;
use mass_spectrometry::prelude::{
    HungarianCosine, LinearEntropy, ModifiedHungarianCosine, ModifiedLinearCosine,
    SimilarityComputationError, SimilarityConfigError, Spectrum,
};

#[derive(Clone)]
struct RawSpectrum {
    precursor_mz: f64,
    peaks: Vec<(f64, f64)>,
}

impl Spectrum for RawSpectrum {
    type Precision = f64;

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

fn raw_spectrum(precursor_mz: f64, peaks: &[(f64, f64)]) -> RawSpectrum {
    RawSpectrum {
        precursor_mz,
        peaks: peaks.to_vec(),
    }
}

#[test]
fn hungarian_cosine_rejects_negative_tolerance_at_construction() {
    let result = HungarianCosine::new(1.0, 1.0, -0.1);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn modified_hungarian_cosine_rejects_negative_tolerance_at_construction() {
    let result = ModifiedHungarianCosine::new(1.0, 1.0, -0.1);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

fn large_peak_spectra() -> (RawSpectrum, RawSpectrum) {
    // Use RawSpectrum to bypass mz range validation — 1e200 exceeds MAX_MZ
    // but is needed to trigger peak product overflow with high powers in f64.
    // mz^3 * intensity^3 = (1e200)^3 * (1e200)^3 = 1e1200 → overflow.
    let left = raw_spectrum(100.0, &[(1.0e200, 1.0e200)]);
    let right = raw_spectrum(100.0, &[(1.0e200, 1.0e200)]);
    (left, right)
}

#[test]
fn hungarian_cosine_rejects_non_finite_peak_product_at_runtime() {
    let (left, right) = large_peak_spectra();
    let scorer = HungarianCosine::new(3.0, 3.0, 0.1).expect("valid scorer config");

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
    let left = raw_spectrum(f64::NAN, &[(100.0, 1.0)]);
    let right = raw_spectrum(100.0, &[(100.0, 1.0)]);
    let scorer = ModifiedLinearCosine::new(1.0, 1.0, 0.1).expect("valid scorer config");

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
    let right = raw_spectrum(f64::NAN, &[(100.0, 1.0)]);
    let scorer = ModifiedLinearCosine::new(1.0, 1.0, 0.1).expect("valid scorer config");

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
    let left = raw_spectrum(100.0, &[(100.0, f64::NAN)]);
    let right = raw_spectrum(100.0, &[(100.0, 1.0)]);
    let scorer = LinearEntropy::unweighted(0.1).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("non-finite left intensity should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("intensity")
    );
}

#[test]
fn entropy_similarity_rejects_non_finite_right_intensity_at_runtime() {
    let left = raw_spectrum(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum(100.0, &[(100.0, f64::INFINITY)]);
    let scorer = LinearEntropy::unweighted(0.1).expect("valid scorer config");

    let error = scorer
        .similarity(&left, &right)
        .expect_err("non-finite right intensity should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("intensity")
    );
}
