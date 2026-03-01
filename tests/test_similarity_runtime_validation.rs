use mass_spectrometry::prelude::{
    EntropySimilarity, HungarianCosine, ModifiedHungarianCosine, ScalarSimilarity,
    SimilarityComputationError, Spectrum,
};

struct RawSpectrum {
    precursor_mz: f32,
    mz: Vec<f32>,
    intensity: Vec<f32>,
}

impl RawSpectrum {
    fn new(precursor_mz: f32, mz: Vec<f32>, intensity: Vec<f32>) -> Self {
        assert_eq!(mz.len(), intensity.len());
        Self {
            precursor_mz,
            mz,
            intensity,
        }
    }
}

impl Spectrum for RawSpectrum {
    type Intensity = f32;
    type Mz = f32;
    type SortedIntensitiesIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, f32>>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, f32>>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = core::iter::Zip<Self::SortedMzIter<'a>, Self::SortedIntensitiesIter<'a>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.mz.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.intensity.iter().copied()
    }

    fn intensity_nth(&self, n: usize) -> Self::Intensity {
        self.intensity[n]
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.mz.iter().copied()
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.mz[index..].iter().copied()
    }

    fn mz_nth(&self, n: usize) -> Self::Mz {
        self.mz[n]
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.mz().zip(self.intensities())
    }

    fn peak_nth(&self, n: usize) -> (Self::Mz, Self::Intensity) {
        (self.mz_nth(n), self.intensity_nth(n))
    }

    fn precursor_mz(&self) -> Self::Mz {
        self.precursor_mz
    }
}

#[test]
fn hungarian_cosine_rejects_non_finite_peak_product() {
    let left = RawSpectrum::new(100.0, vec![50.0], vec![f32::NAN]);
    let right = RawSpectrum::new(100.0, vec![50.0], vec![1.0]);
    let scorer = HungarianCosine::new(1.0_f32, 0.5_f32, 0.1_f32).expect("valid scorer config");

    let result = scorer.similarity(&left, &right);
    assert!(matches!(
        result,
        Err(SimilarityComputationError::NonFiniteValue("peak_product"))
    ));
}

#[test]
fn modified_hungarian_cosine_rejects_non_finite_peak_product() {
    let left = RawSpectrum::new(100.0, vec![50.0], vec![f32::NAN]);
    let right = RawSpectrum::new(101.0, vec![50.0], vec![1.0]);
    let scorer =
        ModifiedHungarianCosine::new(1.0_f32, 0.5_f32, 0.1_f32).expect("valid scorer config");

    let result = scorer.similarity(&left, &right);
    assert!(matches!(
        result,
        Err(SimilarityComputationError::NonFiniteValue("peak_product"))
    ));
}

#[test]
fn entropy_similarity_rejects_non_finite_input() {
    let left = RawSpectrum::new(100.0, vec![50.0], vec![f32::INFINITY]);
    let right = RawSpectrum::new(100.0, vec![50.0], vec![1.0]);
    let scorer = EntropySimilarity::unweighted(0.1_f32).expect("valid scorer config");

    let result = scorer.similarity(&left, &right);
    assert!(matches!(
        result,
        Err(SimilarityComputationError::NonFiniteValue("left_intensity"))
    ));
}

#[test]
fn hungarian_cosine_unchecked_rejects_negative_tolerance_at_runtime() {
    let left = RawSpectrum::new(100.0, vec![50.0], vec![1.0]);
    let right = RawSpectrum::new(100.0, vec![50.0], vec![1.0]);
    let scorer = HungarianCosine::new_unchecked(1.0_f32, 1.0_f32, -0.1_f32);

    let result = scorer.similarity(&left, &right);
    assert!(matches!(
        result,
        Err(SimilarityComputationError::NegativeTolerance)
    ));
}

#[test]
fn modified_hungarian_cosine_unchecked_rejects_negative_tolerance_at_runtime() {
    let left = RawSpectrum::new(100.0, vec![50.0], vec![1.0]);
    let right = RawSpectrum::new(101.0, vec![50.0], vec![1.0]);
    let scorer = ModifiedHungarianCosine::new_unchecked(1.0_f32, 1.0_f32, -0.1_f32);

    let result = scorer.similarity(&left, &right);
    assert!(matches!(
        result,
        Err(SimilarityComputationError::NegativeTolerance)
    ));
}
