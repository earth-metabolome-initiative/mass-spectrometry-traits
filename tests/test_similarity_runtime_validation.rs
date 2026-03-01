use geometric_traits::prelude::ScalarSimilarity;
use mass_spectrometry::prelude::{
    GenericSpectrum, GreedyCosine, HungarianCosine, ModifiedHungarianCosine,
    SimilarityComputationError, SimilarityConfigError, SpectrumAlloc, SpectrumMut,
};

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
