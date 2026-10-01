//! Tests for similarity scorer configuration validation.

use mass_spectrometry::prelude::{
    HungarianCosine, LinearEntropy, ModifiedHungarianCosine, SimilarityConfigError,
};

#[test]
fn hungarian_cosine_rejects_negative_tolerance() {
    let result = HungarianCosine::new(1.0, 1.0, -0.1);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn modified_hungarian_cosine_rejects_negative_tolerance() {
    let result = ModifiedHungarianCosine::new(1.0, 1.0, -0.1);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn entropy_similarity_rejects_negative_tolerance() {
    let result = LinearEntropy::weighted(-0.01);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn hungarian_cosine_rejects_nan_mz_power() {
    let result = HungarianCosine::new(f64::NAN, 1.0, 0.1);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NonFiniteParameter("mz_power"))
    ));
}

#[test]
fn modified_hungarian_cosine_rejects_infinite_intensity_power() {
    let result = ModifiedHungarianCosine::new(1.0, f64::INFINITY, 0.1);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NonFiniteParameter("intensity_power"))
    ));
}

#[test]
fn entropy_similarity_rejects_nan_tolerance() {
    let result = LinearEntropy::unweighted(f64::NAN);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NonFiniteParameter("mz_tolerance"))
    ));
}
