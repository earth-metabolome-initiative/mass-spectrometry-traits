//! Tests for similarity scorer configuration validation.

use mass_spectrometry::prelude::{
    EntropySimilarity, ExactCosine, ModifiedCosine, SimilarityConfigError,
};

#[test]
fn exact_cosine_rejects_negative_tolerance() {
    let result = ExactCosine::new(1.0_f32, 1.0_f32, -0.1_f32);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn modified_cosine_rejects_negative_tolerance() {
    let result = ModifiedCosine::new(1.0_f32, 1.0_f32, -0.1_f32);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn entropy_similarity_rejects_negative_tolerance() {
    let result = EntropySimilarity::weighted(-0.01_f32);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NegativeTolerance)
    ));
}

#[test]
fn exact_cosine_rejects_nan_mz_power() {
    let result = ExactCosine::new(f32::NAN, 1.0_f32, 0.1_f32);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NonFiniteParameter("mz_power"))
    ));
}

#[test]
fn modified_cosine_rejects_infinite_intensity_power() {
    let result = ModifiedCosine::new(1.0_f32, f32::INFINITY, 0.1_f32);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NonFiniteParameter("intensity_power"))
    ));
}

#[test]
fn entropy_similarity_rejects_nan_tolerance() {
    let result = EntropySimilarity::unweighted(f32::NAN);
    assert!(matches!(
        result,
        Err(SimilarityConfigError::NonFiniteParameter("mz_tolerance"))
    ));
}
