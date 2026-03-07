//! Implementation of linear-time cosine similarity for mass spectra.
//!
//! When consecutive peaks within each spectrum are more than 2x the matching
//! tolerance apart, each peak can match at most one peak in the other spectrum.
//! This eliminates assignment ambiguity, allowing a simple two-pointer sweep
//! to produce the same result as Hungarian (optimal) assignment in O(n+m) time.

use geometric_traits::prelude::ScalarSimilarity;

use super::cosine_common::{
    CosineConfig, ensure_finite, finalize_similarity_score, impl_cosine_wrapper_config_api,
    linear_cosine_sweep, prepare_peak_products, validate_well_separated,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Linear-time cosine similarity for mass spectra.
///
/// Equivalent to [`super::HungarianCosine`] when spectra are *well-separated*:
/// consecutive peaks within each spectrum must be greater than
/// `2 * mz_tolerance`.
/// Under this invariant the two-pointer sweep is provably optimal.
///
/// Returns an error when the strict peak-spacing precondition is violated.
pub struct LinearCosine {
    config: CosineConfig,
}

impl_cosine_wrapper_config_api!(
    LinearCosine,
    "the linear cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl<S1, S2> ScalarSimilarity<S1, S2> for LinearCosine
where
    S1: Spectrum,
    S2: Spectrum,
{
    type Similarity = Result<(f64, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks =
            prepare_peak_products(left, self.config.mz_power(), self.config.intensity_power())?;
        let right_peaks =
            prepare_peak_products(right, self.config.mz_power(), self.config.intensity_power())?;

        if left_peaks.max == 0.0 || right_peaks.max == 0.0 {
            return Ok((0.0, 0));
        }

        let tolerance = ensure_finite(self.config.mz_tolerance(), "mz_tolerance")?;

        // Collect mz as f64.
        let left_mz: alloc::vec::Vec<f64> = left.mz().collect();
        let right_mz: alloc::vec::Vec<f64> = right.mz().collect();

        validate_well_separated(&left_mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_mz, tolerance, "right spectrum")?;

        let (score_sum, n_matches) = linear_cosine_sweep(
            &left_mz,
            &right_mz,
            &left_peaks.products,
            &right_peaks.products,
            tolerance,
            0.0,
        );

        finalize_similarity_score(score_sum, n_matches, left_peaks.norm, right_peaks.norm)
    }
}

impl<S1, S2> ScalarSpectralSimilarity<S1, S2> for LinearCosine
where
    S1: Spectrum,
    S2: Spectrum,
{
}
