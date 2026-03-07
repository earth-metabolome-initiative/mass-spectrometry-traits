//! Implementation of linear-time spectral entropy similarity.
//!
//! When spectra are *well-separated* (consecutive peaks > `2 * mz_tolerance`),
//! the two-pointer sweep is provably optimal.

use geometric_traits::prelude::ScalarSimilarity;

use super::cosine_common::{collect_linear_matches, ensure_finite, validate_well_separated};
use super::entropy_common::{
    entropy_score_pairs, finalize_entropy_score, impl_entropy_config_api,
    impl_entropy_spectral_similarity, prepare_entropy_peaks,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Linear-time spectral entropy similarity.
///
/// Requires spectra to be *well-separated*: consecutive peaks within each
/// spectrum must be greater than `2 * mz_tolerance`. Under this invariant the
/// two-pointer sweep is provably optimal.
///
/// Returns an error when the strict peak-spacing precondition is violated.
pub struct LinearEntropy {
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    weighted: bool,
}

impl_entropy_config_api!(LinearEntropy, "linear entropy similarity");
impl_entropy_spectral_similarity!(LinearEntropy);

impl<S1, S2> ScalarSimilarity<S1, S2> for LinearEntropy
where
    S1: Spectrum,
    S2: Spectrum,
{
    type Similarity = Result<(f64, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks =
            prepare_entropy_peaks(left, self.weighted, self.mz_power, self.intensity_power)?;
        let right_peaks =
            prepare_entropy_peaks(right, self.weighted, self.mz_power, self.intensity_power)?;

        if left_peaks.int.is_empty() || right_peaks.int.is_empty() {
            return Ok((0.0, 0));
        }

        let tolerance = ensure_finite(self.mz_tolerance, "mz_tolerance")?;

        validate_well_separated(&left_peaks.mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_peaks.mz, tolerance, "right spectrum")?;

        let pairs = collect_linear_matches(&left_peaks.mz, &right_peaks.mz, tolerance, 0.0);
        let (raw_score, n_matches) =
            entropy_score_pairs(&pairs, &left_peaks.int, &right_peaks.int)?;

        finalize_entropy_score(raw_score, n_matches)
    }
}
