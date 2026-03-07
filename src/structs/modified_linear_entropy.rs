//! Implementation of modified linear-time spectral entropy similarity.
//!
//! Extends [`super::LinearEntropy`] by also matching fragment peaks shifted by
//! the precursor mass difference when that shift exceeds the configured
//! tolerance. Direct and shifted match candidates are merged and resolved via
//! optimal DP-based assignment on the conflict graph's path components.

use geometric_traits::prelude::ScalarSimilarity;

use super::cosine_common::{
    ensure_finite, optimal_modified_linear_matches, validate_well_separated,
};
use super::entropy_common::{
    entropy_pair, entropy_score_pairs, finalize_entropy_score, impl_entropy_config_api,
    impl_entropy_spectral_similarity, prepare_entropy_peaks,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Modified linear-time spectral entropy similarity.
///
/// Combines direct and precursor-shifted peak matches from two linear sweeps
/// when `|precursor_delta| > mz_tolerance`, then resolves conflicts via
/// optimal DP-based assignment. Requires the same strict well-separated
/// precondition as [`super::LinearEntropy`]
/// (consecutive peaks > `2 * mz_tolerance`).
pub struct ModifiedLinearEntropy {
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    weighted: bool,
}

impl_entropy_config_api!(ModifiedLinearEntropy, "modified linear entropy similarity");
impl_entropy_spectral_similarity!(ModifiedLinearEntropy);

impl<S1, S2> ScalarSimilarity<S1, S2> for ModifiedLinearEntropy
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

        let left_prec = ensure_finite(left.precursor_mz(), "left_precursor_mz")?;
        let right_prec = ensure_finite(right.precursor_mz(), "right_precursor_mz")?;

        let selected = optimal_modified_linear_matches(
            &left_peaks.mz,
            &right_peaks.mz,
            tolerance,
            left_prec,
            right_prec,
            |i, j| entropy_pair(left_peaks.int[i], right_peaks.int[j]),
        );

        let (raw_score, n_matches) =
            entropy_score_pairs(&selected, &left_peaks.int, &right_peaks.int)?;

        finalize_entropy_score(raw_score, n_matches)
    }
}
