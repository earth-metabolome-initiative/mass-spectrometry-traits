//! Implementation of modified linear-time spectral entropy similarity.
//!
//! Extends [`super::LinearEntropy`] by also matching fragment peaks shifted by
//! the precursor mass difference when that shift exceeds the configured
//! tolerance. Direct and shifted match candidates are merged and resolved via
//! greedy assignment (sort by descending intensity product, pick best
//! available).

use geometric_traits::prelude::{Number, ScalarSimilarity};
use num_traits::{Float, Zero};

use super::cosine_common::{
    greedy_modified_linear_matches, to_f64_checked_for_computation, validate_well_separated,
};
use super::entropy_common::{
    entropy_score_pairs, finalize_entropy_score, impl_entropy_config_api,
    impl_entropy_spectral_similarity, prepare_entropy_peaks,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Modified linear-time spectral entropy similarity.
///
/// Combines direct and precursor-shifted peak matches from two linear sweeps
/// when `|precursor_delta| > mz_tolerance`, then resolves conflicts via greedy
/// assignment. Requires the same strict well-separated precondition as
/// [`super::LinearEntropy`] (consecutive peaks > `2 * mz_tolerance`).
pub struct ModifiedLinearEntropy<MZ> {
    mz_tolerance: MZ,
    weighted: bool,
}

impl_entropy_config_api!(ModifiedLinearEntropy, "modified linear entropy similarity");
impl_entropy_spectral_similarity!(ModifiedLinearEntropy);

impl<S1, S2> ScalarSimilarity<S1, S2> for ModifiedLinearEntropy<S1::Mz>
where
    S1::Mz: Float + Number,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks = prepare_entropy_peaks(left, self.weighted)?;
        let right_peaks = prepare_entropy_peaks(right, self.weighted)?;

        if left_peaks.int.is_empty() || right_peaks.int.is_empty() {
            return Ok((S1::Mz::zero(), 0));
        }

        let tolerance = to_f64_checked_for_computation(self.mz_tolerance, "mz_tolerance")?;

        validate_well_separated(&left_peaks.mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_peaks.mz, tolerance, "right spectrum")?;

        // Compute the precursor mass shift.
        let left_precursor =
            to_f64_checked_for_computation(left.precursor_mz(), "left_precursor_mz")?;
        let right_precursor =
            to_f64_checked_for_computation(right.precursor_mz(), "right_precursor_mz")?;
        let shift = left_precursor - right_precursor;

        // Collect direct + shifted matches, merge, greedy-select.
        let selected = greedy_modified_linear_matches(
            &left_peaks.mz,
            &right_peaks.mz,
            tolerance,
            shift,
            |i, j| left_peaks.int[i] * right_peaks.int[j],
        );

        let (raw_score, n_matches) =
            entropy_score_pairs(&selected, &left_peaks.int, &right_peaks.int)?;

        finalize_entropy_score(raw_score, n_matches)
    }
}
