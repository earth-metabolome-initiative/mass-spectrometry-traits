//! Implementation of modified linear-time cosine similarity for mass spectra.
//!
//! Extends [`super::LinearCosine`] by also matching fragment peaks shifted by
//! the precursor mass difference when that shift exceeds the configured
//! tolerance. Direct and shifted match candidates are merged and resolved via
//! greedy assignment (sort by descending product weight, pick best available),
//! identical to [`super::ModifiedGreedyCosine`] but on the smaller candidate
//! set from two linear sweeps. Unlike [`super::ModifiedHungarianCosine`], this
//! is not an optimal assignment solver.

use alloc::vec::Vec;

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use num_traits::{Float, Pow, ToPrimitive, Zero};

use super::cosine_common::{
    CosineConfig, finalize_similarity_score, greedy_modified_linear_matches,
    impl_cosine_wrapper_config_api, prepare_peak_products, to_f64_checked_for_computation,
    validate_well_separated,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Modified linear-time cosine similarity for mass spectra.
///
/// Combines direct and precursor-shifted peak matches from two linear sweeps
/// when `|precursor_delta| > mz_tolerance`, then resolves conflicts via greedy
/// assignment. This matches the greedy conflict-resolution behavior of
/// [`super::ModifiedGreedyCosine`] rather than the optimal assignment behavior
/// of [`super::ModifiedHungarianCosine`].
/// Requires the same
/// strict well-separated precondition as [`super::LinearCosine`]
/// (consecutive peaks > `2 * mz_tolerance`).
pub struct ModifiedLinearCosine<EXP, MZ> {
    config: CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    ModifiedLinearCosine,
    "the modified linear cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ModifiedLinearCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks =
            prepare_peak_products(left, self.config.mz_power(), self.config.intensity_power())?;
        let right_peaks =
            prepare_peak_products(right, self.config.mz_power(), self.config.intensity_power())?;

        if left_peaks.max_f64 == 0.0 || right_peaks.max_f64 == 0.0 {
            return Ok((S1::Mz::zero(), 0));
        }

        let tolerance = to_f64_checked_for_computation(self.config.mz_tolerance(), "mz_tolerance")?;

        // Collect mz as f64 with numeric validation.
        let left_mz: Vec<f64> = left
            .peaks()
            .map(|(mz, _)| to_f64_checked_for_computation(mz, "left_mz"))
            .collect::<Result<Vec<f64>, SimilarityComputationError>>()?;
        let right_mz: Vec<f64> = right
            .peaks()
            .map(|(mz, _)| to_f64_checked_for_computation(mz, "right_mz"))
            .collect::<Result<Vec<f64>, SimilarityComputationError>>()?;

        validate_well_separated(&left_mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_mz, tolerance, "right spectrum")?;

        // Compute the precursor mass shift.
        let left_precursor =
            to_f64_checked_for_computation(left.precursor_mz(), "left_precursor_mz")?;
        let right_precursor =
            to_f64_checked_for_computation(right.precursor_mz(), "right_precursor_mz")?;
        let shift = left_precursor - right_precursor;

        // Collect direct + shifted matches, merge, greedy-select.
        let selected =
            greedy_modified_linear_matches(&left_mz, &right_mz, tolerance, shift, |i, j| {
                left_peaks.as_f64[i] * right_peaks.as_f64[j]
            });

        let mut score_sum = S1::Mz::zero();
        let n_matches = selected.len();
        for (i, j) in selected {
            score_sum += left_peaks.products[i] * right_peaks.products[j];
        }

        finalize_similarity_score(score_sum, n_matches, left_peaks.norm, right_peaks.norm)
    }
}

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for ModifiedLinearCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
