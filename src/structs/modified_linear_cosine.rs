//! Implementation of modified linear-time cosine similarity for mass spectra.
//!
//! Extends [`super::LinearCosine`] by also matching fragment peaks shifted by
//! the precursor mass difference when that shift exceeds the configured
//! tolerance. Direct and shifted match candidates are merged and resolved via
//! optimal DP-based assignment on the conflict graph's path components,
//! producing the same score as [`super::ModifiedHungarianCosine`] in linear
//! time for well-separated spectra (match counts may differ on near-zero
//! edges due to f64 tie-breaking).

use alloc::vec::Vec;

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use num_traits::{Float, Pow, ToPrimitive, Zero};

use super::cosine_common::{
    CosineConfig, finalize_similarity_score, impl_cosine_wrapper_config_api,
    optimal_modified_linear_matches, prepare_peak_products, to_f64_checked_for_computation,
    validate_well_separated,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Modified linear-time cosine similarity for mass spectra.
///
/// Combines direct and precursor-shifted peak matches from two linear sweeps
/// when `|precursor_delta| > mz_tolerance`, then resolves conflicts via
/// optimal DP-based assignment on the conflict graph's path components.
/// For well-separated spectra this produces the same score as
/// [`super::ModifiedHungarianCosine`] in linear time (match counts may
/// differ on near-zero edges due to f64 tie-breaking).
/// Requires the same strict well-separated precondition as
/// [`super::LinearCosine`] (consecutive peaks > `2 * mz_tolerance`).
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

        let left_prec = to_f64_checked_for_computation(left.precursor_mz(), "left_precursor_mz")?;
        let right_prec =
            to_f64_checked_for_computation(right.precursor_mz(), "right_precursor_mz")?;

        // Collect direct + shifted matches, merge, optimal DP-select.
        // Benefit = normalised product (0 for zero-product edges so they
        // never displace real matches, at least ε for nonzero-product edges
        // so they are always selected when non-conflicting).
        let max_left = left_peaks.max_f64;
        let max_right = right_peaks.max_f64;
        let selected = optimal_modified_linear_matches(
            &left_mz,
            &right_mz,
            tolerance,
            left_prec,
            right_prec,
            |i, j| {
                if (left_peaks.products[i] * right_peaks.products[j]).is_zero() {
                    return 0.0;
                }
                let normalized =
                    (left_peaks.as_f64[i] / max_left) * (right_peaks.as_f64[j] / max_right);
                normalized.max(f64::EPSILON)
            },
        );

        let mut score_sum = S1::Mz::zero();
        let mut n_matches = 0usize;
        for (i, j) in selected {
            let product = left_peaks.products[i] * right_peaks.products[j];
            if !product.is_zero() {
                score_sum += product;
                n_matches += 1;
            }
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
