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

use geometric_traits::prelude::ScalarSimilarity;

use super::cosine_common::{
    CosineConfig, ensure_finite, finalize_similarity_score, impl_cosine_wrapper_config_api,
    optimal_modified_linear_matches, prepare_peak_products, validate_well_separated,
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
pub struct ModifiedLinearCosine {
    config: CosineConfig,
}

impl_cosine_wrapper_config_api!(
    ModifiedLinearCosine,
    "the modified linear cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl<S1, S2> ScalarSimilarity<S1, S2> for ModifiedLinearCosine
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
        let left_mz: Vec<f64> = left.mz().collect();
        let right_mz: Vec<f64> = right.mz().collect();

        validate_well_separated(&left_mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_mz, tolerance, "right spectrum")?;

        let left_prec = ensure_finite(left.precursor_mz(), "left_precursor_mz")?;
        let right_prec = ensure_finite(right.precursor_mz(), "right_precursor_mz")?;

        let max_left = left_peaks.max;
        let max_right = right_peaks.max;
        let selected = optimal_modified_linear_matches(
            &left_mz,
            &right_mz,
            tolerance,
            left_prec,
            right_prec,
            |i, j| {
                if (left_peaks.products[i] * right_peaks.products[j]) == 0.0 {
                    return 0.0;
                }
                let normalized =
                    (left_peaks.products[i] / max_left) * (right_peaks.products[j] / max_right);
                normalized.max(f64::EPSILON)
            },
        );

        let mut score_sum = 0.0_f64;
        let mut n_matches = 0usize;
        for (i, j) in selected {
            let product = left_peaks.products[i] * right_peaks.products[j];
            if product != 0.0 {
                score_sum += product;
                n_matches += 1;
            }
        }

        finalize_similarity_score(score_sum, n_matches, left_peaks.norm, right_peaks.norm)
    }
}

impl<S1, S2> ScalarSpectralSimilarity<S1, S2> for ModifiedLinearCosine
where
    S1: Spectrum,
    S2: Spectrum,
{
}
