//! Implementation of linear-time cosine similarity for mass spectra.
//!
//! When consecutive peaks within each spectrum are more than 2x the matching
//! tolerance apart, each peak can match at most one peak in the other spectrum.
//! This eliminates assignment ambiguity, allowing a simple two-pointer sweep
//! to produce the same result as Hungarian (optimal) assignment in O(n+m) time.

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use num_traits::{Float, Pow, ToPrimitive, Zero};

use super::cosine_common::{
    CosineConfig, finalize_similarity_score, impl_cosine_wrapper_config_api, linear_cosine_sweep,
    prepare_peak_products, to_f64_checked_for_computation, validate_well_separated,
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
pub struct LinearCosine<EXP, MZ> {
    config: CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    LinearCosine,
    "the linear cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for LinearCosine<EXP, S1::Mz>
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
        let left_mz: alloc::vec::Vec<f64> = left
            .peaks()
            .map(|(mz, _)| to_f64_checked_for_computation(mz, "left_mz"))
            .collect::<Result<alloc::vec::Vec<f64>, SimilarityComputationError>>()?;
        let right_mz: alloc::vec::Vec<f64> = right
            .peaks()
            .map(|(mz, _)| to_f64_checked_for_computation(mz, "right_mz"))
            .collect::<Result<alloc::vec::Vec<f64>, SimilarityComputationError>>()?;

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

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for LinearCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
