//! Implementation of the modified greedy cosine similarity for mass spectra.
//!
//! Modified Greedy Cosine extends Greedy Cosine by also matching fragment peaks
//! shifted by the precursor mass difference. This matches the matchms
//! `ModifiedCosine` algorithm.

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use multi_ranged::BiRange;
use num_traits::{Float, Pow, ToPrimitive, Zero};

use super::cosine_common::{compute_cosine_similarity_greedy, impl_cosine_wrapper_config_api};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Modified greedy cosine similarity for mass spectra.
///
/// Extends [`super::GreedyCosine`] by also matching fragment peaks shifted by
/// the precursor mass difference, using greedy assignment. This matches the
/// matchms `ModifiedCosine` algorithm.
pub struct ModifiedGreedyCosine<EXP, MZ> {
    config: super::cosine_common::CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    ModifiedGreedyCosine,
    "the modified greedy cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ModifiedGreedyCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let shift = left.precursor_mz() - right.precursor_mz();
        let negated_shift = S1::Mz::zero() - shift;

        compute_cosine_similarity_greedy::<_, _, _, BiRange<u32>, _, _>(
            left,
            right,
            self.config.mz_power(),
            self.config.intensity_power(),
            |row, col| row.modified_matching_peaks(col, self.config.mz_tolerance(), shift),
            |row, col| row.modified_matching_peaks(col, self.config.mz_tolerance(), negated_shift),
        )
    }
}

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for ModifiedGreedyCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
