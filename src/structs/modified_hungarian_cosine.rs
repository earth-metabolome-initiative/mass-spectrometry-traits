//! Implementation of the modified Hungarian cosine similarity for mass spectra.
//!
//! Modified Hungarian Cosine extends Hungarian Cosine by also matching fragment peaks
//! shifted by the precursor mass difference. This captures neutral-loss-related
//! peak correspondences between spectra with different precursor masses.
//!
//! Unlike matchms `ModifiedCosine` (greedy assignment), this uses Crouse
//! rectangular LAPJV for optimal assignment.

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use multi_ranged::BiRange;
use num_traits::{Float, Pow, ToPrimitive};

use super::cosine_common::{
    compute_cosine_similarity, impl_cosine_wrapper_config_api, modified_precursor_shift_pair,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Modified cosine similarity for mass spectra.
///
/// Extends [`super::HungarianCosine`] by also matching fragment peaks shifted by
/// the precursor mass difference, using optimal (Crouse LAPJV) assignment.
pub struct ModifiedHungarianCosine<EXP, MZ> {
    config: super::cosine_common::CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    ModifiedHungarianCosine,
    "the modified Hungarian cosine similarity",
    "Returns the tolerance for the mass-shift of the mass/charge ratio."
);

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ModifiedHungarianCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let (shift, negated_shift) =
            modified_precursor_shift_pair(left.precursor_mz(), right.precursor_mz());

        compute_cosine_similarity::<_, _, _, BiRange<u32>, _, _>(
            left,
            right,
            self.config.mz_power(),
            self.config.intensity_power(),
            |row, col| row.modified_matching_peaks(col, self.config.mz_tolerance(), shift),
            |row, col| row.modified_matching_peaks(col, self.config.mz_tolerance(), negated_shift),
        )
    }
}

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for ModifiedHungarianCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
