//! Implementation of the Hungarian cosine similarity for mass spectra.

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use multi_ranged::SimpleRange;
use num_traits::{Float, Pow, ToPrimitive};

use super::cosine_common::{compute_cosine_similarity, impl_cosine_wrapper_config_api};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Implementation of the Hungarian cosine similarity for mass spectra.
pub struct HungarianCosine<EXP, MZ> {
    config: super::cosine_common::CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    HungarianCosine,
    "the Hungarian cosine similarity",
    "Returns the tolerance for the mass-shift of the mass/charge ratio."
);

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for HungarianCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        compute_cosine_similarity::<_, _, _, SimpleRange<u32>, _, _>(
            left,
            right,
            self.config.mz_power(),
            self.config.intensity_power(),
            |row, col| row.matching_peaks(col, self.config.mz_tolerance()),
            |row, col| row.matching_peaks(col, self.config.mz_tolerance()),
        )
    }
}

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for HungarianCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
