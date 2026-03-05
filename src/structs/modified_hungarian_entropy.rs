//! Implementation of modified Hungarian (optimal) spectral entropy similarity.
//!
//! Extends [`super::HungarianEntropy`] by also matching fragment peaks shifted
//! by the precursor mass difference, using Crouse rectangular LAPJV for optimal
//! assignment.

use geometric_traits::prelude::{Number, ScalarSimilarity};
use num_traits::{Float, ToPrimitive};

use super::cosine_common::to_f64_checked_for_computation;
use super::entropy_common::{
    compute_entropy_from_graph, entropy_score_from_matching, impl_entropy_config_api,
    impl_entropy_spectral_similarity,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Modified Hungarian (optimal) spectral entropy similarity.
///
/// Extends [`super::HungarianEntropy`] by also matching fragment peaks shifted
/// by the precursor mass difference when that shift exceeds the configured
/// tolerance, using optimal (Crouse LAPJV) assignment.
pub struct ModifiedHungarianEntropy<EXP, MZ> {
    mz_power: EXP,
    intensity_power: EXP,
    mz_tolerance: MZ,
    weighted: bool,
}

impl_entropy_config_api!(
    ModifiedHungarianEntropy,
    "modified Hungarian entropy similarity"
);
impl_entropy_spectral_similarity!(ModifiedHungarianEntropy);

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ModifiedHungarianEntropy<EXP, S1::Mz>
where
    EXP: Number + ToPrimitive,
    S1::Mz: Float + Number,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let mz_tolerance = self.mz_tolerance;
        let mz_power_f64 = to_f64_checked_for_computation(self.mz_power, "mz_power")
            .map_err(|_| SimilarityComputationError::ValueNotRepresentable("mz_power"))?;
        let intensity_power_f64 =
            to_f64_checked_for_computation(self.intensity_power, "intensity_power").map_err(
                |_| SimilarityComputationError::ValueNotRepresentable("intensity_power"),
            )?;
        compute_entropy_from_graph(
            left,
            right,
            self.weighted,
            mz_power_f64,
            intensity_power_f64,
            |row, col| {
                let shift = Spectrum::precursor_mz(row) - Spectrum::precursor_mz(col);
                Spectrum::modified_matching_peaks(row, col, mz_tolerance, shift)
            },
            |row, col| {
                let shift = Spectrum::precursor_mz(row) - Spectrum::precursor_mz(col);
                Spectrum::modified_matching_peaks(row, col, mz_tolerance, shift)
            },
            entropy_score_from_matching,
        )
    }
}
