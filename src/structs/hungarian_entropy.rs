//! Implementation of Hungarian (optimal) spectral entropy similarity.
//!
//! Uses Crouse rectangular LAPJV for optimal peak assignment, scoring matched
//! pairs with Jensen-Shannon divergence.

use geometric_traits::prelude::{Number, ScalarSimilarity};
use num_traits::Float;

use super::entropy_common::{
    compute_entropy_from_graph, entropy_score_from_matching, impl_entropy_config_api,
    impl_entropy_spectral_similarity,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Hungarian (optimal) spectral entropy similarity.
///
/// Optimal spectral entropy similarity using Crouse rectangular LAPJV
/// for peak assignment.
pub struct HungarianEntropy<MZ> {
    mz_tolerance: MZ,
    weighted: bool,
}

impl_entropy_config_api!(HungarianEntropy, "Hungarian entropy similarity");
impl_entropy_spectral_similarity!(HungarianEntropy);

impl<S1, S2> ScalarSimilarity<S1, S2> for HungarianEntropy<S1::Mz>
where
    S1::Mz: Float + Number,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let mz_tolerance = self.mz_tolerance;
        compute_entropy_from_graph(
            left,
            right,
            self.weighted,
            |row, col| Spectrum::matching_peaks(row, col, mz_tolerance),
            |row, col| Spectrum::matching_peaks(row, col, mz_tolerance),
            entropy_score_from_matching,
        )
    }
}
