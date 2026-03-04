//! Implementation of linear-time spectral entropy similarity.
//!
//! Equivalent to [`super::HungarianEntropy`] when spectra are *well-separated*
//! (consecutive peaks > `2 * mz_tolerance`). Under this invariant the
//! two-pointer sweep is provably optimal.

use geometric_traits::prelude::{Number, ScalarSimilarity};
use num_traits::{Float, Zero};

use super::cosine_common::{
    collect_linear_matches, to_f64_checked_for_computation, validate_well_separated,
};
use super::entropy_common::{
    entropy_score_pairs, finalize_entropy_score, impl_entropy_config_api,
    impl_entropy_spectral_similarity, prepare_entropy_peaks,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Linear-time spectral entropy similarity.
///
/// Equivalent to [`super::HungarianEntropy`] when spectra are *well-separated*:
/// consecutive peaks within each spectrum must be greater than
/// `2 * mz_tolerance`. Under this invariant the two-pointer sweep is provably
/// optimal.
///
/// Returns an error when the strict peak-spacing precondition is violated.
pub struct LinearEntropy<MZ> {
    mz_tolerance: MZ,
    weighted: bool,
}

impl_entropy_config_api!(LinearEntropy, "linear entropy similarity");
impl_entropy_spectral_similarity!(LinearEntropy);

impl<S1, S2> ScalarSimilarity<S1, S2> for LinearEntropy<S1::Mz>
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

        let pairs = collect_linear_matches(&left_peaks.mz, &right_peaks.mz, tolerance, 0.0);
        let (raw_score, n_matches) =
            entropy_score_pairs(&pairs, &left_peaks.int, &right_peaks.int)?;

        finalize_entropy_score(raw_score, n_matches)
    }
}
