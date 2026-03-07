//! Implementation of linear-time spectral entropy similarity.
//!
//! When spectra are *well-separated* (consecutive peaks > `2 * mz_tolerance`),
//! the two-pointer sweep is provably optimal.

use geometric_traits::prelude::{Number, ScalarSimilarity};
use num_traits::{Float, NumCast, ToPrimitive, Zero};

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
/// Requires spectra to be *well-separated*: consecutive peaks within each
/// spectrum must be greater than `2 * mz_tolerance`. Under this invariant the
/// two-pointer sweep is provably optimal.
///
/// Returns an error when the strict peak-spacing precondition is violated.
pub struct LinearEntropy<EXP, MZ> {
    mz_power: EXP,
    intensity_power: EXP,
    mz_tolerance: MZ,
    weighted: bool,
}

impl_entropy_config_api!(LinearEntropy, "linear entropy similarity");
impl_entropy_spectral_similarity!(LinearEntropy);

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for LinearEntropy<EXP, S1::Mz>
where
    EXP: Number + ToPrimitive,
    S1::Mz: Float + Number,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let mz_power_f64 = to_f64_checked_for_computation(self.mz_power, "mz_power")
            .map_err(|_| SimilarityComputationError::ValueNotRepresentable("mz_power"))?;
        let intensity_power_f64 =
            to_f64_checked_for_computation(self.intensity_power, "intensity_power").map_err(
                |_| SimilarityComputationError::ValueNotRepresentable("intensity_power"),
            )?;

        let left_peaks: super::entropy_common::PreparedEntropyPeaks<f64> =
            prepare_entropy_peaks(left, self.weighted, mz_power_f64, intensity_power_f64)?;
        let right_peaks: super::entropy_common::PreparedEntropyPeaks<f64> =
            prepare_entropy_peaks(right, self.weighted, mz_power_f64, intensity_power_f64)?;

        if left_peaks.int.is_empty() || right_peaks.int.is_empty() {
            return Ok((S1::Mz::zero(), 0));
        }

        let tolerance = to_f64_checked_for_computation(self.mz_tolerance, "mz_tolerance")?;

        validate_well_separated(&left_peaks.mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_peaks.mz, tolerance, "right spectrum")?;

        let pairs = collect_linear_matches(&left_peaks.mz, &right_peaks.mz, tolerance, 0.0);
        let (raw_score, n_matches) =
            entropy_score_pairs(&pairs, &left_peaks.int, &right_peaks.int)?;

        let (sim_f64, n) = finalize_entropy_score(raw_score, n_matches)?;
        let sim: S1::Mz = NumCast::from(sim_f64).ok_or(
            SimilarityComputationError::ValueNotRepresentable("similarity_score"),
        )?;
        Ok((sim, n))
    }
}
