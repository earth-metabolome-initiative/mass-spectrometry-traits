//! Implementation of the modified greedy cosine similarity for mass spectra.
//!
//! Modified Greedy Cosine extends Greedy Cosine by also matching fragment peaks
//! shifted by the precursor mass difference. This matches the matchms
//! `ModifiedCosine` algorithm.

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use multi_ranged::BiRange;
use num_traits::{Float, Pow, ToPrimitive, Zero};

use super::cosine_common::{CosineConfig, compute_cosine_similarity_greedy};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Modified greedy cosine similarity for mass spectra.
///
/// Extends [`super::GreedyCosine`] by also matching fragment peaks shifted by
/// the precursor mass difference, using greedy assignment. This matches the
/// matchms `ModifiedCosine` algorithm.
pub struct ModifiedGreedyCosine<EXP, MZ> {
    config: CosineConfig<EXP, MZ>,
}

impl<EXP: Number, MZ: Number> ModifiedGreedyCosine<EXP, MZ> {
    /// Creates a new instance without validating numeric parameters.
    #[inline]
    pub fn new_unchecked(mz_power: EXP, intensity_power: EXP, mz_tolerance: MZ) -> Self {
        Self {
            config: CosineConfig::new_unchecked(mz_power, intensity_power, mz_tolerance),
        }
    }

    /// Returns the tolerance for the mass/charge ratio.
    #[inline]
    pub fn mz_tolerance(&self) -> MZ {
        self.config.mz_tolerance()
    }

    /// Returns the power to which the mass/charge ratio is raised.
    #[inline]
    pub fn mz_power(&self) -> EXP {
        self.config.mz_power()
    }

    /// Returns the power to which the intensity is raised.
    #[inline]
    pub fn intensity_power(&self) -> EXP {
        self.config.intensity_power()
    }
}

impl<EXP, MZ> ModifiedGreedyCosine<EXP, MZ>
where
    EXP: Number + ToPrimitive,
    MZ: Number + ToPrimitive + PartialOrd,
{
    /// Creates a new instance of the modified greedy cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `mz_power`: The power to which the mass/charge ratio is raised.
    /// * `intensity_power`: The power to which the intensity is raised.
    /// * `mz_tolerance`: The tolerance for the mass/charge ratio.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if any numeric parameter is not
    /// finite/representable or if `mz_tolerance` is negative.
    #[inline]
    pub fn new(
        mz_power: EXP,
        intensity_power: EXP,
        mz_tolerance: MZ,
    ) -> Result<Self, SimilarityConfigError> {
        Ok(Self {
            config: CosineConfig::new(mz_power, intensity_power, mz_tolerance)?,
        })
    }
}

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
