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
use num_traits::{Float, Pow, ToPrimitive, Zero};

use super::cosine_common::{
    compute_cosine_similarity, validate_non_negative_tolerance, validate_numeric_parameter,
};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Modified cosine similarity for mass spectra.
///
/// Extends [`super::HungarianCosine`] by also matching fragment peaks shifted by
/// the precursor mass difference, using optimal (Crouse LAPJV) assignment.
pub struct ModifiedHungarianCosine<EXP, MZ> {
    /// The power to which the mass/charge ratio is raised.
    mz_power: EXP,
    /// The power to which the intensity is raised.
    intensity_power: EXP,
    /// The tolerance for the mass-shift of the mass/charge ratio.
    mz_tolerance: MZ,
}

impl<EXP: Number, MZ: Number> ModifiedHungarianCosine<EXP, MZ> {
    /// Creates a new instance of the modified cosine similarity without
    /// validating numeric parameters.
    pub fn new_unchecked(mz_power: EXP, intensity_power: EXP, mz_tolerance: MZ) -> Self {
        Self {
            mz_power,
            intensity_power,
            mz_tolerance,
        }
    }

    /// Returns the tolerance for the mass-shift of the mass/charge ratio.
    pub fn mz_tolerance(&self) -> MZ {
        self.mz_tolerance
    }

    /// Returns the power to which the mass/charge ratio is raised.
    pub fn mz_power(&self) -> EXP {
        self.mz_power
    }

    /// Returns the power to which the intensity is raised.
    pub fn intensity_power(&self) -> EXP {
        self.intensity_power
    }
}

impl<EXP, MZ> ModifiedHungarianCosine<EXP, MZ>
where
    EXP: Number + ToPrimitive,
    MZ: Number + ToPrimitive + PartialOrd,
{
    /// Creates a new instance of the modified cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `mz_power`: The power to which the mass/charge ratio is raised.
    /// * `intensity_power`: The power to which the intensity is raised.
    /// * `mz_tolerance`: The tolerance for the mass-shift of the mass/charge
    ///   ratio.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if any numeric parameter is not
    /// finite/representable or if `mz_tolerance` is negative.
    pub fn new(
        mz_power: EXP,
        intensity_power: EXP,
        mz_tolerance: MZ,
    ) -> Result<Self, SimilarityConfigError> {
        validate_numeric_parameter(mz_power, "mz_power")?;
        validate_numeric_parameter(intensity_power, "intensity_power")?;
        validate_non_negative_tolerance(mz_tolerance)?;
        Ok(Self::new_unchecked(mz_power, intensity_power, mz_tolerance))
    }
}

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ModifiedHungarianCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        // Compute shift before the swap: shift = left.precursor_mz() - right.precursor_mz()
        let shift = left.precursor_mz() - right.precursor_mz();
        let negated_shift = S1::Mz::zero() - shift;

        compute_cosine_similarity::<_, _, _, BiRange<u32>, _, _>(
            left,
            right,
            self.mz_power,
            self.intensity_power,
            |row, col| row.modified_matching_peaks(col, self.mz_tolerance, shift),
            |row, col| row.modified_matching_peaks(col, self.mz_tolerance, negated_shift),
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
