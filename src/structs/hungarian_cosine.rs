//! Implementation of the cosine distance for mass spectra.

use geometric_traits::prelude::{Finite, Number, ScalarSimilarity, TotalOrd};
use multi_ranged::SimpleRange;
use num_traits::{Float, Pow, ToPrimitive};

use super::cosine_common::{
    compute_cosine_similarity, validate_non_negative_tolerance, validate_numeric_parameter,
};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Implementation of the cosine distance for mass spectra.
pub struct HungarianCosine<EXP, MZ> {
    /// The power to which the mass/charge ratio is raised.
    mz_power: EXP,
    /// The power to which the intensity is raised.
    intensity_power: EXP,
    /// The tolerance for the mass-shift of the mass/charge ratio.
    mz_tolerance: MZ,
}

impl<EXP: Number, MZ: Number> HungarianCosine<EXP, MZ> {
    /// Creates a new instance of the Hungarian cosine distance without
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

impl<EXP, MZ> HungarianCosine<EXP, MZ>
where
    EXP: Number + ToPrimitive,
    MZ: Number + ToPrimitive + PartialOrd,
{
    /// Creates a new instance of the Hungarian cosine distance.
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
            self.mz_power,
            self.intensity_power,
            |row, col| row.matching_peaks(col, self.mz_tolerance),
            |row, col| row.matching_peaks(col, self.mz_tolerance),
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
