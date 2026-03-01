//! Implementation of the modified cosine similarity for mass spectra.
//!
//! Modified Cosine extends Exact Cosine by also matching fragment peaks
//! shifted by the precursor mass difference. This captures neutral-loss-related
//! peak correspondences between spectra with different precursor masses.
//!
//! Unlike matchms `ModifiedCosine` (greedy assignment), this uses Crouse
//! rectangular LAPJV for optimal assignment.

use geometric_traits::prelude::{
    Crouse, Finite, GenericImplicitValuedMatrix2D, Number, RangedCSR2D, ScalarSimilarity,
    SparseMatrix, TotalOrd,
};
use multi_ranged::BiRange;
use num_traits::{Float, One, Pow, ToPrimitive, Zero};

use super::cosine_common::{
    accumulate_assignment_scores, prepare_peak_products, validate_non_negative_tolerance,
    validate_numeric_parameter,
};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Modified cosine similarity for mass spectra.
///
/// Extends [`super::ExactCosine`] by also matching fragment peaks shifted by
/// the precursor mass difference, using optimal (Crouse LAPJV) assignment.
pub struct ModifiedCosine<EXP, MZ> {
    /// The power to which the mass/charge ratio is raised.
    mz_power: EXP,
    /// The power to which the intensity is raised.
    intensity_power: EXP,
    /// The tolerance for the mass-shift of the mass/charge ratio.
    mz_tolerance: MZ,
}

impl<EXP: Number, MZ: Number> ModifiedCosine<EXP, MZ> {
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

impl<EXP, MZ> ModifiedCosine<EXP, MZ>
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

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ModifiedCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks = prepare_peak_products(left, self.mz_power, self.intensity_power)?;
        let right_peaks = prepare_peak_products(right, self.mz_power, self.intensity_power)?;

        // Compute shift before the swap: shift = left.precursor_mz() - right.precursor_mz()
        let shift = left.precursor_mz() - right.precursor_mz();

        if left_peaks.max_f64 == 0.0 || right_peaks.max_f64 == 0.0 {
            return Ok((S1::Mz::zero(), 0));
        }

        // Sort spectra so the smaller one is on the row side of the bipartite
        // graph. When swapping, negate the shift.
        let (matching, row_f64, col_f64, row_products, col_products, max_row, max_col) =
            if left.len() <= right.len() {
                let matching = left.modified_matching_peaks(right, self.mz_tolerance, shift);
                (
                    matching,
                    &left_peaks.as_f64,
                    &right_peaks.as_f64,
                    &left_peaks.products,
                    &right_peaks.products,
                    left_peaks.max_f64,
                    right_peaks.max_f64,
                )
            } else {
                // Negate shift when swapping row/column roles.
                let negated_shift = S1::Mz::zero() - shift;
                let matching =
                    right.modified_matching_peaks(left, self.mz_tolerance, negated_shift);
                (
                    matching,
                    &right_peaks.as_f64,
                    &left_peaks.as_f64,
                    &right_peaks.products,
                    &left_peaks.products,
                    right_peaks.max_f64,
                    left_peaks.max_f64,
                )
            };

        let map: GenericImplicitValuedMatrix2D<RangedCSR2D<u32, u32, BiRange<u32>>, _, f64> =
            GenericImplicitValuedMatrix2D::new(matching, |(i, j)| {
                1.0f64 + f64::EPSILON
                    - (row_f64[i as usize] / max_row) * (col_f64[j as usize] / max_col)
            });

        if map.is_empty() {
            return Ok((S1::Mz::zero(), 0));
        }

        let non_edge_cost: f64 = 1.0f64 + f64::EPSILON;
        let max_cost: f64 = non_edge_cost + 1.0;

        let assignments: Vec<(u32, u32)> = map
            .crouse(non_edge_cost, max_cost)
            .map_err(|_| SimilarityComputationError::AssignmentFailed)?;

        let (score_sum, n_matches) =
            accumulate_assignment_scores(&assignments, row_products, col_products);

        let similarity = score_sum / (left_peaks.norm * right_peaks.norm);

        if similarity > S1::Mz::one() {
            Ok((S1::Mz::one(), n_matches))
        } else {
            Ok((similarity, n_matches))
        }
    }
}

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for ModifiedCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
