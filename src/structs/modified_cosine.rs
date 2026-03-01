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
    /// Creates a new instance of the modified cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `mz_power`: The power to which the mass/charge ratio is raised.
    /// * `intensity_power`: The power to which the intensity is raised.
    /// * `mz_tolerance`: The tolerance for the mass-shift of the mass/charge
    ///   ratio.
    pub fn new(mz_power: EXP, intensity_power: EXP, mz_tolerance: MZ) -> Self {
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

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ModifiedCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = (S1::Mz, u16);

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let mut left_peak_products = Vec::with_capacity(left.len());
        let mut left_peak_squared_sums: S1::Mz = S1::Mz::zero();
        let mut right_peak_products = Vec::with_capacity(right.len());
        let mut right_peak_squared_sums: S1::Mz = S1::Mz::zero();

        for (mz, intensity) in left.peaks() {
            let score = mz.pow(self.mz_power) * intensity.pow(self.intensity_power);
            left_peak_products.push(score);
            left_peak_squared_sums += score * score;
        }
        for (mz, intensity) in right.peaks() {
            let score = mz.pow(self.mz_power) * intensity.pow(self.intensity_power);
            right_peak_products.push(score);
            right_peak_squared_sums += score * score;
        }

        let left_peak_norm: S1::Mz = left_peak_squared_sums.sqrt();
        let right_peak_norm: S1::Mz = right_peak_squared_sums.sqrt();

        // Compute shift before the swap: shift = left.precursor_mz() - right.precursor_mz()
        let shift = left.precursor_mz() - right.precursor_mz();

        // Promote peak products to f64 for the LAP computation.
        let left_f64: Vec<f64> = left_peak_products
            .iter()
            .map(|p| p.to_f64().unwrap())
            .collect();
        let right_f64: Vec<f64> = right_peak_products
            .iter()
            .map(|p| p.to_f64().unwrap())
            .collect();

        let max_left: f64 = left_f64.iter().cloned().fold(0.0f64, f64::max);
        let max_right: f64 = right_f64.iter().cloned().fold(0.0f64, f64::max);

        if max_left == 0.0 || max_right == 0.0 {
            return (S1::Mz::zero(), 0);
        }

        // Sort spectra so the smaller one is on the row side of the bipartite
        // graph. When swapping, negate the shift.
        let (matching, row_f64, col_f64, row_products, col_products, max_row, max_col) =
            if left.len() <= right.len() {
                let matching = left.modified_matching_peaks(right, self.mz_tolerance, shift);
                (
                    matching,
                    &left_f64,
                    &right_f64,
                    &left_peak_products,
                    &right_peak_products,
                    max_left,
                    max_right,
                )
            } else {
                // Negate shift when swapping row/column roles.
                let negated_shift = S1::Mz::zero() - shift;
                let matching =
                    right.modified_matching_peaks(left, self.mz_tolerance, negated_shift);
                (
                    matching,
                    &right_f64,
                    &left_f64,
                    &right_peak_products,
                    &left_peak_products,
                    max_right,
                    max_left,
                )
            };

        let map: GenericImplicitValuedMatrix2D<RangedCSR2D<u32, u16, BiRange<u16>>, _, f64> =
            GenericImplicitValuedMatrix2D::new(matching, |(i, j)| {
                1.0f64 + f64::EPSILON
                    - (row_f64[i as usize] / max_row) * (col_f64[j as usize] / max_col)
            });

        if map.is_empty() {
            return (S1::Mz::zero(), 0);
        }

        let non_edge_cost: f64 = 1.0f64 + f64::EPSILON;
        let max_cost: f64 = non_edge_cost + 1.0;

        let assignments: Vec<(u16, u16)> = map
            .crouse(non_edge_cost, max_cost)
            .expect("Crouse rectangular LAPJV failed");

        let mut score_sum = S1::Mz::zero();
        let mut n_matches: u16 = 0;
        for &(i, j) in &assignments {
            score_sum += row_products[i as usize] * col_products[j as usize];
            n_matches += 1;
        }

        let similarity = score_sum / (left_peak_norm * right_peak_norm);

        if similarity > S1::Mz::one() {
            (S1::Mz::one(), n_matches)
        } else {
            (similarity, n_matches)
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
