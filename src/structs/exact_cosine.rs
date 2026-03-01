//! Implementation of the cosine distance for mass spectra.

use geometric_traits::prelude::{
    Crouse, Finite, GenericImplicitValuedMatrix2D, Number, RangedCSR2D, ScalarSimilarity,
    SparseMatrix, TotalOrd,
};
use multi_ranged::SimpleRange;
use num_traits::{Float, One, Pow, ToPrimitive, Zero};

use super::cosine_common::{accumulate_assignment_scores, prepare_peak_products};
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Implementation of the cosine distance for mass spectra.
pub struct ExactCosine<EXP, MZ> {
    /// The power to which the mass/charge ratio is raised.
    mz_power: EXP,
    /// The power to which the intensity is raised.
    intensity_power: EXP,
    /// The tolerance for the mass-shift of the mass/charge ratio.
    mz_tolerance: MZ,
}

impl<EXP: Number, MZ: Number> ExactCosine<EXP, MZ> {
    /// Creates a new instance of the Hungarian cosine distance.
    ///
    /// # Arguments
    ///
    /// * `mz_power`: The power to which the mass/charge ratio is raised.
    /// * `intensity_power`: The power to which the intensity is raised.
    /// * `mz_tolerance`: The tolerance for the mass-shift of the mass/charge
    ///   ratio.
    ///
    /// # Returns
    ///
    /// A new instance of the Hungarian cosine distance.
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

impl<EXP, S1, S2> ScalarSimilarity<S1, S2> for ExactCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = (S1::Mz, usize);

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks = prepare_peak_products(left, self.mz_power, self.intensity_power);
        let right_peaks = prepare_peak_products(right, self.mz_power, self.intensity_power);

        if left_peaks.max_f64 == 0.0 || right_peaks.max_f64 == 0.0 {
            return (S1::Mz::zero(), 0);
        }

        // Sort spectra so the smaller one is on the row side of the bipartite
        // graph. Crouse requires nr ≤ nc; placing the smaller spectrum on
        // rows avoids an internal transpose in the solver.
        let (matching, row_f64, col_f64, row_products, col_products, max_row, max_col) =
            if left.len() <= right.len() {
                let matching = left.matching_peaks(right, self.mz_tolerance);
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
                let matching = right.matching_peaks(left, self.mz_tolerance);
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

        let map: GenericImplicitValuedMatrix2D<RangedCSR2D<u32, u32, SimpleRange<u32>>, _, f64> =
            GenericImplicitValuedMatrix2D::new(matching, |(i, j)| {
                1.0f64 + f64::EPSILON
                    - (row_f64[i as usize] / max_row) * (col_f64[j as usize] / max_col)
            });

        if map.is_empty() {
            return (S1::Mz::zero(), 0);
        }

        // Use Crouse rectangular LAPJV: compactifies the sparse matrix, builds
        // a dense rectangular matrix with non_edge_cost for missing entries,
        // then solves with Crouse 2016 augmentation-only LAPJV.  Unlike the
        // Jaqaman expansion (which charges η > 2×max per unmatched peak and
        // forces suboptimal matches), this charges only 1+ε per unmatched
        // entry, correctly leaving peaks unmatched when that yields a better
        // overall assignment.
        let non_edge_cost: f64 = 1.0f64 + f64::EPSILON;
        let max_cost: f64 = non_edge_cost + 1.0;

        let assignments: Vec<(u32, u32)> = map
            .crouse(non_edge_cost, max_cost)
            .expect("Crouse rectangular LAPJV failed");

        // All returned assignments are real within-tolerance edges (non-edge
        // assignments are already filtered by Crouse).
        let (score_sum, n_matches) =
            accumulate_assignment_scores(&assignments, row_products, col_products);

        let similarity = score_sum / (left_peaks.norm * right_peaks.norm);

        if similarity > S1::Mz::one() {
            (S1::Mz::one(), n_matches)
        } else {
            (similarity, n_matches)
        }
    }
}

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for ExactCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
