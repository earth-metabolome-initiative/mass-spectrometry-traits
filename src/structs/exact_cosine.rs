//! Implementation of the cosine distance for mass spectra.

use geometric_traits::prelude::{
    Crouse, Finite, GenericImplicitValuedMatrix2D, Number, RangedCSR2D, ScalarSimilarity,
    SparseMatrix, TotalOrd,
};
use multi_ranged::SimpleRange;
use num_traits::{Float, One, Pow, ToPrimitive, Zero};

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

        // Promote peak products to f64 for the LAP computation.
        // With f32, the normalized affine cost 1+ε−(wᵢ/max_l)·(wⱼ/max_r)
        // has only ~7 significant digits. When the weight dynamic range is
        // large (e.g. mz_pow=0 where weights = intensities spanning 1e3–1e9),
        // many cost entries collapse to the same f32 value, causing the LAP
        // solver to return a suboptimal assignment.
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
        // graph. Crouse requires nr ≤ nc; placing the smaller spectrum on
        // rows avoids an internal transpose in the solver.
        let (matching, row_f64, col_f64, row_products, col_products, max_row, max_col) =
            if left.len() <= right.len() {
                let matching = left.matching_peaks(right, self.mz_tolerance);
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
                let matching = right.matching_peaks(left, self.mz_tolerance);
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

        let map: GenericImplicitValuedMatrix2D<RangedCSR2D<u32, u16, SimpleRange<u16>>, _, f64> =
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

        let assignments: Vec<(u16, u16)> = map
            .crouse(non_edge_cost, max_cost)
            .expect("Crouse rectangular LAPJV failed");

        // All returned assignments are real within-tolerance edges (non-edge
        // assignments are already filtered by Crouse).
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

impl<S1, S2, EXP> ScalarSpectralSimilarity<S1, S2> for ExactCosine<EXP, S1::Mz>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Finite + TotalOrd,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
