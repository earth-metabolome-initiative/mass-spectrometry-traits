use geometric_traits::prelude::{Finite, Number, TotalOrd};
use num_traits::{Float, Pow, ToPrimitive, Zero};

use crate::traits::Spectrum;

pub(crate) struct PreparedPeaks<MZ> {
    pub(crate) products: Vec<MZ>,
    pub(crate) norm: MZ,
    pub(crate) as_f64: Vec<f64>,
    pub(crate) max_f64: f64,
}

pub(crate) fn prepare_peak_products<EXP, S>(
    spectrum: &S,
    mz_power: EXP,
    intensity_power: EXP,
) -> PreparedPeaks<S::Mz>
where
    EXP: Number,
    S: Spectrum<Intensity = <S as Spectrum>::Mz>,
    S::Mz: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
{
    let mut products = Vec::with_capacity(spectrum.len());
    let mut squared_sum = S::Mz::zero();

    for (mz, intensity) in spectrum.peaks() {
        let score = mz.pow(mz_power) * intensity.pow(intensity_power);
        products.push(score);
        squared_sum += score * score;
    }

    let norm = squared_sum.sqrt();
    let as_f64: Vec<f64> = products
        .iter()
        .map(|p| {
            p.to_f64()
                .expect("Peak product must be representable as f64")
        })
        .collect();
    let max_f64 = as_f64.iter().copied().fold(0.0_f64, f64::max);

    PreparedPeaks {
        products,
        norm,
        as_f64,
        max_f64,
    }
}

pub(crate) fn accumulate_assignment_scores<MZ>(
    assignments: &[(u32, u32)],
    row_products: &[MZ],
    col_products: &[MZ],
) -> (MZ, usize)
where
    MZ: Number + Zero,
{
    let mut score_sum = MZ::zero();
    let mut n_matches = 0usize;

    for &(i, j) in assignments {
        score_sum += row_products[i as usize] * col_products[j as usize];
        n_matches += 1;
    }

    (score_sum, n_matches)
}
