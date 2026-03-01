use geometric_traits::prelude::{Finite, Number, TotalOrd};
use num_traits::{Float, Pow, ToPrimitive, Zero};

use crate::structs::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
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
) -> Result<PreparedPeaks<S::Mz>, SimilarityComputationError>
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
    let mut as_f64 = Vec::with_capacity(products.len());
    for p in &products {
        let Some(v) = p.to_f64() else {
            return Err(SimilarityComputationError::ValueNotRepresentable(
                "peak_product",
            ));
        };
        as_f64.push(v);
    }
    let max_f64 = as_f64.iter().copied().fold(0.0_f64, f64::max);

    Ok(PreparedPeaks {
        products,
        norm,
        as_f64,
        max_f64,
    })
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

pub(crate) fn validate_numeric_parameter<T: ToPrimitive>(
    value: T,
    name: &'static str,
) -> Result<(), SimilarityConfigError> {
    let Some(v) = value.to_f64() else {
        return Err(SimilarityConfigError::NonRepresentableParameter(name));
    };
    if !v.is_finite() {
        return Err(SimilarityConfigError::NonFiniteParameter(name));
    }
    Ok(())
}

pub(crate) fn validate_non_negative_tolerance<T>(
    mz_tolerance: T,
) -> Result<(), SimilarityConfigError>
where
    T: Number + ToPrimitive + PartialOrd,
{
    validate_numeric_parameter(mz_tolerance, "mz_tolerance")?;
    if mz_tolerance < T::zero() {
        return Err(SimilarityConfigError::NegativeTolerance);
    }
    Ok(())
}
