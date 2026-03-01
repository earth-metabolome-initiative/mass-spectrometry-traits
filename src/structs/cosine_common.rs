use alloc::vec::Vec;

use geometric_traits::prelude::{
    Crouse, Finite, GenericImplicitValuedMatrix2D, Number, RangedCSR2D, SparseMatrix, TotalOrd,
};
use multi_ranged::MultiRanged;
use num_traits::{Float, Pow, ToPrimitive, Zero};

use crate::structs::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::Spectrum;

pub(crate) struct PreparedPeaks<MZ> {
    pub(crate) products: Vec<MZ>,
    pub(crate) norm: MZ,
    pub(crate) as_f64: Vec<f64>,
    pub(crate) max_f64: f64,
}

pub(crate) struct MatchingScoreInputs<'a, MZ, R: MultiRanged<Step = u32>> {
    pub(crate) matching: RangedCSR2D<u32, u32, R>,
    pub(crate) row_f64: &'a [f64],
    pub(crate) col_f64: &'a [f64],
    pub(crate) row_products: &'a [MZ],
    pub(crate) col_products: &'a [MZ],
    pub(crate) max_row: f64,
    pub(crate) max_col: f64,
    pub(crate) left_norm: MZ,
    pub(crate) right_norm: MZ,
}

pub(crate) struct CosineConfig<EXP, MZ> {
    mz_power: EXP,
    intensity_power: EXP,
    mz_tolerance: MZ,
}

impl<EXP: Number, MZ: Number> CosineConfig<EXP, MZ> {
    #[inline]
    pub(crate) fn new_unchecked(mz_power: EXP, intensity_power: EXP, mz_tolerance: MZ) -> Self {
        Self {
            mz_power,
            intensity_power,
            mz_tolerance,
        }
    }

    #[inline]
    pub(crate) fn mz_tolerance(&self) -> MZ {
        self.mz_tolerance
    }

    #[inline]
    pub(crate) fn mz_power(&self) -> EXP {
        self.mz_power
    }

    #[inline]
    pub(crate) fn intensity_power(&self) -> EXP {
        self.intensity_power
    }
}

impl<EXP, MZ> CosineConfig<EXP, MZ>
where
    EXP: Number + ToPrimitive,
    MZ: Number + ToPrimitive + PartialOrd,
{
    #[inline]
    pub(crate) fn new(
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

#[inline]
pub(crate) fn ensure_finite_f64(
    value: f64,
    name: &'static str,
) -> Result<(), SimilarityComputationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SimilarityComputationError::NonFiniteValue(name))
    }
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
    let mut as_f64 = Vec::with_capacity(spectrum.len());
    let mut squared_sum = S::Mz::zero();

    for (mz, intensity) in spectrum.peaks() {
        let score = mz.pow(mz_power) * intensity.pow(intensity_power);
        let Some(score_f64) = score.to_f64() else {
            return Err(SimilarityComputationError::ValueNotRepresentable(
                "peak_product",
            ));
        };
        ensure_finite_f64(score_f64, "peak_product")?;

        products.push(score);
        as_f64.push(score_f64);
        squared_sum += score * score;
    }

    let norm = squared_sum.sqrt();
    let Some(norm_f64) = norm.to_f64() else {
        return Err(SimilarityComputationError::ValueNotRepresentable(
            "peak_norm",
        ));
    };
    ensure_finite_f64(norm_f64, "peak_norm")?;

    let max_f64 = as_f64.iter().copied().fold(0.0_f64, f64::max);
    ensure_finite_f64(max_f64, "peak_product_max")?;

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

pub(crate) fn score_from_matching<MZ, R>(
    inputs: MatchingScoreInputs<'_, MZ, R>,
) -> Result<(MZ, usize), SimilarityComputationError>
where
    MZ: Float + Number + Finite + TotalOrd + ToPrimitive,
    R: MultiRanged<Step = u32>,
{
    ensure_finite_f64(inputs.max_row, "row_peak_product_max")?;
    ensure_finite_f64(inputs.max_col, "col_peak_product_max")?;

    let map: GenericImplicitValuedMatrix2D<RangedCSR2D<u32, u32, R>, _, f64> =
        GenericImplicitValuedMatrix2D::new(inputs.matching, |(i, j)| {
            1.0f64 + f64::EPSILON
                - (inputs.row_f64[i as usize] / inputs.max_row)
                    * (inputs.col_f64[j as usize] / inputs.max_col)
        });

    if map.is_empty() {
        return Ok((MZ::zero(), 0));
    }

    let non_edge_cost: f64 = 1.0f64 + f64::EPSILON;
    let max_cost: f64 = non_edge_cost + 1.0;

    let assignments: Vec<(u32, u32)> = map
        .crouse(non_edge_cost, max_cost)
        .map_err(|_| SimilarityComputationError::AssignmentFailed)?;

    let (score_sum, n_matches) =
        accumulate_assignment_scores(&assignments, inputs.row_products, inputs.col_products);

    let denominator = inputs.left_norm * inputs.right_norm;
    let Some(denominator_f64) = denominator.to_f64() else {
        return Err(SimilarityComputationError::ValueNotRepresentable(
            "similarity_denominator",
        ));
    };
    ensure_finite_f64(denominator_f64, "similarity_denominator")?;
    if denominator_f64 == 0.0 {
        return Ok((MZ::zero(), 0));
    }

    let similarity = score_sum / denominator;
    let Some(similarity_f64) = similarity.to_f64() else {
        return Err(SimilarityComputationError::ValueNotRepresentable(
            "similarity_score",
        ));
    };
    ensure_finite_f64(similarity_f64, "similarity_score")?;

    if similarity > MZ::one() {
        Ok((MZ::one(), n_matches))
    } else {
        Ok((similarity, n_matches))
    }
}

pub(crate) fn score_from_matching_greedy<MZ, R>(
    inputs: MatchingScoreInputs<'_, MZ, R>,
) -> Result<(MZ, usize), SimilarityComputationError>
where
    MZ: Float + Number + Finite + TotalOrd + ToPrimitive,
    R: MultiRanged<Step = u32>,
{
    if inputs.matching.is_empty() {
        return Ok((MZ::zero(), 0));
    }

    // Collect all candidate edges with their weights (descending sort).
    let mut candidates: Vec<(f64, u32, u32)> = SparseMatrix::sparse_coordinates(&inputs.matching)
        .map(|(i, j)| {
            let weight = inputs.row_f64[i as usize] * inputs.col_f64[j as usize];
            (weight, i, j)
        })
        .collect();

    candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    // Greedy selection: pick the highest-weight edge whose row and column
    // have not yet been used.
    let n_rows = inputs.row_f64.len();
    let n_cols = inputs.col_f64.len();
    let mut used_rows = alloc::vec![false; n_rows];
    let mut used_cols = alloc::vec![false; n_cols];
    let mut assignments: Vec<(u32, u32)> = Vec::new();

    for &(_, i, j) in &candidates {
        if !used_rows[i as usize] && !used_cols[j as usize] {
            used_rows[i as usize] = true;
            used_cols[j as usize] = true;
            assignments.push((i, j));
        }
    }

    let (score_sum, n_matches) =
        accumulate_assignment_scores(&assignments, inputs.row_products, inputs.col_products);

    let denominator = inputs.left_norm * inputs.right_norm;
    let Some(denominator_f64) = denominator.to_f64() else {
        return Err(SimilarityComputationError::ValueNotRepresentable(
            "similarity_denominator",
        ));
    };
    ensure_finite_f64(denominator_f64, "similarity_denominator")?;
    if denominator_f64 == 0.0 {
        return Ok((MZ::zero(), 0));
    }

    let similarity = score_sum / denominator;
    let Some(similarity_f64) = similarity.to_f64() else {
        return Err(SimilarityComputationError::ValueNotRepresentable(
            "similarity_score",
        ));
    };
    ensure_finite_f64(similarity_f64, "similarity_score")?;

    if similarity > MZ::one() {
        Ok((MZ::one(), n_matches))
    } else {
        Ok((similarity, n_matches))
    }
}

pub(crate) fn compute_cosine_similarity<EXP, S1, S2, R, ForwardMatch, ReverseMatch>(
    left: &S1,
    right: &S2,
    mz_power: EXP,
    intensity_power: EXP,
    forward_match: ForwardMatch,
    reverse_match: ReverseMatch,
) -> Result<(S1::Mz, usize), SimilarityComputationError>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
    R: MultiRanged<Step = u32>,
    ForwardMatch: FnOnce(&S1, &S2) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
    ReverseMatch: FnOnce(&S2, &S1) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
{
    let left_peaks = prepare_peak_products(left, mz_power, intensity_power)?;
    let right_peaks = prepare_peak_products(right, mz_power, intensity_power)?;

    if left_peaks.max_f64 == 0.0 || right_peaks.max_f64 == 0.0 {
        return Ok((S1::Mz::zero(), 0));
    }

    if left.len() <= right.len() {
        let matching = forward_match(left, right)?;
        score_from_matching(MatchingScoreInputs {
            matching,
            row_f64: &left_peaks.as_f64,
            col_f64: &right_peaks.as_f64,
            row_products: &left_peaks.products,
            col_products: &right_peaks.products,
            max_row: left_peaks.max_f64,
            max_col: right_peaks.max_f64,
            left_norm: left_peaks.norm,
            right_norm: right_peaks.norm,
        })
    } else {
        let matching = reverse_match(right, left)?;
        score_from_matching(MatchingScoreInputs {
            matching,
            row_f64: &right_peaks.as_f64,
            col_f64: &left_peaks.as_f64,
            row_products: &right_peaks.products,
            col_products: &left_peaks.products,
            max_row: right_peaks.max_f64,
            max_col: left_peaks.max_f64,
            left_norm: left_peaks.norm,
            right_norm: right_peaks.norm,
        })
    }
}

pub(crate) fn compute_cosine_similarity_greedy<EXP, S1, S2, R, ForwardMatch, ReverseMatch>(
    left: &S1,
    right: &S2,
    mz_power: EXP,
    intensity_power: EXP,
    forward_match: ForwardMatch,
    reverse_match: ReverseMatch,
) -> Result<(S1::Mz, usize), SimilarityComputationError>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
    R: MultiRanged<Step = u32>,
    ForwardMatch: FnOnce(&S1, &S2) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
    ReverseMatch: FnOnce(&S2, &S1) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
{
    let left_peaks = prepare_peak_products(left, mz_power, intensity_power)?;
    let right_peaks = prepare_peak_products(right, mz_power, intensity_power)?;

    if left_peaks.max_f64 == 0.0 || right_peaks.max_f64 == 0.0 {
        return Ok((S1::Mz::zero(), 0));
    }

    if left.len() <= right.len() {
        let matching = forward_match(left, right)?;
        score_from_matching_greedy(MatchingScoreInputs {
            matching,
            row_f64: &left_peaks.as_f64,
            col_f64: &right_peaks.as_f64,
            row_products: &left_peaks.products,
            col_products: &right_peaks.products,
            max_row: left_peaks.max_f64,
            max_col: right_peaks.max_f64,
            left_norm: left_peaks.norm,
            right_norm: right_peaks.norm,
        })
    } else {
        let matching = reverse_match(right, left)?;
        score_from_matching_greedy(MatchingScoreInputs {
            matching,
            row_f64: &right_peaks.as_f64,
            col_f64: &left_peaks.as_f64,
            row_products: &right_peaks.products,
            col_products: &left_peaks.products,
            max_row: right_peaks.max_f64,
            max_col: left_peaks.max_f64,
            left_norm: left_peaks.norm,
            right_norm: right_peaks.norm,
        })
    }
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
