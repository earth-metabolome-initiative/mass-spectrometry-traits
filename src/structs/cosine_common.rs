use alloc::vec::Vec;

use geometric_traits::prelude::{
    Crouse, Finite, GenericImplicitValuedMatrix2D, Number, RangedCSR2D, SparseMatrix, TotalOrd,
};
use multi_ranged::MultiRanged;
use num_traits::{Float, Pow, ToPrimitive, Zero};

use crate::numeric_validation::{NumericValidationError, checked_to_f64};
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
        Ok(Self {
            mz_power,
            intensity_power,
            mz_tolerance,
        })
    }
}

macro_rules! impl_cosine_wrapper_config_api {
    ($type_name:ident, $constructor_description:expr, $tolerance_doc:expr) => {
        impl<EXP: geometric_traits::prelude::Number, MZ: geometric_traits::prelude::Number>
            $type_name<EXP, MZ>
        {
            #[doc = $tolerance_doc]
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

        impl<EXP, MZ> $type_name<EXP, MZ>
        where
            EXP: geometric_traits::prelude::Number + num_traits::ToPrimitive,
            MZ: geometric_traits::prelude::Number + num_traits::ToPrimitive + PartialOrd,
        {
            #[doc = concat!("Creates a new instance of ", $constructor_description, ".")]
            ///
            /// # Arguments
            ///
            /// * `mz_power`: The power to which the mass/charge ratio is raised.
            /// * `intensity_power`: The power to which the intensity is raised.
            #[doc = concat!("* `mz_tolerance`: ", $tolerance_doc)]
            ///
            /// # Errors
            ///
            /// Returns [`super::similarity_errors::SimilarityConfigError`] if
            /// any numeric parameter is not finite/representable or if
            /// `mz_tolerance` is negative.
            #[inline]
            pub fn new(
                mz_power: EXP,
                intensity_power: EXP,
                mz_tolerance: MZ,
            ) -> Result<Self, super::similarity_errors::SimilarityConfigError> {
                Ok(Self {
                    config: super::cosine_common::CosineConfig::new(
                        mz_power,
                        intensity_power,
                        mz_tolerance,
                    )?,
                })
            }
        }
    };
}

pub(crate) use impl_cosine_wrapper_config_api;

macro_rules! impl_cosine_wrapper_similarity {
    (
        $type_name:ident,
        $compute_fn:path,
        $mz_tolerance:ident,
        $row:ident,
        $col:ident,
        $forward_match:expr,
        $reverse_match:expr
    ) => {
        impl<EXP, S1, S2> geometric_traits::prelude::ScalarSimilarity<S1, S2>
            for $type_name<EXP, S1::Mz>
        where
            EXP: geometric_traits::prelude::Number,
            S1::Mz: num_traits::Pow<EXP, Output = S1::Mz>
                + num_traits::Float
                + geometric_traits::prelude::Number
                + geometric_traits::prelude::Finite
                + geometric_traits::prelude::TotalOrd
                + num_traits::ToPrimitive,
            S1: crate::traits::Spectrum<Intensity = <S1 as crate::traits::Spectrum>::Mz>,
            S2: crate::traits::Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
        {
            type Similarity =
                Result<(S1::Mz, usize), super::similarity_errors::SimilarityComputationError>;

            fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
                let $mz_tolerance = self.config.mz_tolerance();
                $compute_fn(
                    left,
                    right,
                    self.config.mz_power(),
                    self.config.intensity_power(),
                    |$row, $col| $forward_match,
                    |$row, $col| $reverse_match,
                )
            }
        }

        impl<S1, S2, EXP> crate::traits::ScalarSpectralSimilarity<S1, S2>
            for $type_name<EXP, S1::Mz>
        where
            EXP: geometric_traits::prelude::Number,
            S1::Mz: num_traits::Pow<EXP, Output = S1::Mz>
                + num_traits::Float
                + geometric_traits::prelude::Finite
                + geometric_traits::prelude::TotalOrd,
            S1: crate::traits::Spectrum<Intensity = <S1 as crate::traits::Spectrum>::Mz>,
            S2: crate::traits::Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
        {
        }
    };
}

pub(crate) use impl_cosine_wrapper_similarity;

#[inline]
pub(crate) fn to_f64_checked_for_computation<T: ToPrimitive>(
    value: T,
    name: &'static str,
) -> Result<f64, SimilarityComputationError> {
    checked_to_f64(value, name).map_err(|error| match error {
        NumericValidationError::NonRepresentable(name) => {
            SimilarityComputationError::ValueNotRepresentable(name)
        }
        NumericValidationError::NonFinite(name) => SimilarityComputationError::NonFiniteValue(name),
    })
}

#[inline]
fn to_f64_checked_for_config<T: ToPrimitive>(
    value: T,
    name: &'static str,
) -> Result<f64, SimilarityConfigError> {
    checked_to_f64(value, name).map_err(|error| match error {
        NumericValidationError::NonRepresentable(name) => {
            SimilarityConfigError::NonRepresentableParameter(name)
        }
        NumericValidationError::NonFinite(name) => SimilarityConfigError::NonFiniteParameter(name),
    })
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
        let score_f64 = to_f64_checked_for_computation(score, "peak_product")?;

        products.push(score);
        as_f64.push(score_f64);
        squared_sum += score * score;
    }

    let norm = squared_sum.sqrt();
    let _ = to_f64_checked_for_computation(norm, "peak_norm")?;

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

#[inline]
pub(crate) fn finalize_similarity_score<MZ>(
    score_sum: MZ,
    n_matches: usize,
    left_norm: MZ,
    right_norm: MZ,
) -> Result<(MZ, usize), SimilarityComputationError>
where
    MZ: Float + Number + Finite + TotalOrd + ToPrimitive,
{
    let denominator = left_norm * right_norm;
    let denominator_f64 = to_f64_checked_for_computation(denominator, "similarity_denominator")?;
    if denominator_f64 == 0.0 {
        return Ok((MZ::zero(), 0));
    }

    let similarity = score_sum / denominator;
    let similarity_f64 = to_f64_checked_for_computation(similarity, "similarity_score")?;

    if similarity_f64 > 1.0 {
        Ok((MZ::one(), n_matches))
    } else {
        Ok((similarity, n_matches))
    }
}

pub(crate) fn score_from_matching<MZ, R>(
    inputs: MatchingScoreInputs<'_, MZ, R>,
) -> Result<(MZ, usize), SimilarityComputationError>
where
    MZ: Float + Number + Finite + TotalOrd + ToPrimitive,
    R: MultiRanged<Step = u32>,
{
    // `row_f64`, `col_f64`, `max_row`, and `max_col` are derived from
    // `prepare_peak_products`, which already validates representability and
    // finiteness. Keep runtime checks focused on arithmetic states created
    // below instead of re-validating immutable inputs.
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
    finalize_similarity_score(score_sum, n_matches, inputs.left_norm, inputs.right_norm)
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
    let mut candidates: Vec<(f64, u32, u32)> = Vec::new();
    for (i, j) in SparseMatrix::sparse_coordinates(&inputs.matching) {
        let weight = inputs.row_f64[i as usize] * inputs.col_f64[j as usize];
        let _ = to_f64_checked_for_computation(weight, "candidate_weight")?;
        candidates.push((weight, i, j));
    }

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
    finalize_similarity_score(score_sum, n_matches, inputs.left_norm, inputs.right_norm)
}

fn compute_cosine_similarity_with_scoring<EXP, S1, S2, R, ForwardMatch, ReverseMatch, ScoreFn>(
    left: &S1,
    right: &S2,
    mz_power: EXP,
    intensity_power: EXP,
    forward_match: ForwardMatch,
    reverse_match: ReverseMatch,
    score_fn: ScoreFn,
) -> Result<(S1::Mz, usize), SimilarityComputationError>
where
    EXP: Number,
    S1::Mz: Pow<EXP, Output = S1::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
    R: MultiRanged<Step = u32>,
    ForwardMatch: FnOnce(&S1, &S2) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
    ReverseMatch: FnOnce(&S2, &S1) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
    ScoreFn: Fn(
            MatchingScoreInputs<'_, S1::Mz, R>,
        ) -> Result<(S1::Mz, usize), SimilarityComputationError>
        + Copy,
{
    let left_peaks = prepare_peak_products(left, mz_power, intensity_power)?;
    let right_peaks = prepare_peak_products(right, mz_power, intensity_power)?;

    if left_peaks.max_f64 == 0.0 || right_peaks.max_f64 == 0.0 {
        return Ok((S1::Mz::zero(), 0));
    }

    if left.len() <= right.len() {
        let matching = forward_match(left, right)?;
        score_fn(MatchingScoreInputs {
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
        score_fn(MatchingScoreInputs {
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
    compute_cosine_similarity_with_scoring(
        left,
        right,
        mz_power,
        intensity_power,
        forward_match,
        reverse_match,
        score_from_matching,
    )
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
    compute_cosine_similarity_with_scoring(
        left,
        right,
        mz_power,
        intensity_power,
        forward_match,
        reverse_match,
        score_from_matching_greedy,
    )
}

pub(crate) fn validate_numeric_parameter<T: ToPrimitive>(
    value: T,
    name: &'static str,
) -> Result<(), SimilarityConfigError> {
    let _ = to_f64_checked_for_config(value, name)?;
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

/// Two-pointer sweep that accumulates `left_products[i] * right_products[j]`
/// for each matched pair, returning `(score_sum, n_matches)`.
///
/// `shift` is `left_precursor - right_precursor`: a left peak at `m` matches a
/// right peak near `m - shift`.  For direct (unshifted) matching, pass 0.0.
///
/// Requires that both mz slices are sorted and well-separated (consecutive
/// peaks > 2 * tolerance apart), guaranteeing at most one match per peak.
#[inline]
pub(crate) fn linear_cosine_sweep<MZ>(
    left_mz: &[f64],
    right_mz: &[f64],
    left_products: &[MZ],
    right_products: &[MZ],
    tolerance: f64,
    shift: f64,
) -> (MZ, usize)
where
    MZ: Number + Zero,
{
    let mut score_sum = MZ::zero();
    let mut n_matches = 0usize;
    let mut j = 0usize;

    for (i, &lmz) in left_mz.iter().enumerate() {
        let target = lmz - shift;
        // Advance j past peaks that are below the tolerance window.
        while j < right_mz.len() && right_mz[j] < target - tolerance {
            j += 1;
        }
        if j < right_mz.len() && (right_mz[j] - target).abs() <= tolerance {
            score_sum += left_products[i] * right_products[j];
            n_matches += 1;
            j += 1;
        }
    }

    (score_sum, n_matches)
}

/// Two-pointer sweep that collects matched `(left_index, right_index)` pairs.
///
/// Same semantics as [`linear_cosine_sweep`]: `shift` is `left_precursor -
/// right_precursor`, so a left peak at `m` matches a right peak near
/// `m - shift`.
#[inline]
pub(crate) fn collect_linear_matches(
    left_mz: &[f64],
    right_mz: &[f64],
    tolerance: f64,
    shift: f64,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let mut j = 0usize;

    for (i, &lmz) in left_mz.iter().enumerate() {
        let target = lmz - shift;
        while j < right_mz.len() && right_mz[j] < target - tolerance {
            j += 1;
        }
        if j < right_mz.len() && (right_mz[j] - target).abs() <= tolerance {
            matches.push((i, j));
            j += 1;
        }
    }

    matches
}

/// Runtime validation that consecutive peaks in the given mz slice are
/// greater than `2 * tolerance` apart.
pub(crate) fn validate_well_separated(
    mz: &[f64],
    tolerance: f64,
    label: &'static str,
) -> Result<(), SimilarityComputationError> {
    let min_gap = 2.0 * tolerance;
    for w in mz.windows(2) {
        if w[1] - w[0] <= min_gap {
            return Err(SimilarityComputationError::InvalidPeakSpacing(label));
        }
    }
    Ok(())
}
