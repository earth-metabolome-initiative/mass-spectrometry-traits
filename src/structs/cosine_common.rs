use alloc::vec::Vec;

use geometric_traits::prelude::{Crouse, GenericImplicitValuedMatrix2D, RangedCSR2D, SparseMatrix};
use multi_ranged::MultiRanged;

use crate::structs::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{Spectrum, SpectrumFloat};

pub(crate) struct PreparedPeaks {
    pub(crate) products: Vec<f64>,
    pub(crate) norm: f64,
    pub(crate) max: f64,
}

pub(crate) struct MatchingScoreInputs<'a, R: MultiRanged<Step = u32>> {
    pub(crate) matching: RangedCSR2D<u32, u32, R>,
    pub(crate) row_products: &'a [f64],
    pub(crate) col_products: &'a [f64],
    pub(crate) max_row: f64,
    pub(crate) max_col: f64,
    pub(crate) left_norm: f64,
    pub(crate) right_norm: f64,
}

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct CosineConfig {
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
}

impl CosineConfig {
    #[inline]
    pub(crate) fn mz_tolerance(&self) -> f64 {
        self.mz_tolerance
    }

    #[inline]
    pub(crate) fn mz_power(&self) -> f64 {
        self.mz_power
    }

    #[inline]
    pub(crate) fn intensity_power(&self) -> f64 {
        self.intensity_power
    }

    #[inline]
    pub(crate) fn new(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
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
        impl $type_name {
            #[doc = $tolerance_doc]
            #[inline]
            pub fn mz_tolerance(&self) -> f64 {
                self.config.mz_tolerance()
            }

            /// Returns the power to which the mass/charge ratio is raised.
            #[inline]
            pub fn mz_power(&self) -> f64 {
                self.config.mz_power()
            }

            /// Returns the power to which the intensity is raised.
            #[inline]
            pub fn intensity_power(&self) -> f64 {
                self.config.intensity_power()
            }

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
            /// any numeric parameter is not finite or if
            /// `mz_tolerance` is negative.
            #[inline]
            pub fn new(
                mz_power: f64,
                intensity_power: f64,
                mz_tolerance: f64,
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
        impl<S1, S2> geometric_traits::prelude::ScalarSimilarity<S1, S2> for $type_name
        where
            S1: crate::traits::Spectrum,
            S2: crate::traits::Spectrum,
        {
            type Similarity =
                Result<(f64, usize), super::similarity_errors::SimilarityComputationError>;

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

        impl<S1, S2> crate::traits::ScalarSpectralSimilarity<S1, S2> for $type_name
        where
            S1: crate::traits::Spectrum,
            S2: crate::traits::Spectrum,
        {
        }
    };
}

pub(crate) use impl_cosine_wrapper_similarity;

#[inline]
pub(crate) fn ensure_finite(
    value: f64,
    name: &'static str,
) -> Result<f64, SimilarityComputationError> {
    if !value.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue(name));
    }
    Ok(value)
}

#[inline]
fn ensure_finite_for_config(value: f64, name: &'static str) -> Result<f64, SimilarityConfigError> {
    if !value.is_finite() {
        return Err(SimilarityConfigError::NonFiniteParameter(name));
    }
    Ok(value)
}

/// Compute `mz^mz_power * intensity^intensity_power` for each peak,
/// normalized so the maximum product is 1.0. Uses two-phase normalization
/// (components first, then products) to prevent f64 underflow when the raw
/// product would be smaller than the minimum subnormal (~5e-324).
pub(crate) fn normalized_peak_products<S: Spectrum>(
    spectrum: &S,
    mz_power: f64,
    intensity_power: f64,
) -> Result<Vec<f64>, SimilarityComputationError> {
    let n = spectrum.len();
    let mut mz_comps = Vec::with_capacity(n);
    let mut int_comps = Vec::with_capacity(n);

    for (mz, intensity) in spectrum.peaks() {
        let mz = mz.to_f64();
        let intensity = intensity.to_f64();
        let mc = mz.powf(mz_power);
        let ic = intensity.powf(intensity_power);
        ensure_finite(mc, "peak_product")?;
        ensure_finite(ic, "peak_product")?;
        mz_comps.push(mc);
        int_comps.push(ic);
    }

    // Normalize each component by its max so both are in [0, 1].
    let mz_max = mz_comps.iter().copied().fold(0.0_f64, f64::max);
    let int_max = int_comps.iter().copied().fold(0.0_f64, f64::max);

    if mz_max > 0.0 {
        for v in &mut mz_comps {
            *v /= mz_max;
        }
    }
    if int_max > 0.0 {
        for v in &mut int_comps {
            *v /= int_max;
        }
    }

    // Products of values in [0, 1] cannot underflow.
    let mut products: Vec<f64> = mz_comps
        .into_iter()
        .zip(int_comps)
        .map(|(m, i)| m * i)
        .collect();

    // Re-normalize so the maximum product is exactly 1.0.
    let prod_max = products.iter().copied().fold(0.0_f64, f64::max);
    if prod_max > 0.0 {
        for p in &mut products {
            *p /= prod_max;
        }
    }

    Ok(products)
}

pub(crate) fn prepare_peak_products<S: Spectrum>(
    spectrum: &S,
    mz_power: f64,
    intensity_power: f64,
) -> Result<PreparedPeaks, SimilarityComputationError> {
    let products = normalized_peak_products(spectrum, mz_power, intensity_power)?;

    let norm: f64 = products.iter().map(|&p| p * p).sum::<f64>().sqrt();
    ensure_finite(norm, "peak_norm")?;

    let max = products.iter().copied().fold(0.0_f64, f64::max);

    Ok(PreparedPeaks {
        products,
        norm,
        max,
    })
}

pub(crate) fn accumulate_assignment_scores(
    assignments: &[(u32, u32)],
    row_products: &[f64],
    col_products: &[f64],
) -> (f64, usize) {
    let mut score_sum = 0.0_f64;
    let mut n_matches = 0usize;

    for &(i, j) in assignments {
        let product = row_products[i as usize] * col_products[j as usize];
        if product != 0.0 {
            score_sum += product;
            n_matches += 1;
        }
    }

    (score_sum, n_matches)
}

#[inline]
pub(crate) fn finalize_similarity_score(
    score_sum: f64,
    n_matches: usize,
    left_norm: f64,
    right_norm: f64,
) -> Result<(f64, usize), SimilarityComputationError> {
    let denominator = left_norm * right_norm;
    ensure_finite(denominator, "similarity_denominator")?;
    if denominator == 0.0 {
        return Ok((0.0, 0));
    }

    let similarity = score_sum / denominator;
    ensure_finite(similarity, "similarity_score")?;

    if similarity > 1.0 {
        Ok((1.0, n_matches))
    } else {
        Ok((similarity, n_matches))
    }
}

pub(crate) fn score_from_matching<R>(
    inputs: MatchingScoreInputs<'_, R>,
) -> Result<(f64, usize), SimilarityComputationError>
where
    R: MultiRanged<Step = u32>,
{
    let non_edge_cost: f64 = 1.0f64 + f64::EPSILON;

    let map: GenericImplicitValuedMatrix2D<RangedCSR2D<u32, u32, R>, _, f64> =
        GenericImplicitValuedMatrix2D::new(inputs.matching, |(i, j)| {
            let cost = 1.0f64 + f64::EPSILON
                - (inputs.row_products[i as usize] / inputs.max_row)
                    * (inputs.col_products[j as usize] / inputs.max_col);
            if cost >= non_edge_cost {
                if (inputs.row_products[i as usize] * inputs.col_products[j as usize]) == 0.0 {
                    non_edge_cost
                } else {
                    1.0f64
                }
            } else {
                cost
            }
        });

    if map.is_empty() {
        return Ok((0.0, 0));
    }

    let max_cost: f64 = non_edge_cost + 1.0;

    let assignments: Vec<(u32, u32)> = map
        .crouse(non_edge_cost, max_cost)
        .map_err(|_| SimilarityComputationError::AssignmentFailed)?;

    let (score_sum, n_matches) =
        accumulate_assignment_scores(&assignments, inputs.row_products, inputs.col_products);
    finalize_similarity_score(score_sum, n_matches, inputs.left_norm, inputs.right_norm)
}

fn canonical_row_order(left: &[f64], right: &[f64]) -> bool {
    for (&l, &r) in left.iter().zip(right.iter()) {
        match l.total_cmp(&r) {
            core::cmp::Ordering::Less => return true,
            core::cmp::Ordering::Greater => return false,
            core::cmp::Ordering::Equal => continue,
        }
    }
    true
}

fn compute_cosine_similarity_with_scoring<S1, S2, R, ForwardMatch, ReverseMatch, ScoreFn>(
    left: &S1,
    right: &S2,
    mz_power: f64,
    intensity_power: f64,
    forward_match: ForwardMatch,
    reverse_match: ReverseMatch,
    score_fn: ScoreFn,
) -> Result<(f64, usize), SimilarityComputationError>
where
    S1: Spectrum,
    S2: Spectrum,
    R: MultiRanged<Step = u32>,
    ForwardMatch: FnOnce(&S1, &S2) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
    ReverseMatch: FnOnce(&S2, &S1) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>,
    ScoreFn:
        Fn(MatchingScoreInputs<'_, R>) -> Result<(f64, usize), SimilarityComputationError> + Copy,
{
    let left_peaks = prepare_peak_products(left, mz_power, intensity_power)?;
    let right_peaks = prepare_peak_products(right, mz_power, intensity_power)?;

    if left_peaks.max == 0.0 || right_peaks.max == 0.0 {
        return Ok((0.0, 0));
    }

    let use_forward = if left.len() != right.len() {
        left.len() < right.len()
    } else {
        canonical_row_order(&left_peaks.products, &right_peaks.products)
    };

    if use_forward {
        let matching = forward_match(left, right)?;
        score_fn(MatchingScoreInputs {
            matching,
            row_products: &left_peaks.products,
            col_products: &right_peaks.products,
            max_row: left_peaks.max,
            max_col: right_peaks.max,
            left_norm: left_peaks.norm,
            right_norm: right_peaks.norm,
        })
    } else {
        let matching = reverse_match(right, left)?;
        score_fn(MatchingScoreInputs {
            matching,
            row_products: &right_peaks.products,
            col_products: &left_peaks.products,
            max_row: right_peaks.max,
            max_col: left_peaks.max,
            left_norm: left_peaks.norm,
            right_norm: right_peaks.norm,
        })
    }
}

pub(crate) fn compute_cosine_similarity<S1, S2, R, ForwardMatch, ReverseMatch>(
    left: &S1,
    right: &S2,
    mz_power: f64,
    intensity_power: f64,
    forward_match: ForwardMatch,
    reverse_match: ReverseMatch,
) -> Result<(f64, usize), SimilarityComputationError>
where
    S1: Spectrum,
    S2: Spectrum,
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

pub(crate) fn validate_numeric_parameter(
    value: f64,
    name: &'static str,
) -> Result<(), SimilarityConfigError> {
    let _ = ensure_finite_for_config(value, name)?;
    Ok(())
}

pub(crate) fn validate_non_negative_tolerance(
    mz_tolerance: f64,
) -> Result<(), SimilarityConfigError> {
    validate_numeric_parameter(mz_tolerance, "mz_tolerance")?;
    if mz_tolerance < 0.0 {
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
pub(crate) fn linear_cosine_sweep(
    left_mz: &[f64],
    right_mz: &[f64],
    left_products: &[f64],
    right_products: &[f64],
    tolerance: f64,
    shift: f64,
) -> (f64, usize) {
    let mut score_sum = 0.0_f64;
    let mut n_matches = 0usize;
    let mut j = 0usize;

    for (i, &lmz) in left_mz.iter().enumerate() {
        let target = lmz - shift;
        // Advance j past peaks that are below the tolerance window.
        while j < right_mz.len() && right_mz[j] < target - tolerance {
            j += 1;
        }
        if j < right_mz.len()
            && right_mz[j] >= target - tolerance
            && right_mz[j] <= target + tolerance
        {
            let product = left_products[i] * right_products[j];
            if product != 0.0 {
                score_sum += product;
                n_matches += 1;
            }
            j += 1;
        }
    }

    (score_sum, n_matches)
}

/// Two-pointer sweep that collects matched `(left_index, right_index)` pairs.
///
/// A left value at `l` matches a right value `r` when
/// `|l - (r + shift)| <= tolerance`.  For direct matching pass `shift = 0`.
/// For neutral-loss matching, callers pass pre-computed neutral-loss arrays
/// and `shift = 0` so that the comparison becomes `|nl_left - nl_right|`.
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
        // Advance j past right peaks whose shifted value is below the window.
        while j < right_mz.len() && right_mz[j] + shift < lmz - tolerance {
            j += 1;
        }
        while j < right_mz.len()
            && right_mz[j] + shift >= lmz - tolerance
            && right_mz[j] + shift <= lmz + tolerance
        {
            matches.push((i, j));
            j += 1;
        }
    }

    matches
}

/// Collect direct and (when precursor difference > tolerance) shifted
/// linear matches, merge and deduplicate, then find the optimal
/// (maximum-weight) non-conflicting assignment via dynamic programming on
/// the conflict graph's path components.
pub(crate) fn optimal_modified_linear_matches(
    left_mz: &[f64],
    right_mz: &[f64],
    tolerance: f64,
    left_precursor: f64,
    right_precursor: f64,
    benefit_fn: impl Fn(usize, usize) -> f64,
) -> Vec<(usize, usize)> {
    let direct = collect_linear_matches(left_mz, right_mz, tolerance, 0.0);
    let candidates: Vec<(usize, usize)> = if right_precursor >= left_precursor - tolerance
        && right_precursor <= left_precursor + tolerance
    {
        direct
    } else {
        let left_nl: Vec<f64> = left_mz.iter().map(|&mz| mz - left_precursor).collect();
        let right_nl: Vec<f64> = right_mz.iter().map(|&mz| mz - right_precursor).collect();
        let shifted = collect_linear_matches(&left_nl, &right_nl, tolerance, 0.0);
        let mut pairs = Vec::with_capacity(direct.len() + shifted.len());
        pairs.extend(direct);
        pairs.extend(shifted);
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    };

    if candidates.is_empty() {
        return Vec::new();
    }

    let n_edges = candidates.len();

    use alloc::collections::BTreeMap;
    let mut left_edges: BTreeMap<usize, [Option<usize>; 2]> = BTreeMap::new();
    let mut right_edges: BTreeMap<usize, [Option<usize>; 2]> = BTreeMap::new();

    for (edge_idx, &(li, rj)) in candidates.iter().enumerate() {
        insert_edge_slot(left_edges.entry(li).or_insert([None; 2]), edge_idx);
        insert_edge_slot(right_edges.entry(rj).or_insert([None; 2]), edge_idx);
    }

    let mut neighbors: Vec<[Option<usize>; 2]> = alloc::vec![[None; 2]; n_edges];

    for slots in left_edges.values() {
        if let [Some(a), Some(b)] = *slots {
            insert_neighbor(&mut neighbors[a], b);
            insert_neighbor(&mut neighbors[b], a);
        }
    }
    for slots in right_edges.values() {
        if let [Some(a), Some(b)] = *slots {
            insert_neighbor(&mut neighbors[a], b);
            insert_neighbor(&mut neighbors[b], a);
        }
    }

    let mut visited = alloc::vec![false; n_edges];
    let mut selected = Vec::new();

    for start in 0..n_edges {
        if visited[start] {
            continue;
        }

        let mut end = start;
        let mut from = usize::MAX;
        while let Some(n) = iter_neighbors(neighbors[end]).find(|&n| n != from) {
            from = end;
            end = n;
        }

        let mut path = Vec::new();
        let mut current = end;
        let mut prev = usize::MAX;
        loop {
            visited[current] = true;
            path.push(current);
            match iter_neighbors(neighbors[current]).find(|&n| n != prev) {
                Some(n) => {
                    prev = current;
                    current = n;
                }
                None => break,
            }
        }

        if path.len() == 1 {
            selected.push(candidates[path[0]]);
            continue;
        }

        let benefits: Vec<f64> = path
            .iter()
            .map(|&e| {
                let (li, rj) = candidates[e];
                benefit_fn(li, rj)
            })
            .collect();

        let path_len = path.len();
        let mut dp = alloc::vec![0.0_f64; path_len + 1];
        dp[1] = benefits[0];
        for i in 2..=path_len {
            let take = dp[i - 2] + benefits[i - 1];
            let skip = dp[i - 1];
            dp[i] = if take >= skip { take } else { skip };
        }

        let mut i = path_len;
        while i > 0 {
            if i == 1 {
                if benefits[0] >= 0.0 {
                    selected.push(candidates[path[0]]);
                }
                break;
            }
            let take = dp[i - 2] + benefits[i - 1];
            if take >= dp[i - 1] {
                selected.push(candidates[path[i - 1]]);
                i -= 2;
            } else {
                i -= 1;
            }
        }
    }

    selected
}

fn insert_edge_slot(slots: &mut [Option<usize>; 2], edge_idx: usize) {
    if slots[0].is_none() {
        slots[0] = Some(edge_idx);
    } else {
        debug_assert!(slots[1].is_none(), "node has degree > 2");
        slots[1] = Some(edge_idx);
    }
}

fn insert_neighbor(slots: &mut [Option<usize>; 2], neighbor: usize) {
    if slots[0].is_none() {
        slots[0] = Some(neighbor);
    } else if slots[0] != Some(neighbor) {
        debug_assert!(
            slots[1].is_none() || slots[1] == Some(neighbor),
            "conflict-graph degree > 2"
        );
        slots[1] = Some(neighbor);
    }
}

fn iter_neighbors(slots: [Option<usize>; 2]) -> impl Iterator<Item = usize> {
    slots.into_iter().flatten()
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

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use geometric_traits::prelude::{MatrixMut, SparseMatrixMut};
    use multi_ranged::SimpleRange;

    use super::*;

    #[derive(Clone)]
    struct RawSpectrum {
        precursor_mz: f64,
        peaks: Vec<(f64, f64)>,
    }

    impl Spectrum for RawSpectrum {
        type Precision = f64;

        type SortedIntensitiesIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedMzIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedPeaksIter<'a>
            = core::iter::Copied<core::slice::Iter<'a, (f64, f64)>>
        where
            Self: 'a;

        fn len(&self) -> usize {
            self.peaks.len()
        }

        fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
            self.peaks.iter().map(|peak| peak.1)
        }

        fn intensity_nth(&self, n: usize) -> f64 {
            self.peaks[n].1
        }

        fn mz(&self) -> Self::SortedMzIter<'_> {
            self.peaks.iter().map(|peak| peak.0)
        }

        fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
            self.peaks[index..].iter().map(|peak| peak.0)
        }

        fn mz_nth(&self, n: usize) -> f64 {
            self.peaks[n].0
        }

        fn peaks(&self) -> Self::SortedPeaksIter<'_> {
            self.peaks.iter().copied()
        }

        fn peak_nth(&self, n: usize) -> (f64, f64) {
            self.peaks[n]
        }

        fn precursor_mz(&self) -> f64 {
            self.precursor_mz
        }
    }

    #[test]
    fn normalized_peak_products_preserves_all_zero_products() {
        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(10.0, 0.0), (20.0, 0.0)],
        };

        let products =
            normalized_peak_products(&spectrum, 1.0, 1.0).expect("normalization should succeed");
        assert_eq!(products, alloc::vec![0.0, 0.0]);
    }

    #[test]
    fn finalize_similarity_score_handles_zero_clamp_and_non_finite_denominator() {
        let zero = finalize_similarity_score(1.0, 4, 0.0, 2.0).expect("zero norm should succeed");
        assert_eq!(zero, (0.0, 0));

        let clamped =
            finalize_similarity_score(2.0, 3, 1.0, 1.0).expect("clamped similarity should succeed");
        assert_eq!(clamped, (1.0, 3));

        let error = finalize_similarity_score(1.0, 1, f64::INFINITY, 1.0)
            .expect_err("non-finite denominator should be rejected");
        assert_eq!(
            error,
            SimilarityComputationError::NonFiniteValue("similarity_denominator")
        );
    }

    #[test]
    fn accumulate_assignment_scores_ignores_zero_products() {
        let assignments = [(0_u32, 0_u32), (1_u32, 1_u32)];
        let (score, matches) = accumulate_assignment_scores(&assignments, &[0.0, 2.0], &[3.0, 4.0]);
        assert_eq!(score, 8.0);
        assert_eq!(matches, 1);
    }

    #[test]
    fn canonical_row_order_breaks_ties_lexicographically() {
        assert!(canonical_row_order(&[1.0, 2.0], &[1.0, 3.0]));
        assert!(!canonical_row_order(&[1.0, 3.0], &[1.0, 2.0]));
        assert!(canonical_row_order(&[1.0, 2.0], &[1.0, 2.0]));
    }

    #[test]
    fn optimal_modified_linear_matches_selects_best_edge_from_two_edge_path() {
        let selected =
            optimal_modified_linear_matches(&[1.0], &[1.0, 1.05], 0.1, 10.0, 10.0, |_, j| {
                if j == 0 { 10.0 } else { 1.0 }
            });

        assert_eq!(selected, alloc::vec![(0, 0)]);
    }

    #[test]
    fn insert_neighbor_ignores_duplicate_neighbor() {
        let mut slots = [Some(1_usize), None];
        insert_neighbor(&mut slots, 1);
        assert_eq!(slots, [Some(1), None]);

        insert_neighbor(&mut slots, 2);
        assert_eq!(slots, [Some(1), Some(2)]);
    }

    #[test]
    fn validate_well_separated_rejects_boundary_gap() {
        let error = validate_well_separated(&[10.0, 10.2], 0.1, "test spectrum")
            .expect_err("gap equal to 2 * tolerance should be rejected");
        assert_eq!(
            error,
            SimilarityComputationError::InvalidPeakSpacing("test spectrum")
        );
    }

    fn matching_from_pairs(
        rows: u32,
        cols: u32,
        pairs: &[(u32, u32)],
    ) -> RangedCSR2D<u32, u32, SimpleRange<u32>> {
        let mut matching = SparseMatrixMut::with_sparse_shape((rows, cols));
        MatrixMut::increase_shape(&mut matching, (rows, cols))
            .expect("shape increase should succeed");
        for &(row, col) in pairs {
            MatrixMut::add(&mut matching, (row, col)).expect("edge insertion should succeed");
        }
        matching
    }

    #[test]
    fn cosine_config_round_trips_values() {
        let config = CosineConfig::new(0.5, 2.0, 0.1).expect("config should build");
        assert_eq!(config.mz_power(), 0.5);
        assert_eq!(config.intensity_power(), 2.0);
        assert_eq!(config.mz_tolerance(), 0.1);
    }

    #[test]
    fn prepare_peak_products_reports_norm_and_max() {
        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(10.0, 1.0), (20.0, 4.0)],
        };

        let prepared =
            prepare_peak_products(&spectrum, 0.0, 1.0).expect("peak preparation should succeed");
        assert_eq!(prepared.products, alloc::vec![0.25, 1.0]);
        assert!((prepared.norm - (1.0625_f64).sqrt()).abs() <= 1.0e-12);
        assert_eq!(prepared.max, 1.0);
    }

    #[test]
    fn score_from_matching_handles_zero_and_negative_cost_branches() {
        let zero_score = score_from_matching(MatchingScoreInputs {
            matching: matching_from_pairs(1, 1, &[(0, 0)]),
            row_products: &[0.0],
            col_products: &[1.0],
            max_row: 1.0,
            max_col: 1.0,
            left_norm: 1.0,
            right_norm: 1.0,
        })
        .expect("zero-product matching should succeed");
        assert_eq!(zero_score, (0.0, 0));

        let negative_score = score_from_matching(MatchingScoreInputs {
            matching: matching_from_pairs(1, 1, &[(0, 0)]),
            row_products: &[-1.0],
            col_products: &[1.0],
            max_row: 1.0,
            max_col: 1.0,
            left_norm: 1.0,
            right_norm: 1.0,
        })
        .expect("negative-product matching should still score");
        assert_eq!(negative_score, (-1.0, 1));
    }

    #[test]
    fn compute_cosine_similarity_uses_forward_match_for_shorter_left() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(10.0, 2.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(10.0, 2.0), (20.0, 1.0)],
        };
        let forward_called = Cell::new(false);

        let result = compute_cosine_similarity::<_, _, SimpleRange<u32>, _, _>(
            &left,
            &right,
            0.0,
            1.0,
            |_, _| {
                forward_called.set(true);
                Ok(matching_from_pairs(1, 2, &[(0, 0)]))
            },
            |_, _| panic!("reverse branch should not be used"),
        )
        .expect("similarity should succeed");

        assert!(forward_called.get());
        assert_eq!(result.1, 1);
        assert!(result.0.is_finite() && result.0 > 0.0);
    }

    #[test]
    fn compute_cosine_similarity_can_take_reverse_branch_for_tie_breaks() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(10.0, 1.0), (20.0, 1.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(10.0, 1.0), (100.0, 1.0)],
        };
        let reverse_called = Cell::new(false);

        let result = compute_cosine_similarity::<_, _, SimpleRange<u32>, _, _>(
            &left,
            &right,
            1.0,
            1.0,
            |_, _| panic!("forward branch should not be used"),
            |_, _| {
                reverse_called.set(true);
                Ok(matching_from_pairs(2, 2, &[(0, 0)]))
            },
        )
        .expect("similarity should succeed");

        assert!(reverse_called.get());
        assert_eq!(result.1, 1);
        assert!(result.0.is_finite() && result.0 > 0.0);
    }

    #[test]
    fn optimal_modified_linear_matches_backtracks_single_edge_and_skips_visited_start() {
        let selected =
            optimal_modified_linear_matches(&[1.0], &[1.0, 1.4], 0.5, 10.0, 10.0, |_, j| {
                if j == 0 { 3.0 } else { 1.0 }
            });

        assert_eq!(selected, alloc::vec![(0, 0)]);
    }
}
