//! Linear entropy similarity (Li et al.) on the GPU.
//!
//! Mirrors `LinearEntropy::similarity` (CPU) in `src/structs/linear_entropy.rs`:
//! per-row `mz^p * intensity^q` products are normalized to a probability
//! distribution. Optionally re-weighted by Shannon entropy when `H < 3.0`.
//! Pairs are matched by a two-pointer sweep with `mz_tolerance` window. Each
//! pair contributes `(a+b)log2(a+b) - a log2 a - b log2 b` to the raw score.
//! The final similarity is `(raw_score / 2).clamp(0.0, 1.0)`.

use burn_cubecl::cubecl::prelude::*;

use crate::burn::SpectralPairScorer;
use crate::burn::kernels::modified_dp::{
    build_modified_conflict_graph, collect_modified_candidates, first_modified_neighbor_not_from,
    sort_and_dedupe_modified_candidates,
};
use crate::burn::metrics::{LinearEntropyMetric, ModifiedLinearEntropyMetric};

#[cube]
impl SpectralPairScorer for LinearEntropyMetric {
    fn score_rows<F: Float>(
        left_mz: &Tensor<F>,
        left_intensity: &Tensor<F>,
        _left_precursor: &Tensor<F>,
        left_row: usize,
        right_mz: &Tensor<F>,
        right_intensity: &Tensor<F>,
        _right_precursor: &Tensor<F>,
        right_row: usize,
        mz_p: F,
        intensity_p: F,
        tolerance: F,
        _eps: F,
        #[comptime] max_peaks: u32,
        #[comptime] weighted: bool,
    ) -> F {
        linear_entropy_score_rows::<F>(
            left_mz,
            left_intensity,
            left_row,
            right_mz,
            right_intensity,
            right_row,
            mz_p,
            intensity_p,
            tolerance,
            max_peaks,
            weighted,
        )
    }
}

/// Linear entropy scorer body, shared by `LinearEntropyMetric` and (via
/// the modified-DP path) `ModifiedLinearEntropyMetric` in Step 8.
#[cube]
pub fn linear_entropy_score_rows<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_row: usize,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_row: usize,
    mz_p: F,
    intensity_p: F,
    tolerance: F,
    #[comptime] max_peaks: u32,
    #[comptime] weighted: bool,
) -> F {
    let max_peaks_usize = comptime!(max_peaks as usize);
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let two = F::new(2.0_f32);

    let mut left_products = Array::<F>::new(max_peaks_usize);
    let mut right_products = Array::<F>::new(max_peaks_usize);

    let mut similarity = zero;

    if left_peaks <= max_peaks_usize && right_peaks <= max_peaks_usize {
        prepare_entropy_row::<F>(
            left_mz,
            left_intensity,
            left_row,
            &mut left_products,
            left_peaks,
            max_peaks_usize,
            mz_p,
            intensity_p,
            weighted,
        );
        prepare_entropy_row::<F>(
            right_mz,
            right_intensity,
            right_row,
            &mut right_products,
            right_peaks,
            max_peaks_usize,
            mz_p,
            intensity_p,
            weighted,
        );

        let mut left_cursor = 0usize;
        let mut right_cursor = 0usize;
        let mut score = zero;
        while left_cursor < left_peaks && right_cursor < right_peaks {
            let left_int = left_products[left_cursor];
            if left_int <= zero {
                left_cursor += 1;
            } else {
                let right_int = right_products[right_cursor];
                if right_int <= zero {
                    right_cursor += 1;
                } else {
                    let mz_left =
                        left_mz[left_mz.stride(0) * left_row + left_mz.stride(1) * left_cursor];
                    let mz_right = right_mz
                        [right_mz.stride(0) * right_row + right_mz.stride(1) * right_cursor];
                    let delta = mz_left - mz_right;

                    if delta.abs() <= tolerance {
                        score += entropy_pair::<F>(left_int, right_int);
                        left_cursor += 1;
                        right_cursor += 1;
                    } else if mz_left + tolerance < mz_right {
                        left_cursor += 1;
                    } else {
                        right_cursor += 1;
                    }
                }
            }
        }

        similarity = (score / two).max(zero).min(one);
    }

    similarity
}

/// Single-row preparation: compute `mz^p * intensity^q`, normalize to a
/// probability distribution (sum = 1), then optionally apply Shannon-entropy
/// re-weighting when `H < 3.0` and the `weighted` comptime flag is set.
#[cube]
pub fn prepare_entropy_row<F: Float>(
    mz_tensor: &Tensor<F>,
    intensity_tensor: &Tensor<F>,
    row: usize,
    products: &mut Array<F>,
    peak_count: usize,
    #[comptime] max_peaks: usize,
    mz_p: F,
    intensity_p: F,
    #[comptime] weighted: bool,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let three = F::new(3.0_f32);
    let quarter = F::new(0.25_f32);

    let mut sum = zero;
    for peak in 0..peak_count {
        let intensity =
            intensity_tensor[row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
        if intensity > zero {
            let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
            let product = mz.powf(mz_p) * intensity.powf(intensity_p);
            products[peak] = product;
            sum += product;
        } else {
            products[peak] = zero;
        }
    }
    for peak in peak_count..max_peaks {
        products[peak] = zero;
    }

    if sum > zero {
        for peak in 0..peak_count {
            products[peak] /= sum;
        }
    }

    if weighted {
        let mut entropy = zero;
        for peak in 0..peak_count {
            let p = products[peak];
            if p > zero {
                entropy -= p * p.ln();
            }
        }
        if entropy < three {
            let power = quarter * (one + entropy);
            let mut weighted_sum = zero;
            for peak in 0..peak_count {
                let p = products[peak];
                if p > zero {
                    let w = p.powf(power);
                    products[peak] = w;
                    weighted_sum += w;
                } else {
                    products[peak] = zero;
                }
            }
            if weighted_sum > zero {
                for peak in 0..peak_count {
                    products[peak] /= weighted_sum;
                }
            }
        }
    }
}

#[cube]
impl SpectralPairScorer for ModifiedLinearEntropyMetric {
    fn score_rows<F: Float>(
        left_mz: &Tensor<F>,
        left_intensity: &Tensor<F>,
        left_precursor: &Tensor<F>,
        left_row: usize,
        right_mz: &Tensor<F>,
        right_intensity: &Tensor<F>,
        right_precursor: &Tensor<F>,
        right_row: usize,
        mz_p: F,
        intensity_p: F,
        tolerance: F,
        eps: F,
        #[comptime] max_peaks: u32,
        #[comptime] weighted: bool,
    ) -> F {
        modified_linear_entropy_score_rows::<F>(
            left_mz,
            left_intensity,
            left_precursor,
            left_row,
            right_mz,
            right_intensity,
            right_precursor,
            right_row,
            mz_p,
            intensity_p,
            tolerance,
            eps,
            max_peaks,
            weighted,
        )
    }
}

/// Modified (precursor-shifted) linear entropy scorer.
///
/// Same structure as `modified_linear_cosine_score_rows`, collect direct
/// and precursor-shifted candidate matches, dedupe, build the conflict graph,
/// run DP across each path component, but the prepass is the entropy
/// prepass (`mz^p * intensity^q` normalized to a probability distribution +
/// optional Shannon weighting) and the per-pair contribution is
/// `entropy_pair(left[i], right[j])`. Final similarity is
/// `(raw_score / 2).clamp(0.0, 1.0)`.
#[cube]
pub fn modified_linear_entropy_score_rows<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_precursor: &Tensor<F>,
    left_row: usize,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_precursor: &Tensor<F>,
    right_row: usize,
    mz_p: F,
    intensity_p: F,
    tolerance: F,
    eps: F,
    #[comptime] max_peaks: u32,
    #[comptime] weighted: bool,
) -> F {
    let max_peaks_usize = comptime!(max_peaks as usize);
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let two = F::new(2.0_f32);
    let invalid = RuntimeCell::<u32>::new(4_294_967_295u32).read();
    let candidate_capacity = comptime!(max_peaks_usize * 2usize);
    let dp_capacity = comptime!(max_peaks_usize * 2usize + 1usize);

    let mut similarity = zero;

    if left_peaks <= max_peaks_usize && right_peaks <= max_peaks_usize {
        let mut left_products = Array::<F>::new(max_peaks_usize);
        let mut right_products = Array::<F>::new(max_peaks_usize);

        prepare_entropy_row::<F>(
            left_mz,
            left_intensity,
            left_row,
            &mut left_products,
            left_peaks,
            max_peaks_usize,
            mz_p,
            intensity_p,
            weighted,
        );
        prepare_entropy_row::<F>(
            right_mz,
            right_intensity,
            right_row,
            &mut right_products,
            right_peaks,
            max_peaks_usize,
            mz_p,
            intensity_p,
            weighted,
        );

        let mut candidate_left = Array::<u32>::new(candidate_capacity);
        let mut candidate_right = Array::<u32>::new(candidate_capacity);
        let left_precursor_value = left_precursor[left_row * left_precursor.stride(0)];
        let right_precursor_value = right_precursor[right_row * right_precursor.stride(0)];

        let mut candidate_count = collect_modified_candidates::<F>(
            left_mz,
            &left_products,
            left_row,
            left_peaks,
            left_precursor_value,
            right_mz,
            &right_products,
            right_row,
            right_peaks,
            right_precursor_value,
            tolerance,
            &mut candidate_left,
            &mut candidate_right,
            comptime!(candidate_capacity as u32),
        ) as usize;

        if candidate_count > 0usize {
            candidate_count = sort_and_dedupe_modified_candidates(
                &mut candidate_left,
                &mut candidate_right,
                candidate_count as u32,
            ) as usize;

            let mut left_slot_a = Array::<u32>::new(max_peaks_usize);
            let mut left_slot_b = Array::<u32>::new(max_peaks_usize);
            let mut right_slot_a = Array::<u32>::new(max_peaks_usize);
            let mut right_slot_b = Array::<u32>::new(max_peaks_usize);
            let mut neighbor_a = Array::<u32>::new(candidate_capacity);
            let mut neighbor_b = Array::<u32>::new(candidate_capacity);
            let mut visited = Array::<u32>::new(candidate_capacity);

            build_modified_conflict_graph(
                &candidate_left,
                &candidate_right,
                candidate_count as u32,
                &mut left_slot_a,
                &mut left_slot_b,
                &mut right_slot_a,
                &mut right_slot_b,
                &mut neighbor_a,
                &mut neighbor_b,
                &mut visited,
                invalid,
                max_peaks,
            );

            let mut path = Array::<u32>::new(candidate_capacity);
            let mut benefits = Array::<F>::new(candidate_capacity);
            let mut dp = Array::<F>::new(dp_capacity);
            let mut score = zero;

            for start in 0..candidate_count {
                if visited[start] == 0u32 {
                    let mut end = start;
                    let mut from = invalid;
                    loop {
                        let next = first_modified_neighbor_not_from(
                            neighbor_a[end],
                            neighbor_b[end],
                            from,
                            invalid,
                        );
                        if next == invalid {
                            break;
                        }
                        from = end as u32;
                        end = next as usize;
                    }

                    let mut path_len = 0usize;
                    let mut current = end;
                    let mut previous = invalid;
                    loop {
                        visited[current] = 1u32;
                        path[path_len] = current as u32;
                        path_len += 1;

                        let next = first_modified_neighbor_not_from(
                            neighbor_a[current],
                            neighbor_b[current],
                            previous,
                            invalid,
                        );
                        if next == invalid {
                            break;
                        }
                        previous = current as u32;
                        current = next as usize;
                    }

                    if path_len == 1usize {
                        let edge = path[0] as usize;
                        let left_peak = candidate_left[edge] as usize;
                        let right_peak = candidate_right[edge] as usize;
                        score +=
                            entropy_pair::<F>(left_products[left_peak], right_products[right_peak]);
                    } else {
                        for path_index in 0..path_len {
                            let edge = path[path_index] as usize;
                            let left_peak = candidate_left[edge] as usize;
                            let right_peak = candidate_right[edge] as usize;
                            benefits[path_index] = entropy_pair::<F>(
                                left_products[left_peak],
                                right_products[right_peak],
                            )
                            .max(eps);
                        }

                        dp[0] = zero;
                        dp[1] = benefits[0];
                        for index in 2..(path_len + 1usize) {
                            let take = dp[index - 2usize] + benefits[index - 1usize];
                            let skip = dp[index - 1usize];
                            dp[index] = if take >= skip { take } else { skip };
                        }

                        let mut index = path_len;
                        while index > 0usize {
                            if index == 1usize {
                                let edge = path[0] as usize;
                                let left_peak = candidate_left[edge] as usize;
                                let right_peak = candidate_right[edge] as usize;
                                score += entropy_pair::<F>(
                                    left_products[left_peak],
                                    right_products[right_peak],
                                );
                                break;
                            }
                            let take = dp[index - 2usize] + benefits[index - 1usize];
                            if take >= dp[index - 1usize] {
                                let edge = path[index - 1usize] as usize;
                                let left_peak = candidate_left[edge] as usize;
                                let right_peak = candidate_right[edge] as usize;
                                score += entropy_pair::<F>(
                                    left_products[left_peak],
                                    right_products[right_peak],
                                );
                                index -= 2usize;
                            } else {
                                index -= 1usize;
                            }
                        }
                    }
                }
            }

            similarity = (score / two).max(zero).min(one);
        }
    }

    similarity
}

/// Pair entropy contribution `(a + b) * log2(a + b) - a * log2(a) - b * log2(b)`.
///
/// Mirrors `entropy_common::entropy_pair` (CPU) in `src/structs/entropy_common.rs`.
/// Computed in natural log and rescaled by `1 / ln(2) = log2(e)` because the
/// CubeCL `Float` trait exposes `ln` (via supertrait `Log`) but not `log2`.
/// Skips the `x * ln(x)` term whenever `x` is zero (the convention
/// `0 * log(0) = 0`), so padded/missing peaks contribute zero.
#[cube]
pub fn entropy_pair<F: Float>(a: F, b: F) -> F {
    let zero = F::new(0.0_f32);
    let log2_e = F::new(core::f32::consts::LOG2_E);
    let mut result_ln = zero;
    let ab = a + b;
    if ab > zero {
        result_ln += ab * ab.ln();
    }
    if a > zero {
        result_ln -= a * a.ln();
    }
    if b > zero {
        result_ln -= b * b.ln();
    }
    result_ln * log2_e
}
