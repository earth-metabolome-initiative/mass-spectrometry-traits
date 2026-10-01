use burn_cubecl::cubecl::prelude::*;

use crate::burn::SpectralPairScorer;
use crate::burn::kernels::modified_dp::{
    build_modified_conflict_graph, collect_modified_candidates, first_modified_neighbor_not_from,
    sort_and_dedupe_modified_candidates,
};
use crate::burn::metrics::{LinearCosineMetric, ModifiedLinearCosineMetric};

#[cube]
impl SpectralPairScorer for LinearCosineMetric {
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
        eps: F,
        #[comptime] max_peaks: u32,
        #[comptime] _weighted: bool,
    ) -> F {
        linear_cosine_score_rows::<F>(
            left_mz,
            left_intensity,
            left_row,
            right_mz,
            right_intensity,
            right_row,
            mz_p,
            intensity_p,
            tolerance,
            eps,
            max_peaks,
        )
    }
}

/// Two-pointer sweep over two preprocessed peak rows.
///
/// Mirrors `LinearCosine::similarity` (CPU) in `src/structs/linear_cosine.rs`:
/// computes per-row maxima for m/z, intensity, and product space, stores the
/// normalized peak products in per-thread scratch arrays
/// (`Array::<F>::new(max_peaks)`), then performs the matching sweep by
/// reading from the scratch arrays directly. This caching pattern avoids the
/// repeated `powf` calls that the naive call-`peak_product`-on-demand
/// formulation incurs, each peak's `intensity.powf(p)` and `mz.powf(p)` are
/// computed once per row pair, not once per pass.
#[cube]
pub fn linear_cosine_score_rows<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_row: usize,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_row: usize,
    mz_p: F,
    intensity_p: F,
    tolerance: F,
    eps: F,
    #[comptime] max_peaks: u32,
) -> F {
    let max_peaks_usize = comptime!(max_peaks as usize);
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);

    let mut left_products = Array::<F>::new(max_peaks_usize);
    let mut right_products = Array::<F>::new(max_peaks_usize);

    let left_norm_square = prepare_linear_cosine_row::<F>(
        left_mz,
        left_intensity,
        left_row,
        &mut left_products,
        left_peaks,
        max_peaks_usize,
        mz_p,
        intensity_p,
        eps,
    );
    let right_norm_square = prepare_linear_cosine_row::<F>(
        right_mz,
        right_intensity,
        right_row,
        &mut right_products,
        right_peaks,
        max_peaks_usize,
        mz_p,
        intensity_p,
        eps,
    );

    let mut left_cursor = 0usize;
    let mut right_cursor = 0usize;
    let mut score = zero;
    while left_cursor < left_peaks && right_cursor < right_peaks {
        let left_product = left_products[left_cursor];
        if left_product <= zero {
            left_cursor += 1;
        } else {
            let right_product = right_products[right_cursor];
            if right_product <= zero {
                right_cursor += 1;
            } else {
                let mz_left =
                    left_mz[left_mz.stride(0) * left_row + left_mz.stride(1) * left_cursor];
                let mz_right =
                    right_mz[right_mz.stride(0) * right_row + right_mz.stride(1) * right_cursor];
                let delta = mz_left - mz_right;

                if delta.abs() <= tolerance {
                    score += left_product * right_product;
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

    let left_norm = (left_norm_square + eps).sqrt();
    let right_norm = (right_norm_square + eps).sqrt();
    let similarity = score / (left_norm * right_norm + eps);
    similarity.max(zero).min(one)
}

/// Three-pass prepass for one spectrum row:
///
/// 1. Find `intensity_max = max(intensity[i]^p)` and `mz_max = max(mz[i]^p)`
///    over non-padding peaks (`intensity > 0`).
/// 2. Compute `products[i] = (intensity[i]^p / intensity_max) * (mz[i]^p / mz_max)`
///    and store in `products`. Track `product_max`.
/// 3. Normalize `products[i] /= product_max` in place and accumulate
///    `norm_square = sum(products[i]^2)`.
///
/// Returns `norm_square`. Padding slots (`peak >= peak_count`) are zeroed so
/// the matching sweep can read them unconditionally without a guard.
#[cube]
pub fn prepare_linear_cosine_row<F: Float>(
    mz_tensor: &Tensor<F>,
    intensity_tensor: &Tensor<F>,
    row: usize,
    products: &mut Array<F>,
    peak_count: usize,
    #[comptime] max_peaks: usize,
    mz_p: F,
    intensity_p: F,
    eps: F,
) -> F {
    let zero = F::new(0.0_f32);

    let mut intensity_max = zero;
    let mut mz_max = zero;
    for peak in 0..peak_count {
        let intensity =
            intensity_tensor[row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
        if intensity > zero {
            let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
            intensity_max = intensity_max.max(intensity.max(eps).powf(intensity_p));
            mz_max = mz_max.max(mz.max(eps).powf(mz_p));
        }
    }
    intensity_max += eps;
    mz_max += eps;

    let mut product_max = zero;
    for peak in 0..peak_count {
        let intensity =
            intensity_tensor[row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
        if intensity > zero {
            let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
            let product = (intensity.max(eps).powf(intensity_p) / intensity_max)
                * (mz.max(eps).powf(mz_p) / mz_max);
            products[peak] = product;
            product_max = product_max.max(product);
        } else {
            products[peak] = zero;
        }
    }
    for peak in peak_count..max_peaks {
        products[peak] = zero;
    }
    product_max += eps;

    let mut norm_square = zero;
    for peak in 0..peak_count {
        let p = products[peak] / product_max;
        products[peak] = p;
        norm_square += p * p;
    }
    norm_square
}

#[cube]
impl SpectralPairScorer for ModifiedLinearCosineMetric {
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
        #[comptime] _weighted: bool,
    ) -> F {
        modified_linear_cosine_score_rows::<F>(
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
        )
    }
}

/// Modified (precursor-shifted) linear cosine scorer.
///
/// Mirrors `ModifiedLinearCosine::similarity` (CPU) in
/// `src/structs/modified_linear_cosine.rs`: normalize peak products per row,
/// sweep for direct matches, sweep for precursor-shifted matches when the
/// precursors differ by more than the tolerance, dedupe the candidate edges,
/// build a peak-conflict graph (each peak appears in at most two edges), then
/// DP across each path component for the maximum-weight independent set.
#[cube]
pub fn modified_linear_cosine_score_rows<F: Float>(
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
) -> F {
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let invalid = RuntimeCell::<u32>::new(4_294_967_295u32).read();
    let max_peaks_usize = comptime!(max_peaks as usize);
    let candidate_capacity = comptime!(max_peaks_usize * 2usize);
    let dp_capacity = comptime!(max_peaks_usize * 2usize + 1usize);

    let mut similarity = zero;

    if left_peaks <= max_peaks_usize && right_peaks <= max_peaks_usize {
        let mut left_products = Array::<F>::new(max_peaks_usize);
        let mut right_products = Array::<F>::new(max_peaks_usize);

        let mut left_mz_max = zero;
        let mut left_intensity_max = zero;
        for peak in 0..left_peaks {
            left_products[peak] = zero;
            let intensity = left_intensity
                [left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
            if intensity > zero {
                let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
                left_mz_max = left_mz_max.max(mz.powf(mz_p));
                left_intensity_max = left_intensity_max.max(intensity.powf(intensity_p));
            }
        }

        let mut left_product_max = zero;
        for peak in 0..left_peaks {
            let intensity = left_intensity
                [left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
            if intensity > zero {
                let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
                let mut mz_component = mz.powf(mz_p);
                if left_mz_max > zero {
                    mz_component /= left_mz_max;
                }
                let mut intensity_component = intensity.powf(intensity_p);
                if left_intensity_max > zero {
                    intensity_component /= left_intensity_max;
                }
                let product = mz_component * intensity_component;
                left_products[peak] = product;
                left_product_max = left_product_max.max(product);
            }
        }

        let mut left_norm_square = zero;
        for peak in 0..left_peaks {
            if left_product_max > zero {
                left_products[peak] /= left_product_max;
            }
            left_norm_square += left_products[peak] * left_products[peak];
        }

        let mut right_mz_max = zero;
        let mut right_intensity_max = zero;
        for peak in 0..right_peaks {
            right_products[peak] = zero;
            let intensity = right_intensity
                [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
            if intensity > zero {
                let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
                right_mz_max = right_mz_max.max(mz.powf(mz_p));
                right_intensity_max = right_intensity_max.max(intensity.powf(intensity_p));
            }
        }

        let mut right_product_max = zero;
        for peak in 0..right_peaks {
            let intensity = right_intensity
                [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
            if intensity > zero {
                let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
                let mut mz_component = mz.powf(mz_p);
                if right_mz_max > zero {
                    mz_component /= right_mz_max;
                }
                let mut intensity_component = intensity.powf(intensity_p);
                if right_intensity_max > zero {
                    intensity_component /= right_intensity_max;
                }
                let product = mz_component * intensity_component;
                right_products[peak] = product;
                right_product_max = right_product_max.max(product);
            }
        }

        let mut right_norm_square = zero;
        for peak in 0..right_peaks {
            if right_product_max > zero {
                right_products[peak] /= right_product_max;
            }
            right_norm_square += right_products[peak] * right_products[peak];
        }

        if left_norm_square > zero && right_norm_square > zero {
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
                            score += left_products[left_peak] * right_products[right_peak];
                        } else {
                            for path_index in 0..path_len {
                                let edge = path[path_index] as usize;
                                let left_peak = candidate_left[edge] as usize;
                                let right_peak = candidate_right[edge] as usize;
                                benefits[path_index] = (left_products[left_peak]
                                    * right_products[right_peak])
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
                                    score += left_products[left_peak] * right_products[right_peak];
                                    break;
                                }
                                let take = dp[index - 2usize] + benefits[index - 1usize];
                                if take >= dp[index - 1usize] {
                                    let edge = path[index - 1usize] as usize;
                                    let left_peak = candidate_left[edge] as usize;
                                    let right_peak = candidate_right[edge] as usize;
                                    score += left_products[left_peak] * right_products[right_peak];
                                    index -= 2usize;
                                } else {
                                    index -= 1usize;
                                }
                            }
                        }
                    }
                }

                similarity = (score / ((left_norm_square.sqrt() * right_norm_square.sqrt()) + eps))
                    .max(zero)
                    .min(one);
            }
        }
    }
    similarity
}
