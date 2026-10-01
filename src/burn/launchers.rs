use burn_cubecl::cubecl::prelude::*;

use crate::burn::SpectralPairScorer;

/// Paired (1-to-1 batch) launcher: `score[i] = M::score_rows(left[i], right[i])`.
///
/// Per-row tensors for `mz_power`, `intensity_power`, `mz_tolerance` so each
/// pair can use distinct kernel parameters. Callers that want uniform params
/// can broadcast a scalar to `[batch]` before launch.
///
/// `epsilon` is a scalar, it's a numerical-stability constant, not a
/// scientific hyperparameter the user typically varies per pair.
#[cube(launch)]
pub fn paired_forward<F: Float, M: SpectralPairScorer>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_precursor: &Tensor<F>,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_precursor: &Tensor<F>,
    mz_power: &Tensor<F>,
    intensity_power: &Tensor<F>,
    mz_tolerance: &Tensor<F>,
    output: &mut Tensor<F>,
    epsilon: f32,
    #[comptime] max_peaks: u32,
    #[comptime] weighted: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let row = ABSOLUTE_POS;
    let mz_p = mz_power[row * mz_power.stride(0)];
    let intensity_p = intensity_power[row * intensity_power.stride(0)];
    let tolerance = mz_tolerance[row * mz_tolerance.stride(0)];
    let eps = F::cast_from(epsilon);

    output[row] = M::score_rows::<F>(
        left_mz,
        left_intensity,
        left_precursor,
        row,
        right_mz,
        right_intensity,
        right_precursor,
        row,
        mz_p,
        intensity_p,
        tolerance,
        eps,
        max_peaks,
        weighted,
    );
}

/// Cross / all-pairs launcher: `output[i, j] = M::score_rows(left[i], right[j])`.
///
/// 2D launch grid, `ABSOLUTE_POS_X` indexes left rows (M), `ABSOLUTE_POS_Y`
/// indexes right rows (N). Scoring parameters are scalar because broadcasting
/// per-row tensors onto an MxN grid would force an `[M*N]` parameter tensor.
/// Callers wanting per-pair variation should fall back to `paired_forward`.
#[cube(launch)]
pub fn cross_forward<F: Float, M: SpectralPairScorer>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_precursor: &Tensor<F>,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_precursor: &Tensor<F>,
    output: &mut Tensor<F>,
    mz_p_scalar: f32,
    intensity_p_scalar: f32,
    tolerance_scalar: f32,
    epsilon: f32,
    #[comptime] max_peaks: u32,
    #[comptime] weighted: bool,
) {
    let m_dim = output.shape(0);
    let n_dim = output.shape(1);
    let i = ABSOLUTE_POS_X as usize;
    let j = ABSOLUTE_POS_Y as usize;
    if i >= m_dim || j >= n_dim {
        terminate!();
    }

    let mz_p = F::cast_from(mz_p_scalar);
    let intensity_p = F::cast_from(intensity_p_scalar);
    let tolerance = F::cast_from(tolerance_scalar);
    let eps = F::cast_from(epsilon);

    output[i * output.stride(0) + j * output.stride(1)] = M::score_rows::<F>(
        left_mz,
        left_intensity,
        left_precursor,
        i,
        right_mz,
        right_intensity,
        right_precursor,
        j,
        mz_p,
        intensity_p,
        tolerance,
        eps,
        max_peaks,
        weighted,
    );
}

/// Ranking launcher: deterministic LCG sampling of `k` non-self partners per
/// anchor + top-2 reduce. For each anchor in the slice
/// `[batch_start, batch_start + batch_items)` it returns:
/// * `candidate_index[B, k]`: the `k` partner indices (in local slice coords).
/// * `best_candidate_position[B]`: position of the highest-scoring candidate.
/// * `top2_gap[B]`: clamped `best - second_best` score gap.
/// * `candidate_scores[B, k]`: the per-candidate teacher score (column j is
///   the score between anchor `i` and `candidate_index[i, j]`), surfaced so
///   downstream consumers can compute rank-correlation diagnostics without
///   re-running the scorer.
///
/// Sampling uses an XOR-shift seed combined with a coprime stride so each
/// anchor sees `k` distinct partners, skips itself, and the schedule is
/// reproducible from `(seed, batch_start, anchor)`.
#[cube(launch)]
pub fn ranking_forward<F: Float, I: Int, M: SpectralPairScorer>(
    teacher_mz: &Tensor<F>,
    teacher_intensity: &Tensor<F>,
    teacher_precursor: &Tensor<F>,
    candidate_index: &mut Tensor<I>,
    best_candidate_position: &mut Tensor<I>,
    top2_gap: &mut Tensor<F>,
    candidate_scores: &mut Tensor<F>,
    batch_start: u32,
    batch_items: u32,
    candidates_per_anchor: u32,
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    seed: u32,
    epsilon: f32,
    #[comptime] max_peaks: u32,
    #[comptime] weighted: bool,
) {
    if ABSOLUTE_POS >= best_candidate_position.len() {
        terminate!();
    }

    let anchor = ABSOLUTE_POS;
    if anchor >= batch_items as usize || batch_items < 3 {
        terminate!();
    }

    let candidate_count = candidates_per_anchor.max(2).min(batch_items - 1) as usize;
    let teacher_anchor = batch_start as usize + anchor;
    let mz_p = F::cast_from(mz_power);
    let intensity_p = F::cast_from(intensity_power);
    let tolerance = F::cast_from(mz_tolerance);
    let eps = F::cast_from(epsilon);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let mut state = seed ^ (((anchor as u32) + 1u32) * 40503u32) ^ (batch_start >> 16);
    if state == 0u32 {
        state = 0x6d2b_79f5u32;
    }
    let partner_slots = batch_items - 1;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let mut stride = (state % partner_slots) + 1u32;
    let mut coprime = false;
    while !coprime {
        let mut left = stride;
        let mut right = partner_slots;
        while right != 0u32 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        coprime = left == 1u32;
        if !coprime {
            stride += 1u32;
            if stride > partner_slots {
                stride = 1u32;
            }
        }
    }

    let mut best_score = F::new(-1.0_f32);
    let mut second_best_score = F::new(-1.0_f32);
    let mut best_position = 0usize;

    for candidate_position in 0..candidate_count {
        let mut local_partner =
            ((offset + (candidate_position as u32) * stride) % partner_slots) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }

        candidate_index
            [anchor * candidate_index.stride(0) + candidate_position * candidate_index.stride(1)] =
            I::cast_from(local_partner as u32);

        let partner_row = batch_start as usize + local_partner;
        let score = M::score_rows::<F>(
            teacher_mz,
            teacher_intensity,
            teacher_precursor,
            teacher_anchor,
            teacher_mz,
            teacher_intensity,
            teacher_precursor,
            partner_row,
            mz_p,
            intensity_p,
            tolerance,
            eps,
            max_peaks,
            weighted,
        );

        candidate_scores[anchor * candidate_scores.stride(0)
            + candidate_position * candidate_scores.stride(1)] = score;

        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_position = candidate_position;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    best_candidate_position[anchor] = I::cast_from(best_position as u32);
    top2_gap[anchor] = (best_score - second_best_score).max(zero).min(one);
}

#[cfg(test)]
#[allow(dead_code)]
fn paired_forward_monomorphization_smoke<R: burn_cubecl::cubecl::Runtime>() {
    use crate::burn::metrics::LinearCosineMetric;
    let _ = paired_forward::launch::<f32, LinearCosineMetric, R>;
    let _ = cross_forward::launch::<f32, LinearCosineMetric, R>;
    let _ = ranking_forward::launch::<f32, i32, LinearCosineMetric, R>;
}
