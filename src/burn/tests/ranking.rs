//! Ranking-kernel equivalence tests, swept across [`CANONICAL_PARAMETER_POINTS`]
//! (exponent + tolerance regimes). Each iteration replays the LCG sampling
//! schedule on the CPU and asserts identical candidate indices, best
//! position, and top-2 gap.

use crate::burn::{
    EntropyMetric, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric, RankingConfig, SpectralKernelBackend,
    ranking_kernel,
};

use super::fixtures::{
    CANONICAL_PARAMETER_POINTS, DEFAULT_TEST_POINT, ParameterPoint, ReferenceSpectrum,
    TEST_EPSILON, TEST_INTENSITY_POWER, TEST_MAX_PEAKS, TEST_MZ_POWER, TEST_MZ_TOLERANCE,
    assert_ranking_matches_cpu, assert_ranking_row_self_consistency, cpu_linear_cosine,
    cpu_linear_entropy, cpu_modified_linear_cosine, cpu_modified_linear_entropy, reference_spectra,
    reference_spectra_at, spectrum_batch, spectrum_rows,
};

#[cfg(feature = "burn-cuda")]
type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
type TestBackend = burn::backend::Cpu<f32, i32>;

type TestDevice = burn::tensor::Device<TestBackend>;

fn cosine_ranking_config<M: KernelMetric>(point: ParameterPoint) -> RankingConfig<M> {
    M::ranking_config()
        .with_batch_start(1)
        .with_batch_items(10)
        .with_candidates_per_anchor(7)
        .with_mz_power(point.mz_power)
        .with_intensity_power(point.intensity_power)
        .with_mz_tolerance(point.mz_tolerance)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_seed(12_345)
        .with_epsilon(TEST_EPSILON)
}

fn entropy_ranking_config<M: EntropyMetric>(
    point: ParameterPoint,
    weighted: bool,
) -> RankingConfig<M> {
    cosine_ranking_config::<M>(point).with_weighted(weighted)
}

fn run_ranking_test_with<M, F, MakeConfig>(cpu_score: F, tolerance: f32, make_config: MakeConfig)
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
    F: Fn(ParameterPoint, &ReferenceSpectrum, &ReferenceSpectrum) -> f32 + Copy,
    MakeConfig: Fn(ParameterPoint) -> RankingConfig<M>,
{
    let device = TestDevice::default();

    for &point in CANONICAL_PARAMETER_POINTS {
        let spectra = reference_spectra_at(point.mz_tolerance);
        let spectra = &spectra[..12];
        let rows = spectrum_rows(spectra);

        let config = make_config(point);

        let teacher = spectrum_batch::<TestBackend>(&rows, &device);
        let output = ranking_kernel::<TestBackend, M>(teacher, config);

        let candidate_index = output
            .candidate_index
            .into_data()
            .to_vec::<i32>()
            .expect("candidate indices should be i32");
        let best_position = output
            .best_position
            .into_data()
            .to_vec::<i32>()
            .expect("best positions should be i32");
        let top2_gap = output
            .top2_gap
            .into_data()
            .to_vec::<f32>()
            .expect("top-2 gaps should be f32");
        let candidate_scores_shape = output.candidate_scores.dims();
        let candidate_scores = output
            .candidate_scores
            .into_data()
            .to_vec::<f32>()
            .expect("candidate scores should be f32");
        assert_eq!(
            candidate_scores_shape,
            [
                config.batch_items(),
                config.effective_candidates_per_anchor()
            ],
            "{}: candidate_scores must be [batch_items, k]",
            M::NAME,
        );

        let candidate_count = config.effective_candidates_per_anchor();
        for anchor in 0..config.batch_items() {
            let expected = ranking_reference(spectra, anchor, &config, |left, right| {
                cpu_score(point, left, right)
            });
            let start = anchor * candidate_count;
            let actual_candidates = &candidate_index[start..start + candidate_count];
            let actual_scores = &candidate_scores[start..start + candidate_count];
            assert_eq!(
                actual_candidates,
                expected.candidate_indices.as_slice(),
                "anchor {anchor} ({}) at {point:?}: candidate indices diverged",
                M::NAME,
            );
            assert_eq!(
                best_position[anchor] as usize,
                expected.best_candidate_position,
                "anchor {anchor} ({}) at {point:?}: best position diverged",
                M::NAME,
            );
            assert!(
                (top2_gap[anchor] - expected.top2_gap).abs() < tolerance,
                "anchor {anchor} ({}) at {point:?}: gpu={} cpu={}",
                M::NAME,
                top2_gap[anchor],
                expected.top2_gap,
            );
            assert!(
                !actual_candidates.contains(&(anchor as i32)),
                "anchor {anchor} ({}) at {point:?}: self-pairing in candidates",
                M::NAME,
            );
            for (left, left_value) in actual_candidates.iter().enumerate() {
                for right_value in actual_candidates.iter().skip(left + 1) {
                    assert_ne!(
                        left_value,
                        right_value,
                        "anchor {anchor} ({}) at {point:?}: duplicate candidate {left_value}",
                        M::NAME,
                    );
                }
            }

            // GPU-only self-consistency: finite + range + argmax-score
            // == top-1 + gap == top1-top2. Catches argmax wiring / gap
            // reduction regressions independently of the CPU reference.
            let context = alloc::format!("{} at {point:?}", M::NAME);
            assert_ranking_row_self_consistency(
                &context,
                anchor,
                actual_scores,
                best_position[anchor] as usize,
                top2_gap[anchor],
                tolerance,
            );

            // Per-candidate score equivalence against the LCG-replayed CPU
            // reference. Catches divergence inside the score reduction even
            // when the argmax happens to land on the right row.
            for (column, (&gpu, &cpu)) in actual_scores
                .iter()
                .zip(expected.candidate_scores.iter())
                .enumerate()
            {
                assert!(
                    (gpu - cpu).abs() < tolerance,
                    "anchor {anchor} col {column} ({}) at {point:?}: gpu={gpu} cpu={cpu}",
                    M::NAME,
                );
            }
        }
    }
}

struct RankingReference {
    candidate_indices: Vec<i32>,
    candidate_scores: Vec<f32>,
    best_candidate_position: usize,
    top2_gap: f32,
}

/// CPU reference: replicates the LCG sampling schedule bit-for-bit and scores
/// each candidate using `cpu_score` (already bound to the current parameter
/// point by the caller).
fn ranking_reference<M: KernelMetric>(
    spectra: &[(&'static str, ReferenceSpectrum)],
    anchor: usize,
    config: &RankingConfig<M>,
    cpu_score: impl Fn(&ReferenceSpectrum, &ReferenceSpectrum) -> f32,
) -> RankingReference {
    let batch_start = config.batch_start();
    let batch_items = config.batch_items();
    assert!(batch_start + batch_items <= spectra.len());

    let mut state =
        config.seed() as u32 ^ (((anchor as u32) + 1) * 40503) ^ ((batch_start as u32) >> 16);
    if state == 0 {
        state = 0x6d2b_79f5;
    }
    let partner_slots = (batch_items - 1) as u32;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let mut stride = (state % partner_slots) + 1;
    while gcd(stride, partner_slots) != 1 {
        stride += 1;
        if stride > partner_slots {
            stride = 1;
        }
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;
    let mut best_candidate_position = 0usize;
    let candidates = config.effective_candidates_per_anchor();
    let anchor_spectrum = &spectra[batch_start + anchor].1;
    let mut candidate_indices = Vec::with_capacity(candidates);
    let mut candidate_scores = Vec::with_capacity(candidates);

    for candidate_position in 0..candidates {
        let mut local_partner =
            ((offset + (candidate_position as u32) * stride) % partner_slots) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }
        candidate_indices.push(local_partner as i32);

        let partner_spectrum = &spectra[batch_start + local_partner].1;
        let score = cpu_score(anchor_spectrum, partner_spectrum);
        candidate_scores.push(score);
        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_candidate_position = candidate_position;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    RankingReference {
        candidate_indices,
        candidate_scores,
        best_candidate_position,
        top2_gap: (best_score - second_best_score).clamp(0.0, 1.0),
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[test]
fn ranking_matches_cpu_linear_cosine() {
    run_ranking_test_with::<LinearCosineMetric, _, _>(
        cpu_linear_cosine,
        1.0e-4,
        cosine_ranking_config::<LinearCosineMetric>,
    );
}

#[test]
fn ranking_matches_cpu_modified_linear_cosine() {
    run_ranking_test_with::<ModifiedLinearCosineMetric, _, _>(
        cpu_modified_linear_cosine,
        2.0e-4,
        cosine_ranking_config::<ModifiedLinearCosineMetric>,
    );
}

#[test]
fn ranking_matches_cpu_linear_entropy_unweighted() {
    run_ranking_test_with::<LinearEntropyMetric, _, _>(
        |p, l, r| cpu_linear_entropy(p, false, l, r),
        1.0e-4,
        |p| entropy_ranking_config::<LinearEntropyMetric>(p, false),
    );
}

#[test]
fn ranking_matches_cpu_linear_entropy_weighted() {
    run_ranking_test_with::<LinearEntropyMetric, _, _>(
        |p, l, r| cpu_linear_entropy(p, true, l, r),
        2.0e-4,
        |p| entropy_ranking_config::<LinearEntropyMetric>(p, true),
    );
}

#[test]
fn ranking_matches_cpu_modified_linear_entropy_unweighted() {
    run_ranking_test_with::<ModifiedLinearEntropyMetric, _, _>(
        |p, l, r| cpu_modified_linear_entropy(p, false, l, r),
        2.0e-4,
        |p| entropy_ranking_config::<ModifiedLinearEntropyMetric>(p, false),
    );
}

#[test]
fn ranking_matches_cpu_modified_linear_entropy_weighted() {
    run_ranking_test_with::<ModifiedLinearEntropyMetric, _, _>(
        |p, l, r| cpu_modified_linear_entropy(p, true, l, r),
        2.0e-4,
        |p| entropy_ranking_config::<ModifiedLinearEntropyMetric>(p, true),
    );
}

/// Half-precision smoke test on the CUDA runtime. We instantiate the
/// ranking kernel under `Cuda<half::f16, i32>` to exercise the
/// type-genericism of the `F: FloatElement` plumbing across the kernel,
/// the CubeBackend impl, the `[N, k]` write into `candidate_scores`, and
/// the readback path. f16 dot products lose ~3 decimal digits versus
/// f32, so we relax the equivalence tolerance accordingly and only
/// assert: shape, finiteness, value in `[0, 1]` (with f16 slack), and
/// argmax/gap relationships that should hold regardless of precision.
/// The `cargo doc` API claims half precision is available, this test
/// proves the kernel surfaces it cleanly.
#[cfg(feature = "burn-cuda")]
#[test]
fn ranking_kernel_runs_on_f16() {
    use crate::burn::api::SpectrumBatch;
    use burn::tensor::Tensor as BurnTensor;
    use burn::tensor::TensorData;

    type F16Backend = burn::backend::Cuda<half::f16, i32>;
    type F16Device = burn::tensor::Device<F16Backend>;

    let device = F16Device::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..8];
    let rows = spectrum_rows(spectra);
    let row_count = rows.precursor.len();

    let mz_data: alloc::vec::Vec<half::f16> =
        rows.mz.iter().copied().map(half::f16::from_f32).collect();
    let intensity_data: alloc::vec::Vec<half::f16> = rows
        .intensity
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let precursor_data: alloc::vec::Vec<half::f16> = rows
        .precursor
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();

    let teacher = SpectrumBatch::<F16Backend>::new(
        BurnTensor::<F16Backend, 2>::from_data(
            TensorData::new(mz_data, [row_count, rows.peak_width]),
            &device,
        ),
        BurnTensor::<F16Backend, 2>::from_data(
            TensorData::new(intensity_data, [row_count, rows.peak_width]),
            &device,
        ),
        BurnTensor::<F16Backend, 1>::from_data(
            TensorData::new(precursor_data, [row_count]),
            &device,
        ),
    );

    let config = LinearCosineMetric::ranking_config()
        .with_batch_start(0)
        .with_batch_items(row_count)
        .with_candidates_per_anchor(3)
        .with_mz_power(TEST_MZ_POWER)
        .with_intensity_power(TEST_INTENSITY_POWER)
        .with_mz_tolerance(TEST_MZ_TOLERANCE)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_seed(42)
        .with_epsilon(TEST_EPSILON);

    let output = ranking_kernel::<F16Backend, LinearCosineMetric>(teacher, config);

    let candidate_count = config.effective_candidates_per_anchor();
    assert_eq!(output.candidate_index.dims(), [row_count, candidate_count]);
    assert_eq!(output.best_position.dims(), [row_count]);
    assert_eq!(output.top2_gap.dims(), [row_count]);
    assert_eq!(output.candidate_scores.dims(), [row_count, candidate_count]);

    // `Cuda<half::f16, i32>` may store intermediate float tensors in F32
    // (`Flex32`) for compute stability, so explicitly convert the readback
    // to F32 before draining. We assert only the precision-tolerant
    // invariants (finiteness, range, ordering), not bit equality.
    let candidate_scores: alloc::vec::Vec<f32> = output
        .candidate_scores
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("candidate scores should round-trip as f32 after convert");
    let top2_gap: alloc::vec::Vec<f32> = output
        .top2_gap
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("top2_gap should round-trip as f32 after convert");
    let best_position = output
        .best_position
        .into_data()
        .to_vec::<i32>()
        .expect("best position should be i32");

    let f16_slack = 5.0e-2;
    for (anchor, row_scores) in candidate_scores.chunks(candidate_count).enumerate() {
        assert_ranking_row_self_consistency(
            "f16 ranking",
            anchor,
            row_scores,
            best_position[anchor] as usize,
            top2_gap[anchor],
            f16_slack,
        );
    }
}

/// Exercise the kernel's lower boundary: `batch_items = 3` (the assert
/// floor) and `candidates_per_anchor = 2` (also the floor, after the
/// kernel's `max(2).min(batch_items - 1)` clamp). With three anchors and
/// two non-self partners each, the coprime-stride LCG has a single
/// possible schedule per anchor (the two other rows in some order), so
/// this is the test most likely to surface off-by-one bugs in the stride
/// math, the `local_partner >= anchor` skip, or the score-write indexing.
#[test]
fn ranking_minimum_batch() {
    let device = TestDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..3];

    let config = LinearCosineMetric::ranking_config()
        .with_batch_start(0)
        .with_batch_items(3)
        .with_candidates_per_anchor(2)
        .with_mz_power(TEST_MZ_POWER)
        .with_intensity_power(TEST_INTENSITY_POWER)
        .with_mz_tolerance(TEST_MZ_TOLERANCE)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_seed(7)
        .with_epsilon(TEST_EPSILON);

    assert_eq!(
        config.effective_candidates_per_anchor(),
        2,
        "effective k must clamp to batch_items-1 at the minimum boundary"
    );

    assert_ranking_matches_cpu::<TestBackend, LinearCosineMetric>(
        spectra,
        &device,
        config,
        |l, r| cpu_linear_cosine(DEFAULT_TEST_POINT, l, r),
        1.0e-4,
    );
}
