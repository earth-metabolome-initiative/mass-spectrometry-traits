//! Paired (1-to-1 batch) equivalence tests for the GPU kernels.
//!
//! Each `#[test]` runs the full 74x74 reference-spectrum sweep across every
//! entry in [`CANONICAL_PARAMETER_POINTS`], five exponent regimes at the
//! baseline tolerance plus two tolerance regimes at the baseline exponents.
//! One panic per parameter point that exceeds tolerance, with the failing
//! `(mz_power, intensity_power, mz_tolerance)` in the message.

use crate::burn::{
    EntropyMetric, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric, PairedConfig, SpectralKernelBackend,
    paired_kernel,
};

use super::fixtures::{
    CANONICAL_PARAMETER_POINTS, PAIR_CHUNK_SIZE, ParameterPoint, ReferenceSpectrum, TEST_EPSILON,
    TEST_MAX_PEAKS, all_pair_indices, assert_all_pair_scores_match, cpu_linear_cosine,
    cpu_linear_entropy, cpu_modified_linear_cosine, cpu_modified_linear_entropy, pair_batches,
    pair_rows, pairwise_params_constant, reference_spectra_at,
};

// Backend selection: prefer CUDA when both features are enabled (it's faster
// on dev machines with GPUs). Fall back to the MLIR-based CPU runtime
// (`burn-cpu`) so the equivalence tests can run in CI without a GPU.
#[cfg(feature = "burn-cuda")]
type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
type TestBackend = burn::backend::Cpu<f32, i32>;

type TestDevice = burn::tensor::Device<TestBackend>;

/// Score every reference-spectrum pair on the GPU under metric `M`, with a
/// pre-built [`PairedConfig<M>`]. Shared by the cosine and entropy harnesses
/// since the only thing they differ on is whether the config has
/// `with_weighted` applied.
fn run_paired_test_with_config<M, F>(cpu_score: F, tolerance: f32, config: PairedConfig<M>)
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
    F: Fn(ParameterPoint, &ReferenceSpectrum, &ReferenceSpectrum) -> f32 + Copy,
{
    let device = TestDevice::default();

    for &point in CANONICAL_PARAMETER_POINTS {
        let spectra = reference_spectra_at(point.mz_tolerance);
        let pair_indices = all_pair_indices(spectra.len());

        for chunk in pair_indices.chunks(PAIR_CHUNK_SIZE) {
            let pairs = pair_rows(&spectra, chunk);
            let row_count = pairs.indices.len();
            let (left, right) = pair_batches::<TestBackend>(&pairs, &device);
            let params = pairwise_params_constant::<TestBackend>(
                row_count,
                point.mz_power,
                point.intensity_power,
                point.mz_tolerance,
                &device,
            );

            let scores = paired_kernel::<TestBackend, M>(left, right, params, config)
                .into_data()
                .to_vec::<f32>()
                .expect("kernel output should be f32");

            assert_all_pair_scores_match(
                &spectra,
                &pairs.indices,
                &scores,
                tolerance,
                point.mz_power,
                point.intensity_power,
                |left, right| cpu_score(point, left, right),
            );
        }
    }
}

fn run_paired_test_cosine<M, F>(cpu_score: F, tolerance: f32)
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
    F: Fn(ParameterPoint, &ReferenceSpectrum, &ReferenceSpectrum) -> f32 + Copy,
{
    let config = M::paired_config()
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON);
    run_paired_test_with_config::<M, F>(cpu_score, tolerance, config);
}

fn run_paired_test_entropy<M, F>(cpu_score: F, tolerance: f32, weighted: bool)
where
    M: EntropyMetric,
    TestBackend: SpectralKernelBackend<M>,
    F: Fn(ParameterPoint, &ReferenceSpectrum, &ReferenceSpectrum) -> f32 + Copy,
{
    let config = M::paired_config()
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON)
        .with_weighted(weighted);
    run_paired_test_with_config::<M, F>(cpu_score, tolerance, config);
}

#[test]
fn paired_matches_cpu_linear_cosine() {
    run_paired_test_cosine::<LinearCosineMetric, _>(cpu_linear_cosine, 1.0e-4);
}

#[test]
fn paired_matches_cpu_modified_linear_cosine() {
    run_paired_test_cosine::<ModifiedLinearCosineMetric, _>(cpu_modified_linear_cosine, 2.0e-4);
}

#[test]
fn paired_matches_cpu_linear_entropy_unweighted() {
    run_paired_test_entropy::<LinearEntropyMetric, _>(
        |p, l, r| cpu_linear_entropy(p, false, l, r),
        1.0e-4,
        false,
    );
}

#[test]
fn paired_matches_cpu_linear_entropy_weighted() {
    run_paired_test_entropy::<LinearEntropyMetric, _>(
        |p, l, r| cpu_linear_entropy(p, true, l, r),
        2.0e-4,
        true,
    );
}

#[test]
fn paired_matches_cpu_modified_linear_entropy_unweighted() {
    run_paired_test_entropy::<ModifiedLinearEntropyMetric, _>(
        |p, l, r| cpu_modified_linear_entropy(p, false, l, r),
        2.0e-4,
        false,
    );
}

#[test]
fn paired_matches_cpu_modified_linear_entropy_weighted() {
    run_paired_test_entropy::<ModifiedLinearEntropyMetric, _>(
        |p, l, r| cpu_modified_linear_entropy(p, true, l, r),
        2.0e-4,
        true,
    );
}
