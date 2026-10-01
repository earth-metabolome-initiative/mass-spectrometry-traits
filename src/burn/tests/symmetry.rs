//! Structural invariants of every metric:
//!
//! * **Symmetry**: `sim(a, b) == sim(b, a)` to within score tolerance, for
//!   every metric x every pair in a small reference batch. Catches a class of
//!   bugs (left/right index swap, asymmetric eps placement) that the existing
//!   directed-pair tests can't detect.
//! * **Self-similarity**: `sim(a, a) ~ 1.0` for every reference spectrum
//!   under every metric.

use crate::burn::{
    KernelMetric, LinearCosineMetric, LinearEntropyMetric, ModifiedLinearCosineMetric,
    ModifiedLinearEntropyMetric, PairedConfig, SpectralKernelBackend, paired_kernel,
};

use super::fixtures::{
    PairRows, TEST_INTENSITY_POWER, TEST_MZ_POWER, TEST_MZ_TOLERANCE, cosine_paired_config,
    entropy_paired_config, pair_batches, pair_rows, pairwise_params_constant, reference_spectra,
};

#[cfg(feature = "burn-cuda")]
type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
type TestBackend = burn::backend::Cpu<f32, i32>;

type TestDevice = burn::tensor::Device<TestBackend>;

const SYMMETRY_BATCH: usize = 8;
const SCORE_TOLERANCE: f32 = 2.0e-4;
const SELF_SIM_THRESHOLD: f32 = 0.999;

fn paired_scores_with_config<M>(pairs: &PairRows, config: PairedConfig<M>) -> Vec<f32>
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    let device = TestDevice::default();
    let row_count = pairs.indices.len();
    let (left, right) = pair_batches::<TestBackend>(pairs, &device);
    let params = pairwise_params_constant::<TestBackend>(
        row_count,
        TEST_MZ_POWER,
        TEST_INTENSITY_POWER,
        TEST_MZ_TOLERANCE,
        &device,
    );

    paired_kernel::<TestBackend, M>(left, right, params, config)
        .into_data()
        .to_vec::<f32>()
        .expect("kernel output should be f32")
}

/// Assert `sim(a, b) ~ sim(b, a)` for every pair in the small reference
/// batch, under the given metric.
fn assert_symmetric<M>(metric_name: &str, config: PairedConfig<M>)
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    let spectra = reference_spectra();
    let spectra = &spectra[..SYMMETRY_BATCH];

    let forward_indices: Vec<(usize, usize)> = (0..spectra.len())
        .flat_map(|i| (0..spectra.len()).map(move |j| (i, j)))
        .collect();
    let reverse_indices: Vec<(usize, usize)> =
        forward_indices.iter().map(|&(i, j)| (j, i)).collect();

    let forward = pair_rows(spectra, &forward_indices);
    let reverse = pair_rows(spectra, &reverse_indices);

    let forward_scores = paired_scores_with_config::<M>(&forward, config);
    let reverse_scores = paired_scores_with_config::<M>(&reverse, config);

    for (row, &(i, j)) in forward_indices.iter().enumerate() {
        let ab = forward_scores[row];
        let ba = reverse_scores[row];
        assert!(
            (ab - ba).abs() < SCORE_TOLERANCE,
            "{metric_name}: sim({}, {}) = {ab} vs sim({}, {}) = {ba} \
             (delta={})",
            spectra[i].0,
            spectra[j].0,
            spectra[j].0,
            spectra[i].0,
            (ab - ba).abs()
        );
    }
}

/// Assert `sim(a, a) >= SELF_SIM_THRESHOLD` for every spectrum in the small
/// reference batch, under the given metric.
fn assert_self_similarity<M>(metric_name: &str, config: PairedConfig<M>, threshold: f32)
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    let spectra = reference_spectra();
    let spectra = &spectra[..SYMMETRY_BATCH];

    let self_indices: Vec<(usize, usize)> = (0..spectra.len()).map(|i| (i, i)).collect();
    let self_pairs = pair_rows(spectra, &self_indices);
    let scores = paired_scores_with_config::<M>(&self_pairs, config);

    for (row, &(i, _)) in self_indices.iter().enumerate() {
        let score = scores[row];
        assert!(
            (0.0..=1.0).contains(&score),
            "{metric_name}: self-sim({}) = {score} outside [0, 1]",
            spectra[i].0,
        );
        assert!(
            score >= threshold,
            "{metric_name}: self-sim({}) = {score} below threshold {threshold}",
            spectra[i].0,
        );
    }
}

#[test]
fn symmetric_linear_cosine() {
    assert_symmetric::<LinearCosineMetric>("LinearCosine", cosine_paired_config());
}

#[test]
fn symmetric_modified_linear_cosine() {
    assert_symmetric::<ModifiedLinearCosineMetric>("ModifiedLinearCosine", cosine_paired_config());
}

#[test]
fn symmetric_linear_entropy_unweighted() {
    assert_symmetric::<LinearEntropyMetric>(
        "LinearEntropy(unweighted)",
        entropy_paired_config(false),
    );
}

#[test]
fn symmetric_linear_entropy_weighted() {
    assert_symmetric::<LinearEntropyMetric>("LinearEntropy(weighted)", entropy_paired_config(true));
}

#[test]
fn symmetric_modified_linear_entropy_unweighted() {
    assert_symmetric::<ModifiedLinearEntropyMetric>(
        "ModifiedLinearEntropy(unweighted)",
        entropy_paired_config(false),
    );
}

#[test]
fn symmetric_modified_linear_entropy_weighted() {
    assert_symmetric::<ModifiedLinearEntropyMetric>(
        "ModifiedLinearEntropy(weighted)",
        entropy_paired_config(true),
    );
}

#[test]
fn self_similarity_linear_cosine() {
    assert_self_similarity::<LinearCosineMetric>(
        "LinearCosine",
        cosine_paired_config(),
        SELF_SIM_THRESHOLD,
    );
}

#[test]
fn self_similarity_modified_linear_cosine() {
    assert_self_similarity::<ModifiedLinearCosineMetric>(
        "ModifiedLinearCosine",
        cosine_paired_config(),
        SELF_SIM_THRESHOLD,
    );
}

#[test]
fn self_similarity_linear_entropy_unweighted() {
    assert_self_similarity::<LinearEntropyMetric>(
        "LinearEntropy(unweighted)",
        entropy_paired_config(false),
        SELF_SIM_THRESHOLD,
    );
}

#[test]
fn self_similarity_linear_entropy_weighted() {
    assert_self_similarity::<LinearEntropyMetric>(
        "LinearEntropy(weighted)",
        entropy_paired_config(true),
        SELF_SIM_THRESHOLD,
    );
}

#[test]
fn self_similarity_modified_linear_entropy_unweighted() {
    assert_self_similarity::<ModifiedLinearEntropyMetric>(
        "ModifiedLinearEntropy(unweighted)",
        entropy_paired_config(false),
        SELF_SIM_THRESHOLD,
    );
}

#[test]
fn self_similarity_modified_linear_entropy_weighted() {
    assert_self_similarity::<ModifiedLinearEntropyMetric>(
        "ModifiedLinearEntropy(weighted)",
        entropy_paired_config(true),
        SELF_SIM_THRESHOLD,
    );
}
