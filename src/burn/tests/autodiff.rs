//! Confirms the `Autodiff<Cuda<f32, i32>>` wrapper produces identical scores
//! to the raw `Cuda<f32, i32>` backend across all three kernel shapes.
//!
//! The kernels are non-differentiable (see `src/burn/autodiff.rs`), so the
//! autodiff wrapper only attaches `NoGradientBackward` stubs to the graph ,
//! no actual gradient computation happens. This test guards against future
//! regressions in the wrapper accidentally mutating forward-pass values.

#![cfg(feature = "burn-autodiff")]

use burn::backend::cuda::CudaDevice;
use burn::backend::{Autodiff, Cuda};

use crate::burn::api::SpectrumBatch;
use crate::burn::{
    KernelMetric, LinearCosineMetric, LinearEntropyMetric, ModifiedLinearCosineMetric,
    ModifiedLinearEntropyMetric, cross_kernel, paired_kernel, ranking_kernel,
};

use super::fixtures::{
    DEFAULT_TEST_POINT, PAIR_CHUNK_SIZE, TEST_EPSILON, TEST_INTENSITY_POWER, TEST_MAX_PEAKS,
    TEST_MZ_POWER, TEST_MZ_TOLERANCE, all_pair_indices, assert_ranking_matches_cpu,
    cpu_linear_cosine, cpu_linear_entropy, cpu_modified_linear_cosine, cpu_modified_linear_entropy,
    default_entropy_ranking_config, default_ranking_config, pair_batches, pair_rows,
    pairwise_params_constant, reference_spectra, spectrum_batch, spectrum_rows,
};

type RawBackend = Cuda<f32, i32>;
type AutodiffBackend = Autodiff<RawBackend>;

#[test]
fn autodiff_paired_matches_raw_backend() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    // Smaller slice, this is a sanity check, not a sweep.
    let spectra = &spectra[..16];
    let indices = all_pair_indices(spectra.len());

    let config = LinearCosineMetric::paired_config()
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON);

    for chunk in indices.chunks(PAIR_CHUNK_SIZE) {
        let pairs = pair_rows(spectra, chunk);
        let row_count = pairs.indices.len();
        let (left, right) = pair_batches::<AutodiffBackend>(&pairs, &device);
        let params = pairwise_params_constant::<AutodiffBackend>(
            row_count,
            TEST_MZ_POWER,
            TEST_INTENSITY_POWER,
            TEST_MZ_TOLERANCE,
            &device,
        );

        let scores =
            paired_kernel::<AutodiffBackend, LinearCosineMetric>(left, right, params, config)
                .into_data()
                .to_vec::<f32>()
                .expect("kernel output should be f32");

        for (row, &(left_index, right_index)) in pairs.indices.iter().enumerate() {
            let (_, left) = &spectra[left_index];
            let (_, right) = &spectra[right_index];
            let cpu = cpu_linear_cosine(DEFAULT_TEST_POINT, left, right);
            assert!(
                (scores[row] - cpu).abs() < 1.0e-4,
                "autodiff paired diverged at row {row}: gpu={} cpu={}",
                scores[row],
                cpu,
            );
        }
    }
}

#[test]
fn autodiff_cross_matches_raw_backend() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let left = &spectra[..6];
    let right = &spectra[6..12];

    let left_rows = spectrum_rows(left);
    let right_rows = spectrum_rows(right);
    let left_batch = spectrum_batch::<AutodiffBackend>(&left_rows, &device);
    let right_batch = spectrum_batch::<AutodiffBackend>(&right_rows, &device);

    let config = LinearCosineMetric::cross_config()
        .with_mz_power(TEST_MZ_POWER)
        .with_intensity_power(TEST_INTENSITY_POWER)
        .with_mz_tolerance(TEST_MZ_TOLERANCE)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON);

    let scores =
        cross_kernel::<AutodiffBackend, LinearCosineMetric>(left_batch, right_batch, config)
            .into_data()
            .to_vec::<f32>()
            .expect("cross kernel output should be f32");

    assert_eq!(scores.len(), left.len() * right.len());
    for (i, (_, left_spectrum)) in left.iter().enumerate() {
        for (j, (_, right_spectrum)) in right.iter().enumerate() {
            let gpu_score = scores[i * right.len() + j];
            let cpu = cpu_linear_cosine(DEFAULT_TEST_POINT, left_spectrum, right_spectrum);
            assert!(
                (gpu_score - cpu).abs() < 1.0e-4,
                "autodiff cross diverged at ({i}, {j}): gpu={gpu_score} cpu={cpu}",
            );
        }
    }
}

#[test]
fn autodiff_ranking_matches_cpu_linear_cosine() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    assert_ranking_matches_cpu::<AutodiffBackend, LinearCosineMetric>(
        spectra,
        &device,
        default_ranking_config::<LinearCosineMetric>(),
        |l, r| cpu_linear_cosine(DEFAULT_TEST_POINT, l, r),
        1.0e-4,
    );
}

#[test]
fn autodiff_ranking_matches_cpu_modified_linear_cosine() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    assert_ranking_matches_cpu::<AutodiffBackend, ModifiedLinearCosineMetric>(
        spectra,
        &device,
        default_ranking_config::<ModifiedLinearCosineMetric>(),
        |l, r| cpu_modified_linear_cosine(DEFAULT_TEST_POINT, l, r),
        2.0e-4,
    );
}

#[test]
fn autodiff_ranking_matches_cpu_linear_entropy_unweighted() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    assert_ranking_matches_cpu::<AutodiffBackend, LinearEntropyMetric>(
        spectra,
        &device,
        default_entropy_ranking_config::<LinearEntropyMetric>(false),
        |l, r| cpu_linear_entropy(DEFAULT_TEST_POINT, false, l, r),
        2.0e-4,
    );
}

#[test]
fn autodiff_ranking_matches_cpu_linear_entropy_weighted() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    assert_ranking_matches_cpu::<AutodiffBackend, LinearEntropyMetric>(
        spectra,
        &device,
        default_entropy_ranking_config::<LinearEntropyMetric>(true),
        |l, r| cpu_linear_entropy(DEFAULT_TEST_POINT, true, l, r),
        2.0e-4,
    );
}

#[test]
fn autodiff_ranking_matches_cpu_modified_linear_entropy_unweighted() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    assert_ranking_matches_cpu::<AutodiffBackend, ModifiedLinearEntropyMetric>(
        spectra,
        &device,
        default_entropy_ranking_config::<ModifiedLinearEntropyMetric>(false),
        |l, r| cpu_modified_linear_entropy(DEFAULT_TEST_POINT, false, l, r),
        2.0e-4,
    );
}

#[test]
fn autodiff_ranking_matches_cpu_modified_linear_entropy_weighted() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    assert_ranking_matches_cpu::<AutodiffBackend, ModifiedLinearEntropyMetric>(
        spectra,
        &device,
        default_entropy_ranking_config::<ModifiedLinearEntropyMetric>(true),
        |l, r| cpu_modified_linear_entropy(DEFAULT_TEST_POINT, true, l, r),
        2.0e-4,
    );
}

/// Composition test: `Autodiff<Fusion<Cuda<...>>>`. Both wrapper layers are
/// in the training-loop hot path simultaneously (fusion batches the
/// surrounding tensor ops, autodiff carries the graph forward), but until
/// now we only had per-layer tests in isolation. This guards against
/// regressions where the layers compose in a way that breaks forward-pass
/// value correctness, e.g. the fusion replay handing the wrong tensor IRs
/// back through the autodiff wrapper after the recent four-output ranking
/// custom-op change.
#[cfg(feature = "burn-fusion")]
#[test]
fn autodiff_over_fusion_ranking_matches_cpu_linear_cosine() {
    use burn_cubecl::CubeBackend;
    use burn_cubecl::cubecl::cuda::CudaRuntime;
    use burn_fusion::Fusion;

    type RawCube = CubeBackend<CudaRuntime, f32, i32, u8>;
    type Stacked = burn::backend::Autodiff<Fusion<RawCube>>;

    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    assert_ranking_matches_cpu::<Stacked, LinearCosineMetric>(
        spectra,
        &device,
        default_ranking_config::<LinearCosineMetric>(),
        |l, r| cpu_linear_cosine(DEFAULT_TEST_POINT, l, r),
        1.0e-4,
    );
}

/// The ranking kernel is non-differentiable by design (m/z matching is a
/// discrete tolerance lookup, and the modified variants run DP on a discrete
/// conflict graph). The autodiff wrapper installs `NoGradientBackward` stubs
/// for `top2_gap` and `candidate_scores` so the kernel composes inside an
/// `Autodiff<...>` context without panicking, but no gradient must reach the
/// teacher tensors. This test drives a scalar loss off both float outputs,
/// calls `.backward()`, and asserts that none of `mz`, `intensity`, or
/// `precursor` end up with a gradient entry.
#[test]
fn autodiff_ranking_propagates_no_gradient_to_teacher() {
    use burn::tensor::{Tensor as BurnTensor, TensorData};

    use super::fixtures::spectrum_rows;

    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    let rows = spectrum_rows(spectra);
    let row_count = rows.precursor.len();

    let mz = BurnTensor::<AutodiffBackend, 2>::from_data(
        TensorData::new(rows.mz.clone(), [row_count, rows.peak_width]),
        &device,
    )
    .require_grad();
    let intensity = BurnTensor::<AutodiffBackend, 2>::from_data(
        TensorData::new(rows.intensity.clone(), [row_count, rows.peak_width]),
        &device,
    )
    .require_grad();
    let precursor = BurnTensor::<AutodiffBackend, 1>::from_data(
        TensorData::new(rows.precursor.clone(), [row_count]),
        &device,
    )
    .require_grad();

    let teacher =
        SpectrumBatch::<AutodiffBackend>::new(mz.clone(), intensity.clone(), precursor.clone());

    let output = ranking_kernel::<AutodiffBackend, LinearCosineMetric>(
        teacher,
        default_ranking_config::<LinearCosineMetric>(),
    );

    let loss = output.candidate_scores.mean() + output.top2_gap.mean();
    let gradients = loss.backward();

    assert!(
        mz.grad(&gradients).is_none(),
        "ranking_kernel must not register a gradient for teacher.mz \
         under Autodiff (the kernel is non-differentiable by design)",
    );
    assert!(
        intensity.grad(&gradients).is_none(),
        "ranking_kernel must not register a gradient for teacher.intensity",
    );
    assert!(
        precursor.grad(&gradients).is_none(),
        "ranking_kernel must not register a gradient for teacher.precursor",
    );
}
