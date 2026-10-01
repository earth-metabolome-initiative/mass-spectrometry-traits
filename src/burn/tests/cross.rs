//! Cross / all-pairs (`[M, P] x [N, P] -> [M, N]`) equivalence tests, swept
//! across [`CANONICAL_PARAMETER_POINTS`] (exponent + tolerance regimes).

use crate::burn::{
    CrossConfig, EntropyMetric, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric, SpectralKernelBackend, cross_kernel,
};

use super::fixtures::{
    CANONICAL_PARAMETER_POINTS, ParameterPoint, ReferenceSpectrum, TEST_EPSILON, TEST_MAX_PEAKS,
    cpu_linear_cosine, cpu_linear_entropy, cpu_modified_linear_cosine, cpu_modified_linear_entropy,
    reference_spectra_at, spectrum_batch, spectrum_rows,
};

#[cfg(feature = "burn-cuda")]
type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
type TestBackend = burn::backend::Cpu<f32, i32>;

type TestDevice = burn::tensor::Device<TestBackend>;

const CROSS_M: usize = 8;
const CROSS_N: usize = 12;

fn run_cross_test_with<M, F, MakeConfig>(cpu_score: F, tolerance: f32, make_config: MakeConfig)
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
    F: Fn(ParameterPoint, &ReferenceSpectrum, &ReferenceSpectrum) -> f32 + Copy,
    MakeConfig: Fn(ParameterPoint) -> CrossConfig<M>,
{
    let device = TestDevice::default();

    for &point in CANONICAL_PARAMETER_POINTS {
        let spectra = reference_spectra_at(point.mz_tolerance);
        assert!(spectra.len() >= CROSS_M + CROSS_N);

        let left = &spectra[..CROSS_M];
        let right = &spectra[CROSS_M..CROSS_M + CROSS_N];

        let left_rows = spectrum_rows(left);
        let right_rows = spectrum_rows(right);
        let left_batch = spectrum_batch::<TestBackend>(&left_rows, &device);
        let right_batch = spectrum_batch::<TestBackend>(&right_rows, &device);

        let config = make_config(point);

        let scores = cross_kernel::<TestBackend, M>(left_batch, right_batch, config)
            .into_data()
            .to_vec::<f32>()
            .expect("cross kernel output should be f32");

        assert_eq!(scores.len(), CROSS_M * CROSS_N);

        let mut max_delta = 0.0_f32;
        let mut worst = ("", "");
        let mut failures = 0usize;
        for (i, (left_name, left_spectrum)) in left.iter().enumerate() {
            for (j, (right_name, right_spectrum)) in right.iter().enumerate() {
                let gpu_score = scores[i * CROSS_N + j];
                let cpu = cpu_score(point, left_spectrum, right_spectrum);
                let delta = (gpu_score - cpu).abs();
                if delta > max_delta {
                    max_delta = delta;
                    worst = (left_name, right_name);
                }
                if delta >= tolerance {
                    failures += 1;
                }
            }
        }
        assert!(
            failures == 0,
            "{failures} cross scores exceeded tolerance {tolerance} at \
             (mz_power={}, intensity_power={}, mz_tolerance={}, metric={}); \
             worst pair {} vs {} delta={max_delta}",
            point.mz_power,
            point.intensity_power,
            point.mz_tolerance,
            M::NAME,
            worst.0,
            worst.1,
        );
    }
}

fn cosine_cross_config<M: KernelMetric>(point: ParameterPoint) -> CrossConfig<M> {
    M::cross_config()
        .with_mz_power(point.mz_power)
        .with_intensity_power(point.intensity_power)
        .with_mz_tolerance(point.mz_tolerance)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON)
}

fn entropy_cross_config<M: EntropyMetric>(point: ParameterPoint, weighted: bool) -> CrossConfig<M> {
    cosine_cross_config::<M>(point).with_weighted(weighted)
}

#[test]
fn cross_matches_cpu_linear_cosine() {
    run_cross_test_with::<LinearCosineMetric, _, _>(
        cpu_linear_cosine,
        1.0e-4,
        cosine_cross_config::<LinearCosineMetric>,
    );
}

#[test]
fn cross_matches_cpu_modified_linear_cosine() {
    run_cross_test_with::<ModifiedLinearCosineMetric, _, _>(
        cpu_modified_linear_cosine,
        2.0e-4,
        cosine_cross_config::<ModifiedLinearCosineMetric>,
    );
}

#[test]
fn cross_matches_cpu_linear_entropy_unweighted() {
    run_cross_test_with::<LinearEntropyMetric, _, _>(
        |p, l, r| cpu_linear_entropy(p, false, l, r),
        1.0e-4,
        |p| entropy_cross_config::<LinearEntropyMetric>(p, false),
    );
}

#[test]
fn cross_matches_cpu_linear_entropy_weighted() {
    run_cross_test_with::<LinearEntropyMetric, _, _>(
        |p, l, r| cpu_linear_entropy(p, true, l, r),
        2.0e-4,
        |p| entropy_cross_config::<LinearEntropyMetric>(p, true),
    );
}

#[test]
fn cross_matches_cpu_modified_linear_entropy_unweighted() {
    run_cross_test_with::<ModifiedLinearEntropyMetric, _, _>(
        |p, l, r| cpu_modified_linear_entropy(p, false, l, r),
        2.0e-4,
        |p| entropy_cross_config::<ModifiedLinearEntropyMetric>(p, false),
    );
}

#[test]
fn cross_matches_cpu_modified_linear_entropy_weighted() {
    run_cross_test_with::<ModifiedLinearEntropyMetric, _, _>(
        |p, l, r| cpu_modified_linear_entropy(p, true, l, r),
        2.0e-4,
        |p| entropy_cross_config::<ModifiedLinearEntropyMetric>(p, true),
    );
}
