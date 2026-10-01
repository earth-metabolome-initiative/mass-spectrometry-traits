//! Hand-built edge-case fixtures: input shapes the random reference-spectra
//! sweep doesn't reliably hit.
//!
//! Each case runs every metric variant on a single-row paired batch and
//! asserts the GPU output matches the CPU reference. Structural invariants
//! (`sim in [0, 1]`, identical-spectra self-similarity, disjoint-spectra zero)
//! are asserted alongside CPU equivalence.

use burn::tensor::{Tensor as BurnTensor, TensorData};
use geometric_traits::prelude::ScalarSimilarity;

use crate::burn::{
    EntropyMetric, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric, PairedConfig, PairwiseParams,
    SpectralKernelBackend, SpectrumBatch, paired_kernel,
};
use crate::prelude::*;

use super::fixtures::{
    ReferenceSpectrum, TEST_INTENSITY_POWER, TEST_MAX_PEAKS, TEST_MZ_POWER, TEST_MZ_TOLERANCE,
    cosine_paired_config, entropy_paired_config,
};

#[cfg(feature = "burn-cuda")]
type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
type TestBackend = burn::backend::Cpu<f32, i32>;

type TestDevice = burn::tensor::Device<TestBackend>;

const TOLERANCE_LINEAR: f32 = 1.0e-4;
const TOLERANCE_MODIFIED: f32 = 2.0e-4;

fn build_spectrum(precursor: f32, peaks: &[(f32, f32)]) -> ReferenceSpectrum {
    let mut spectrum =
        ReferenceSpectrum::try_with_capacity(f64::from(precursor), peaks.len().max(1))
            .expect("valid precursor");
    for &(mz, intensity) in peaks {
        spectrum.add_peak(mz, intensity).expect("valid peak");
    }
    spectrum
}

/// Pack one (left, right) pair into row-1 paired tensors and run kernel `M`
/// with a pre-built [`PairedConfig<M>`]. Returns the scalar GPU score.
fn paired_score_one_with_config<M>(
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
    config: PairedConfig<M>,
) -> f32
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    let device = TestDevice::default();
    let peak_width = left.len().max(right.len()).max(1);

    let mut left_mz = vec![0.0_f32; peak_width];
    let mut left_intensity = vec![0.0_f32; peak_width];
    for (i, (mz, intensity)) in left.peaks().enumerate() {
        left_mz[i] = mz;
        left_intensity[i] = intensity;
    }
    let mut right_mz = vec![0.0_f32; peak_width];
    let mut right_intensity = vec![0.0_f32; peak_width];
    for (i, (mz, intensity)) in right.peaks().enumerate() {
        right_mz[i] = mz;
        right_intensity[i] = intensity;
    }

    let left_batch = SpectrumBatch::<TestBackend>::new(
        BurnTensor::from_data(TensorData::new(left_mz, [1, peak_width]), &device),
        BurnTensor::from_data(TensorData::new(left_intensity, [1, peak_width]), &device),
        BurnTensor::from_data(TensorData::new(vec![left.precursor_mz()], [1]), &device),
    );
    let right_batch = SpectrumBatch::<TestBackend>::new(
        BurnTensor::from_data(TensorData::new(right_mz, [1, peak_width]), &device),
        BurnTensor::from_data(TensorData::new(right_intensity, [1, peak_width]), &device),
        BurnTensor::from_data(TensorData::new(vec![right.precursor_mz()], [1]), &device),
    );
    let params = PairwiseParams::<TestBackend>::new(
        BurnTensor::from_data(TensorData::new(vec![TEST_MZ_POWER], [1]), &device),
        BurnTensor::from_data(TensorData::new(vec![TEST_INTENSITY_POWER], [1]), &device),
        BurnTensor::from_data(TensorData::new(vec![TEST_MZ_TOLERANCE], [1]), &device),
    );

    let scores = paired_kernel::<TestBackend, M>(left_batch, right_batch, params, config)
        .into_data()
        .to_vec::<f32>()
        .expect("kernel output should be f32");

    scores[0]
}

fn paired_score_cosine<M>(left: &ReferenceSpectrum, right: &ReferenceSpectrum) -> f32
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    paired_score_one_with_config::<M>(left, right, cosine_paired_config())
}

fn paired_score_entropy<M>(
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
    weighted: bool,
) -> f32
where
    M: EntropyMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    paired_score_one_with_config::<M>(left, right, entropy_paired_config(weighted))
}

fn cpu_linear_cosine(left: &ReferenceSpectrum, right: &ReferenceSpectrum) -> f32 {
    LinearCosine::new(
        f64::from(TEST_MZ_POWER),
        f64::from(TEST_INTENSITY_POWER),
        f64::from(TEST_MZ_TOLERANCE),
    )
    .expect("CPU linear cosine config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

fn cpu_modified_linear_cosine(left: &ReferenceSpectrum, right: &ReferenceSpectrum) -> f32 {
    ModifiedLinearCosine::new(
        f64::from(TEST_MZ_POWER),
        f64::from(TEST_INTENSITY_POWER),
        f64::from(TEST_MZ_TOLERANCE),
    )
    .expect("CPU modified linear cosine config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

fn cpu_linear_entropy(left: &ReferenceSpectrum, right: &ReferenceSpectrum, weighted: bool) -> f32 {
    LinearEntropy::new(
        f64::from(TEST_MZ_POWER),
        f64::from(TEST_INTENSITY_POWER),
        f64::from(TEST_MZ_TOLERANCE),
        weighted,
    )
    .expect("CPU linear entropy config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

fn cpu_modified_linear_entropy(
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
    weighted: bool,
) -> f32 {
    ModifiedLinearEntropy::new(
        f64::from(TEST_MZ_POWER),
        f64::from(TEST_INTENSITY_POWER),
        f64::from(TEST_MZ_TOLERANCE),
        weighted,
    )
    .expect("CPU modified linear entropy config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

/// For each (metric, weighted) combination, assert `paired_kernel ~ cpu_score`
/// on the given `(left, right)` pair within the per-metric tolerance.
fn assert_all_metrics_match_cpu(
    case_name: &str,
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
) {
    let pairs: [(&str, f32, f32); 6] = [
        (
            "LinearCosine",
            paired_score_cosine::<LinearCosineMetric>(left, right),
            cpu_linear_cosine(left, right),
        ),
        (
            "ModifiedLinearCosine",
            paired_score_cosine::<ModifiedLinearCosineMetric>(left, right),
            cpu_modified_linear_cosine(left, right),
        ),
        (
            "LinearEntropy(unweighted)",
            paired_score_entropy::<LinearEntropyMetric>(left, right, false),
            cpu_linear_entropy(left, right, false),
        ),
        (
            "LinearEntropy(weighted)",
            paired_score_entropy::<LinearEntropyMetric>(left, right, true),
            cpu_linear_entropy(left, right, true),
        ),
        (
            "ModifiedLinearEntropy(unweighted)",
            paired_score_entropy::<ModifiedLinearEntropyMetric>(left, right, false),
            cpu_modified_linear_entropy(left, right, false),
        ),
        (
            "ModifiedLinearEntropy(weighted)",
            paired_score_entropy::<ModifiedLinearEntropyMetric>(left, right, true),
            cpu_modified_linear_entropy(left, right, true),
        ),
    ];

    for (name, gpu, cpu) in pairs {
        let tolerance = if name.starts_with("Modified") || name.contains("weighted") {
            TOLERANCE_MODIFIED
        } else {
            TOLERANCE_LINEAR
        };
        assert!(
            (0.0..=1.0).contains(&gpu),
            "case '{case_name}' metric {name}: gpu score {gpu} outside [0, 1]"
        );
        assert!(
            (gpu - cpu).abs() < tolerance,
            "case '{case_name}' metric {name}: gpu={gpu} cpu={cpu} delta={}",
            (gpu - cpu).abs()
        );
    }
}

#[test]
fn edge_case_single_peak_matching() {
    // Two spectra with one peak each at the same m/z. Cosine should yield 1.0.
    // Entropy should yield close to 1.0 (entropy_pair(0.5, 0.5) = 1).
    let s = build_spectrum(150.0, &[(100.0, 1.0)]);
    assert_all_metrics_match_cpu("single_peak_matching", &s, &s);
}

#[test]
fn edge_case_single_peak_disjoint() {
    // Two single-peak spectra at m/z values much further apart than tolerance.
    let left = build_spectrum(150.0, &[(100.0, 1.0)]);
    let right = build_spectrum(150.0, &[(200.0, 1.0)]);
    let gpu_cosine = paired_score_cosine::<LinearCosineMetric>(&left, &right);
    assert!(
        gpu_cosine < 1.0e-4,
        "disjoint cosine should be ~0, got {gpu_cosine}"
    );
    assert_all_metrics_match_cpu("single_peak_disjoint", &left, &right);
}

#[test]
fn edge_case_identical_spectra() {
    // A multi-peak spectrum scored against itself: all metrics should approach
    // 1.0 (within their normalization tolerances).
    let s = build_spectrum(
        300.0,
        &[
            (50.0, 0.3),
            (75.0, 0.8),
            (110.0, 0.5),
            (150.0, 1.0),
            (200.0, 0.6),
            (250.0, 0.4),
        ],
    );
    let self_cosine = paired_score_cosine::<LinearCosineMetric>(&s, &s);
    assert!(
        self_cosine > 1.0 - 1.0e-3,
        "identical linear cosine should be ~1.0, got {self_cosine}"
    );
    assert_all_metrics_match_cpu("identical_spectra", &s, &s);
}

#[test]
fn edge_case_disjoint_multi_peak() {
    // Two multi-peak spectra with no m/z values within tolerance of each other.
    // Score should be 0 across all metrics.
    let left = build_spectrum(300.0, &[(50.0, 0.5), (75.0, 0.7), (110.0, 1.0)]);
    let right = build_spectrum(400.0, &[(200.0, 0.5), (225.0, 0.7), (250.0, 1.0)]);
    for (name, gpu) in [
        (
            "LinearCosine",
            paired_score_cosine::<LinearCosineMetric>(&left, &right),
        ),
        (
            "LinearEntropy(unweighted)",
            paired_score_entropy::<LinearEntropyMetric>(&left, &right, false),
        ),
    ] {
        assert!(
            gpu < 1.0e-4,
            "disjoint multi-peak {name}: expected ~0, got {gpu}"
        );
    }
    assert_all_metrics_match_cpu("disjoint_multi_peak", &left, &right);
}

#[test]
fn edge_case_max_peaks_saturated() {
    // Build two spectra with exactly TEST_MAX_PEAKS = 128 peaks each (the
    // configured upper bound). Each peak is spaced more than 2 * tolerance
    // from its neighbors so the well-separated invariant holds.
    let gap = f64::from(2.0 * TEST_MZ_TOLERANCE + 1.0e-3);
    let peaks: Vec<(f32, f32)> = (0..TEST_MAX_PEAKS)
        .map(|i| {
            let mz = 50.0 + (i as f64) * gap;
            let intensity = 0.1 + ((i % 11) as f32) * 0.08;
            (mz as f32, intensity)
        })
        .collect();
    let left = build_spectrum(800.0, &peaks);
    // Slightly perturb intensities for the right side so it isn't a clone.
    let right_peaks: Vec<(f32, f32)> = peaks
        .iter()
        .enumerate()
        .map(|(i, &(mz, intensity))| (mz, intensity * (1.0 - 0.01 * ((i % 7) as f32))))
        .collect();
    let right = build_spectrum(810.0, &right_peaks);
    assert_eq!(left.len(), TEST_MAX_PEAKS);
    assert_eq!(right.len(), TEST_MAX_PEAKS);
    assert_all_metrics_match_cpu("max_peaks_saturated", &left, &right);
}

#[test]
fn edge_case_modified_precursor_shift_triggers() {
    // Two spectra whose precursor delta exceeds the tolerance and that have
    // peaks aligning ONLY in the precursor-shifted reference frame. Designed
    // to force the modified variants' shifted sweep + DP path.
    //
    // Left has peaks at 100, 150. Right has peaks at 110, 160. Precursor
    // delta = 10, which is way above the 0.02 matching tolerance. Direct
    // matching: no peak within 0.02 of any other -> score ~ 0 for the linear
    // variants. Shifted matching: left peaks shifted by left_precursor (200)
    // become {-100, -50}. Right peaks shifted by right_precursor (210) become
    // {-100, -50}. These align -> modified variants should score > 0.
    let left = build_spectrum(200.0, &[(100.0, 0.7), (150.0, 1.0)]);
    let right = build_spectrum(210.0, &[(110.0, 0.7), (160.0, 1.0)]);

    let linear = paired_score_cosine::<LinearCosineMetric>(&left, &right);
    let modified = paired_score_cosine::<ModifiedLinearCosineMetric>(&left, &right);
    assert!(
        linear < 1.0e-4,
        "linear cosine should be ~0 when no peaks align directly, got {linear}"
    );
    assert!(
        modified > 0.5,
        "modified linear cosine should be substantial after precursor shift, got {modified}"
    );
    assert_all_metrics_match_cpu("modified_precursor_shift", &left, &right);
}
