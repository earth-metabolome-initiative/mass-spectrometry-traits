//! Reference-spectrum fixtures shared by every GPU equivalence test.
//!
//! All 74 in-crate reference spectra are loaded, run through
//! `SiriusMergeClosePeaks` at the test's m/z tolerance, then truncated to
//! `TEST_MAX_PEAKS` via `top_k_peaks`. Rows are packed into dense `[N, P]`
//! arrays with zero-intensity padding, the same layout the kernels consume.

use alloc::vec::Vec;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;

use geometric_traits::prelude::ScalarSimilarity;

use crate::burn::api::{
    PairedConfig, PairwiseParams, RankingConfig, SpectralKernelBackend, SpectrumBatch,
    ranking_kernel,
};
use crate::burn::metrics::{EntropyMetric, KernelMetric};
use crate::prelude::*;

pub type ReferenceSpectrum = GenericSpectrum<f32>;

pub const TEST_MZ_POWER: f32 = 0.15;
pub const TEST_INTENSITY_POWER: f32 = 0.7;
pub const TEST_MZ_TOLERANCE: f32 = 0.02;
pub const TEST_EPSILON: f32 = 1.0e-8;
pub const TEST_MAX_PEAKS: usize = 128;
pub const PAIR_CHUNK_SIZE: usize = 64;

/// A single point in `(mz_power, intensity_power, mz_tolerance)` space, used
/// by every fixture test to sweep across canonical kernel parameter regimes.
#[derive(Clone, Copy, Debug)]
pub struct ParameterPoint {
    pub mz_power: f32,
    pub intensity_power: f32,
    pub mz_tolerance: f32,
}

/// Canonical parameter points the fixture tests sweep across. Two axes are
/// varied independently, exponent at baseline tolerance, then tolerance at
/// baseline exponent, rather than a full cartesian product, to keep test
/// runtime bounded while still validating each axis.
///
/// Exponent regimes:
/// * `(0.0, 1.0)`, matchms cosine default. `LinearEntropy::unweighted` /
///   `::weighted` default. Exercises the `mz_power == 0` GPU path.
/// * `(0.0, 0.5)`, sqrt-intensity, the most common matchms cosine variant.
/// * `(0.5, 0.5)`, symmetric sqrt. Non-zero `mz_power`.
/// * `(1.0, 1.0)`, full mz x intensity product. High dynamic range.
/// * `(0.15, 0.7)`, historical spectral-autoencoder baseline.
///
/// Tolerance regimes (at the baseline exponent):
/// * `0.001`, very tight matching, well-separated invariant most stringent,
///   few candidates, DP rarely engages.
/// * `0.1`, loose matching, many candidates, DP backtracking exercised
///   harder. Precursor-shift trigger fires for more reference-spectra pairs.
pub const CANONICAL_PARAMETER_POINTS: &[ParameterPoint] = &[
    // Exponent sweep at baseline tolerance.
    ParameterPoint {
        mz_power: 0.0,
        intensity_power: 1.0,
        mz_tolerance: TEST_MZ_TOLERANCE,
    },
    ParameterPoint {
        mz_power: 0.0,
        intensity_power: 0.5,
        mz_tolerance: TEST_MZ_TOLERANCE,
    },
    ParameterPoint {
        mz_power: 0.5,
        intensity_power: 0.5,
        mz_tolerance: TEST_MZ_TOLERANCE,
    },
    ParameterPoint {
        mz_power: 1.0,
        intensity_power: 1.0,
        mz_tolerance: TEST_MZ_TOLERANCE,
    },
    ParameterPoint {
        mz_power: TEST_MZ_POWER,
        intensity_power: TEST_INTENSITY_POWER,
        mz_tolerance: TEST_MZ_TOLERANCE,
    },
    // Tolerance sweep at baseline exponents.
    ParameterPoint {
        mz_power: TEST_MZ_POWER,
        intensity_power: TEST_INTENSITY_POWER,
        mz_tolerance: 0.001,
    },
    ParameterPoint {
        mz_power: TEST_MZ_POWER,
        intensity_power: TEST_INTENSITY_POWER,
        mz_tolerance: 0.1,
    },
];

/// Single-point default for tests that want the baseline scoring without sweeping.
#[cfg_attr(
    not(any(feature = "burn-autodiff", feature = "burn-fusion")),
    allow(dead_code)
)]
pub const DEFAULT_TEST_POINT: ParameterPoint = ParameterPoint {
    mz_power: TEST_MZ_POWER,
    intensity_power: TEST_INTENSITY_POWER,
    mz_tolerance: TEST_MZ_TOLERANCE,
};

/// All in-crate reference spectra after Sirius merging + top-k truncation,
/// preprocessed with the default `TEST_MZ_TOLERANCE`. Use
/// [`reference_spectra_at`] when sweeping tolerance.
pub fn reference_spectra() -> Vec<(&'static str, ReferenceSpectrum)> {
    reference_spectra_at(TEST_MZ_TOLERANCE)
}

/// Reference spectra preprocessed at the given matching tolerance.
///
/// `SiriusMergeClosePeaks` uses a merge window of `2 * tolerance`, so the
/// output satisfies the strict well-separated invariant `gap > 2 * tolerance`
/// required by `LinearCosine`. Sweeping tolerance therefore requires
/// re-running the preprocessor, the fixture cannot be cached across
/// tolerance values.
pub fn reference_spectra_at(tolerance: f32) -> Vec<(&'static str, ReferenceSpectrum)> {
    let processor = SiriusMergeClosePeaks::<f32>::new_with_precision(f64::from(tolerance))
        .expect("reference-spectrum preprocess config should be valid");

    macro_rules! spectrum {
        ($method:ident) => {
            (
                stringify!($method),
                processor.process(
                    &ReferenceSpectrum::$method()
                        .expect("reference spectrum should build")
                        .top_k_peaks(TEST_MAX_PEAKS)
                        .expect("reference spectrum top-k should build"),
                ),
            )
        };
    }

    vec![
        spectrum!(acephate),
        spectrum!(acetyl_coenzyme_a),
        spectrum!(adenine),
        spectrum!(adenosine),
        spectrum!(adenosine_5_diphosphate),
        spectrum!(adenosine_5_monophosphate),
        spectrum!(alanine),
        spectrum!(arachidic_acid),
        spectrum!(arachidonic_acid),
        spectrum!(arginine),
        spectrum!(ascorbic_acid),
        spectrum!(aspartic_acid),
        spectrum!(aspirin),
        spectrum!(avermectin),
        spectrum!(biotin),
        spectrum!(boscalid),
        spectrum!(chlorantraniliprole),
        spectrum!(chlorfluazuron),
        spectrum!(chlorotoluron),
        spectrum!(citric_acid),
        spectrum!(clothianidin),
        spectrum!(cocaine),
        spectrum!(cyazofamid),
        spectrum!(cymoxanil),
        spectrum!(cysteine),
        spectrum!(cytidine),
        spectrum!(cytidine_5_diphosphate),
        spectrum!(cytidine_5_triphosphate),
        spectrum!(desmosterol),
        spectrum!(diflubenzuron),
        spectrum!(dihydrosphingosine),
        spectrum!(diniconazole),
        spectrum!(dinotefuran),
        spectrum!(diuron),
        spectrum!(doramectin),
        spectrum!(elaidic_acid),
        spectrum!(epimeloscine),
        spectrum!(eprinomectin),
        spectrum!(ethiprole),
        spectrum!(ethirimol),
        spectrum!(fipronil),
        spectrum!(flonicamid),
        spectrum!(fluazinam),
        spectrum!(fludioxinil),
        spectrum!(flufenoxuron),
        spectrum!(fluometuron),
        spectrum!(flutolanil),
        spectrum!(folic_acid),
        spectrum!(forchlorfenuron),
        spectrum!(fuberidazole),
        spectrum!(glucose),
        spectrum!(halofenozide),
        spectrum!(hexaflumuron),
        spectrum!(hydramethylnon),
        spectrum!(hydroxy_cholesterol),
        spectrum!(ivermectin),
        spectrum!(lufenuron),
        spectrum!(metaflumizone),
        spectrum!(neburon),
        spectrum!(nitenpyram),
        spectrum!(novaluron),
        spectrum!(phenylalanine),
        spectrum!(prothioconazole),
        spectrum!(pymetrozine),
        spectrum!(pyrimethanil),
        spectrum!(salicin),
        spectrum!(stypoltrione),
        spectrum!(sulfentrazone),
        spectrum!(tebufenozide),
        spectrum!(teflubenzuron),
        spectrum!(thidiazuron),
        spectrum!(thiophanate),
        spectrum!(triadimefon),
        spectrum!(triflumuron),
    ]
}

fn peak_width(spectra: &[(&'static str, ReferenceSpectrum)]) -> usize {
    spectra
        .iter()
        .map(|(_name, spectrum)| spectrum.len())
        .max()
        .unwrap_or(0)
}

/// One spectrum per row, packed into dense `[N, peak_width]` arrays.
pub struct SpectrumRows {
    pub mz: Vec<f32>,
    pub intensity: Vec<f32>,
    pub precursor: Vec<f32>,
    pub peak_width: usize,
}

/// Build a [`SpectrumBatch`] on the given device from packed rows. Used by
/// the equivalence tests and the timing example to avoid re-typing the same
/// six lines of `BurnTensor::from_data` boilerplate per call.
pub fn spectrum_batch<B: Backend>(rows: &SpectrumRows, device: &B::Device) -> SpectrumBatch<B> {
    let row_count = rows.precursor.len();
    SpectrumBatch::new(
        BurnTensor::<B, 2>::from_data(
            TensorData::new(rows.mz.clone(), [row_count, rows.peak_width]),
            device,
        ),
        BurnTensor::<B, 2>::from_data(
            TensorData::new(rows.intensity.clone(), [row_count, rows.peak_width]),
            device,
        ),
        BurnTensor::<B, 1>::from_data(TensorData::new(rows.precursor.clone(), [row_count]), device),
    )
}

/// Build the `(left, right)` [`SpectrumBatch`] pair from a [`PairRows`].
pub fn pair_batches<B: Backend>(
    pairs: &PairRows,
    device: &B::Device,
) -> (SpectrumBatch<B>, SpectrumBatch<B>) {
    let row_count = pairs.indices.len();
    let left = SpectrumBatch::new(
        BurnTensor::<B, 2>::from_data(
            TensorData::new(pairs.left_mz.clone(), [row_count, pairs.peak_width]),
            device,
        ),
        BurnTensor::<B, 2>::from_data(
            TensorData::new(pairs.left_intensity.clone(), [row_count, pairs.peak_width]),
            device,
        ),
        BurnTensor::<B, 1>::from_data(
            TensorData::new(pairs.left_precursor.clone(), [row_count]),
            device,
        ),
    );
    let right = SpectrumBatch::new(
        BurnTensor::<B, 2>::from_data(
            TensorData::new(pairs.right_mz.clone(), [row_count, pairs.peak_width]),
            device,
        ),
        BurnTensor::<B, 2>::from_data(
            TensorData::new(pairs.right_intensity.clone(), [row_count, pairs.peak_width]),
            device,
        ),
        BurnTensor::<B, 1>::from_data(
            TensorData::new(pairs.right_precursor.clone(), [row_count]),
            device,
        ),
    );
    (left, right)
}

/// Default cosine-side [`PairedConfig<M>`] for the equivalence tests:
/// `max_peaks = TEST_MAX_PEAKS`, `epsilon = TEST_EPSILON`. Available for any
/// `M: KernelMetric`, including entropy metrics that don't need the weighted
/// prepass toggled.
pub fn cosine_paired_config<M: KernelMetric>() -> PairedConfig<M> {
    M::paired_config()
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON)
}

/// Default entropy-side [`PairedConfig<M>`]. Same numerics as
/// [`cosine_paired_config`] plus the weighted prepass toggle. The
/// [`EntropyMetric`] bound gates this builder to entropy metrics only.
pub fn entropy_paired_config<M: EntropyMetric>(weighted: bool) -> PairedConfig<M> {
    cosine_paired_config::<M>().with_weighted(weighted)
}

/// Build a [`PairwiseParams`] where every row uses the same scoring scalars.
pub fn pairwise_params_constant<B: Backend>(
    row_count: usize,
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    device: &B::Device,
) -> PairwiseParams<B> {
    PairwiseParams::new(
        BurnTensor::<B, 1>::from_data(
            TensorData::new(vec![mz_power; row_count], [row_count]),
            device,
        ),
        BurnTensor::<B, 1>::from_data(
            TensorData::new(vec![intensity_power; row_count], [row_count]),
            device,
        ),
        BurnTensor::<B, 1>::from_data(
            TensorData::new(vec![mz_tolerance; row_count], [row_count]),
            device,
        ),
    )
}

pub fn spectrum_rows(spectra: &[(&'static str, ReferenceSpectrum)]) -> SpectrumRows {
    let peak_width = peak_width(spectra);
    let mut mz = Vec::with_capacity(spectra.len() * peak_width);
    let mut intensity = Vec::with_capacity(spectra.len() * peak_width);
    let mut precursor = Vec::with_capacity(spectra.len());
    for (_name, spectrum) in spectra {
        append_row(spectrum, peak_width, &mut mz, &mut intensity);
        precursor.push(spectrum.precursor_mz());
    }
    SpectrumRows {
        mz,
        intensity,
        precursor,
        peak_width,
    }
}

fn append_row(
    spectrum: &ReferenceSpectrum,
    width: usize,
    mz_values: &mut Vec<f32>,
    intensity_values: &mut Vec<f32>,
) {
    let start = mz_values.len();
    for (mz, intensity) in spectrum.peaks() {
        mz_values.push(mz);
        intensity_values.push(intensity);
    }
    assert!(
        mz_values.len() - start <= width,
        "reference spectrum has more peaks than the fixed row width"
    );
    mz_values.resize(start + width, 0.0);
    intensity_values.resize(start + width, 0.0);
}

pub struct PairRows {
    pub left_mz: Vec<f32>,
    pub left_intensity: Vec<f32>,
    pub left_precursor: Vec<f32>,
    pub right_mz: Vec<f32>,
    pub right_intensity: Vec<f32>,
    pub right_precursor: Vec<f32>,
    pub indices: Vec<(usize, usize)>,
    pub peak_width: usize,
}

pub fn all_pair_indices(count: usize) -> Vec<(usize, usize)> {
    (0..count)
        .flat_map(|left_index| (0..count).map(move |right_index| (left_index, right_index)))
        .collect()
}

pub fn pair_rows(
    spectra: &[(&'static str, ReferenceSpectrum)],
    indices: &[(usize, usize)],
) -> PairRows {
    let peak_width = peak_width(spectra);
    let pair_count = indices.len();
    let mut left_mz = Vec::with_capacity(pair_count * peak_width);
    let mut left_intensity = Vec::with_capacity(pair_count * peak_width);
    let mut left_precursor = Vec::with_capacity(pair_count);
    let mut right_mz = Vec::with_capacity(pair_count * peak_width);
    let mut right_intensity = Vec::with_capacity(pair_count * peak_width);
    let mut right_precursor = Vec::with_capacity(pair_count);

    for &(left_index, right_index) in indices {
        let left_spectrum = &spectra[left_index].1;
        let right_spectrum = &spectra[right_index].1;
        append_row(left_spectrum, peak_width, &mut left_mz, &mut left_intensity);
        append_row(
            right_spectrum,
            peak_width,
            &mut right_mz,
            &mut right_intensity,
        );
        left_precursor.push(left_spectrum.precursor_mz());
        right_precursor.push(right_spectrum.precursor_mz());
    }

    PairRows {
        left_mz,
        left_intensity,
        left_precursor,
        right_mz,
        right_intensity,
        right_precursor,
        indices: indices.to_vec(),
        peak_width,
    }
}

/// Compare GPU per-pair scores against a CPU reference, asserting all deltas
/// are within `tolerance`. Reports the worst offender, including the
/// `(mz_power, intensity_power)` point under test, in the panic message.
pub fn assert_all_pair_scores_match(
    spectra: &[(&'static str, ReferenceSpectrum)],
    indices: &[(usize, usize)],
    scores: &[f32],
    tolerance: f32,
    mz_power: f32,
    intensity_power: f32,
    mut reference_score: impl FnMut(&ReferenceSpectrum, &ReferenceSpectrum) -> f32,
) {
    assert!(
        !indices.is_empty(),
        "reference collection should contain spectra"
    );
    let mut max_delta = 0.0_f32;
    let mut max_row = 0usize;
    let mut max_pair = ("", "");
    let mut failures = 0usize;

    for (row, &(left_index, right_index)) in indices.iter().enumerate() {
        let (left_name, left) = &spectra[left_index];
        let (right_name, right) = &spectra[right_index];
        let expected = reference_score(left, right);
        let delta = (scores[row] - expected).abs();
        if delta > max_delta {
            max_delta = delta;
            max_row = row;
            max_pair = (left_name, right_name);
        }
        if delta >= tolerance {
            failures += 1;
        }
    }

    assert!(
        failures == 0,
        "{failures} pair scores exceeded tolerance {tolerance} \
         at (mz_power={mz_power}, intensity_power={intensity_power}); \
         max row {max_row} {} vs {} delta={max_delta}",
        max_pair.0,
        max_pair.1
    );
}

/// CPU reference: `LinearCosine` similarity at the given parameter point,
/// clamped to `f32` and yielding `0.0` on errors. Used to compare against the
/// GPU kernel output in every equivalence test.
pub fn cpu_linear_cosine(
    p: ParameterPoint,
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
) -> f32 {
    LinearCosine::new(
        f64::from(p.mz_power),
        f64::from(p.intensity_power),
        f64::from(p.mz_tolerance),
    )
    .expect("CPU linear cosine config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

/// CPU reference: `ModifiedLinearCosine` similarity at the given parameter point.
pub fn cpu_modified_linear_cosine(
    p: ParameterPoint,
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
) -> f32 {
    ModifiedLinearCosine::new(
        f64::from(p.mz_power),
        f64::from(p.intensity_power),
        f64::from(p.mz_tolerance),
    )
    .expect("CPU modified linear cosine config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

/// CPU reference: `LinearEntropy` similarity at the given parameter point,
/// with the Shannon-weighted prepass toggled by `weighted`.
pub fn cpu_linear_entropy(
    p: ParameterPoint,
    weighted: bool,
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
) -> f32 {
    LinearEntropy::new(
        f64::from(p.mz_power),
        f64::from(p.intensity_power),
        f64::from(p.mz_tolerance),
        weighted,
    )
    .expect("CPU linear entropy config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

/// CPU reference: `ModifiedLinearEntropy` similarity at the given parameter point.
pub fn cpu_modified_linear_entropy(
    p: ParameterPoint,
    weighted: bool,
    left: &ReferenceSpectrum,
    right: &ReferenceSpectrum,
) -> f32 {
    ModifiedLinearEntropy::new(
        f64::from(p.mz_power),
        f64::from(p.intensity_power),
        f64::from(p.mz_tolerance),
        weighted,
    )
    .expect("CPU modified linear entropy config should be valid")
    .similarity(left, right)
    .map(|(s, _)| s as f32)
    .unwrap_or(0.0)
}

/// GPU-only self-consistency check on one ranking-kernel output row.
/// Used by every ranking equivalence test (raw-backend sweep, wrapper
/// smoke tests, the f16 smoke test) so the same set of invariants is
/// enforced everywhere. The checks are:
///
/// * Every score is finite.
/// * Every score is in `[-tolerance, 1 + tolerance]`.
/// * `gpu_row_scores[gpu_best_position]` equals the row's top-1 score
///   within `tolerance` (catches `best_position` wiring bugs).
/// * `gpu_top2_gap` equals `(top1 - top2).clamp(0, 1)` within `tolerance`.
///
/// Caller passes a `context` string that gets prefixed onto every failure
/// message, typically the metric name plus any sweep coordinates.
pub fn assert_ranking_row_self_consistency(
    context: &str,
    anchor: usize,
    gpu_row_scores: &[f32],
    gpu_best_position: usize,
    gpu_top2_gap: f32,
    tolerance: f32,
) {
    for (column, &score) in gpu_row_scores.iter().enumerate() {
        assert!(
            score.is_finite(),
            "{context}: non-finite gpu score at anchor {anchor} col {column}",
        );
        assert!(
            (-tolerance..=1.0 + tolerance).contains(&score),
            "{context}: gpu score {score} outside [0, 1] (slack {tolerance}) at anchor {anchor} col {column}",
        );
    }
    let mut sorted = gpu_row_scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    let argmax_score = gpu_row_scores[gpu_best_position];
    assert!(
        (argmax_score - sorted[0]).abs() < tolerance,
        "{context}: anchor {anchor}: argmax_score {argmax_score} != top-1 {}",
        sorted[0],
    );
    let expected_gap = (sorted[0] - sorted[1]).clamp(0.0, 1.0);
    assert!(
        (gpu_top2_gap - expected_gap).abs() < tolerance,
        "{context}: anchor {anchor}: top2_gap {gpu_top2_gap} != top1-top2 {expected_gap}",
    );
}

/// Shared ranking-kernel config baked at [`DEFAULT_TEST_POINT`]. Mirrors
/// the `cosine_ranking_config` / `entropy_ranking_config` helpers in
/// `ranking.rs` but with the parameter point fixed, so every wrapper test
/// (Autodiff, Fusion, their composition, the minimum-batch edge case)
/// compares against the same CPU reference values used by the raw-backend
/// sweep at that point.
#[cfg(any(feature = "burn-autodiff", feature = "burn-fusion"))]
pub fn default_ranking_config<M: KernelMetric>() -> RankingConfig<M> {
    M::ranking_config()
        .with_batch_start(1)
        .with_batch_items(10)
        .with_candidates_per_anchor(7)
        .with_mz_power(TEST_MZ_POWER)
        .with_intensity_power(TEST_INTENSITY_POWER)
        .with_mz_tolerance(TEST_MZ_TOLERANCE)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_seed(12_345)
        .with_epsilon(TEST_EPSILON)
}

/// Entropy variant of [`default_ranking_config`] that also sets the
/// weighted/unweighted prepass toggle.
#[cfg(any(feature = "burn-autodiff", feature = "burn-fusion"))]
pub fn default_entropy_ranking_config<M: EntropyMetric>(weighted: bool) -> RankingConfig<M> {
    default_ranking_config::<M>().with_weighted(weighted)
}

/// Per-cell ranking-kernel equivalence assertion. Shared by the raw-backend
/// sweep and by each wrapper smoke-test (`Autodiff`, `Fusion`, and their
/// composition) so the same correctness contract is enforced everywhere.
///
/// For each anchor in `[batch_start, batch_start + batch_items)` we trust the
/// GPU's own `candidate_index` (independently verified by the LCG-replay test
/// in `ranking.rs`) to identify the sampled partners, then score each pair on
/// the CPU and compare per cell. We also rebuild the top-1 and top-2 of the
/// row directly from the GPU scores to verify `best_position` and `top2_gap`
/// without referencing the CPU reference for those two outputs.
pub fn assert_ranking_matches_cpu<B, M>(
    spectra: &[(&'static str, ReferenceSpectrum)],
    device: &B::Device,
    config: RankingConfig<M>,
    cpu_score: impl Fn(&ReferenceSpectrum, &ReferenceSpectrum) -> f32,
    tolerance: f32,
) where
    B: SpectralKernelBackend<M>,
    M: KernelMetric,
{
    let rows = spectrum_rows(spectra);
    let teacher = spectrum_batch::<B>(&rows, device);
    let output = ranking_kernel::<B, M>(teacher, config);

    let batch_items = config.batch_items();
    let batch_start = config.batch_start();
    let candidate_count = config.effective_candidates_per_anchor();
    assert!(
        batch_start + batch_items <= spectra.len(),
        "test setup: batch slice [{batch_start}, {}) outside spectra (len={})",
        batch_start + batch_items,
        spectra.len(),
    );
    assert_eq!(
        output.candidate_index.dims(),
        [batch_items, candidate_count]
    );
    assert_eq!(output.best_position.dims(), [batch_items]);
    assert_eq!(output.top2_gap.dims(), [batch_items]);
    assert_eq!(
        output.candidate_scores.dims(),
        [batch_items, candidate_count]
    );

    let candidate_index = output
        .candidate_index
        .into_data()
        .to_vec::<i32>()
        .expect("candidate indices should be i32");
    let best_position = output
        .best_position
        .into_data()
        .to_vec::<i32>()
        .expect("best position should be i32");
    let top2_gap = output
        .top2_gap
        .into_data()
        .to_vec::<f32>()
        .expect("top-2 gap should be f32");
    let candidate_scores = output
        .candidate_scores
        .into_data()
        .to_vec::<f32>()
        .expect("candidate scores should be f32");

    for anchor in 0..batch_items {
        let anchor_spectrum = &spectra[batch_start + anchor].1;
        let row_start = anchor * candidate_count;
        let row_candidates = &candidate_index[row_start..row_start + candidate_count];
        let row_scores = &candidate_scores[row_start..row_start + candidate_count];

        assert_ranking_row_self_consistency(
            M::NAME,
            anchor,
            row_scores,
            best_position[anchor] as usize,
            top2_gap[anchor],
            tolerance,
        );

        for (column, &gpu) in row_scores.iter().enumerate() {
            let partner_idx = row_candidates[column] as usize;
            let partner_spectrum = &spectra[batch_start + partner_idx].1;
            let cpu = cpu_score(anchor_spectrum, partner_spectrum);
            assert!(
                (gpu - cpu).abs() < tolerance,
                "{}: anchor {anchor} col {column}: gpu={gpu} cpu={cpu}",
                M::NAME,
            );
        }
    }
}
