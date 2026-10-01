//! Property-based equivalence tests: random well-separated spectra exercised
//! through the paired kernel for every metric variant, with random
//! `(mz_power, intensity_power)` per case so the proptest can shrink toward
//! exponent-dependent counterexamples.
//!
//! Cross and ranking call the same per-pair scorer (`score_rows`) as paired,
//! so paired equivalence under random inputs is sufficient to catch scoring
//! bugs across all three shapes. Each proptest is configured for a small
//! number of cases (`PROPTEST_CASES`) because every case launches a GPU
//! kernel.

#![cfg(all(any(feature = "burn-cuda", feature = "burn-cpu"), feature = "proptest"))]

use burn::tensor::{Tensor as BurnTensor, TensorData};
use geometric_traits::prelude::ScalarSimilarity;
use proptest::prelude::*;

use crate::burn::{
    EntropyMetric, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric, PairedConfig, PairwiseParams,
    SpectralKernelBackend, SpectrumBatch, paired_kernel,
};
use crate::prelude::*;

#[cfg(feature = "burn-cuda")]
type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
type TestBackend = burn::backend::Cpu<f32, i32>;

type TestDevice = burn::tensor::Device<TestBackend>;

const PROPTEST_CASES: u32 = 6;
const EPSILON: f32 = 1.0e-8;
const MIN_PEAKS: usize = 3;
const MAX_PEAKS: usize = 32;
const PEAK_WIDTH: usize = 64;
const BATCH_SIZE: usize = 4;

/// Range for the randomized `(mz_power, intensity_power)` exponents.
const POWER_RANGE: core::ops::RangeInclusive<f32> = 0.0_f32..=1.5_f32;

/// Range for the randomized `mz_tolerance`. Lower bound stays well above f32
/// rounding. Upper bound stays below the well-separated invariant the
/// generated spectra satisfy (gap > `2 * MAX_TOLERANCE + 1e-3`).
const TOLERANCE_RANGE: core::ops::RangeInclusive<f32> = 0.001_f32..=0.05_f32;
const MAX_TOLERANCE: f32 = 0.05;

/// One generated spectrum: random precursor + random number of well-separated
/// peaks with random intensities.
#[derive(Clone, Debug)]
struct GeneratedSpectrum {
    precursor: f32,
    peaks: Vec<(f32, f32)>,
}

impl GeneratedSpectrum {
    fn to_reference(&self) -> GenericSpectrum<f32> {
        let mut spectrum =
            GenericSpectrum::<f32>::try_with_capacity(f64::from(self.precursor), self.peaks.len())
                .expect("valid precursor");
        for &(mz, intensity) in &self.peaks {
            spectrum.add_peak(mz, intensity).expect("valid peak");
        }
        spectrum
    }
}

/// Generate a well-separated spectrum: each peak's m/z is at least
/// `2 * MAX_TOLERANCE + 1e-3` more than the previous peak's, so the strict
/// `gap > 2 * tolerance` invariant is satisfied for *every* tolerance the
/// proptest can draw.
fn arb_spectrum() -> impl Strategy<Value = GeneratedSpectrum> {
    let min_gap = f64::from(2.0 * MAX_TOLERANCE + 1.0e-3);
    (50.0_f64..900.0_f64, MIN_PEAKS..=MAX_PEAKS).prop_flat_map(move |(precursor, n)| {
        prop::collection::vec((0.0_f64..0.5_f64, 0.05_f64..1.0_f64), n).prop_map(
            move |gaps_intensities| {
                let mut peaks = Vec::with_capacity(gaps_intensities.len());
                let mut mz = 30.0_f64;
                for (extra_gap, intensity) in gaps_intensities {
                    mz += min_gap + extra_gap;
                    peaks.push((mz as f32, intensity as f32));
                }
                GeneratedSpectrum {
                    precursor: precursor as f32,
                    peaks,
                }
            },
        )
    })
}

fn append_row(spectrum: &GeneratedSpectrum, mz: &mut Vec<f32>, intensity: &mut Vec<f32>) {
    let start = mz.len();
    for &(m, i) in &spectrum.peaks {
        mz.push(m);
        intensity.push(i);
    }
    assert!(mz.len() - start <= PEAK_WIDTH);
    mz.resize(start + PEAK_WIDTH, 0.0);
    intensity.resize(start + PEAK_WIDTH, 0.0);
}

/// Run paired kernel on the generated batch of spectra (all pairs against
/// themselves) at the given parameter point and assert each output is within
/// `score_tolerance` of the CPU score.
fn run_paired_proptest_case_with_config<M>(
    spectra: &[GeneratedSpectrum],
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    cpu_score: impl Fn(&GenericSpectrum<f32>, &GenericSpectrum<f32>) -> f32,
    score_tolerance: f32,
    config: PairedConfig<M>,
) -> Result<(), TestCaseError>
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    let device = TestDevice::default();
    let n = spectra.len();
    let row_count = n * n;

    let mut left_mz = Vec::with_capacity(row_count * PEAK_WIDTH);
    let mut left_intensity = Vec::with_capacity(row_count * PEAK_WIDTH);
    let mut left_precursor = Vec::with_capacity(row_count);
    let mut right_mz = Vec::with_capacity(row_count * PEAK_WIDTH);
    let mut right_intensity = Vec::with_capacity(row_count * PEAK_WIDTH);
    let mut right_precursor = Vec::with_capacity(row_count);

    for left in spectra {
        for right in spectra {
            append_row(left, &mut left_mz, &mut left_intensity);
            append_row(right, &mut right_mz, &mut right_intensity);
            left_precursor.push(left.precursor);
            right_precursor.push(right.precursor);
        }
    }

    let left_batch = SpectrumBatch::<TestBackend>::new(
        BurnTensor::from_data(TensorData::new(left_mz, [row_count, PEAK_WIDTH]), &device),
        BurnTensor::from_data(
            TensorData::new(left_intensity, [row_count, PEAK_WIDTH]),
            &device,
        ),
        BurnTensor::from_data(TensorData::new(left_precursor, [row_count]), &device),
    );
    let right_batch = SpectrumBatch::<TestBackend>::new(
        BurnTensor::from_data(TensorData::new(right_mz, [row_count, PEAK_WIDTH]), &device),
        BurnTensor::from_data(
            TensorData::new(right_intensity, [row_count, PEAK_WIDTH]),
            &device,
        ),
        BurnTensor::from_data(TensorData::new(right_precursor, [row_count]), &device),
    );
    let params = PairwiseParams::<TestBackend>::new(
        BurnTensor::from_data(
            TensorData::new(vec![mz_power; row_count], [row_count]),
            &device,
        ),
        BurnTensor::from_data(
            TensorData::new(vec![intensity_power; row_count], [row_count]),
            &device,
        ),
        BurnTensor::from_data(
            TensorData::new(vec![mz_tolerance; row_count], [row_count]),
            &device,
        ),
    );

    let scores = paired_kernel::<TestBackend, M>(left_batch, right_batch, params, config)
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| TestCaseError::fail(format!("kernel output read failed: {err:?}")))?;

    for (row, (i, j)) in (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).enumerate() {
        let left_ref = spectra[i].to_reference();
        let right_ref = spectra[j].to_reference();
        let expected = cpu_score(&left_ref, &right_ref);
        let delta = (scores[row] - expected).abs();
        prop_assert!(
            delta < score_tolerance,
            "metric {} at (mz_power={mz_power}, intensity_power={intensity_power}, \
             mz_tolerance={mz_tolerance}) row {row} (i={i}, j={j}): \
             gpu={} cpu={} delta={delta}",
            M::NAME,
            scores[row],
            expected,
        );
    }
    Ok(())
}

fn proptest_cosine_config<M: KernelMetric>() -> PairedConfig<M> {
    M::paired_config()
        .with_max_peaks(PEAK_WIDTH)
        .with_epsilon(EPSILON)
}

fn proptest_entropy_config<M: EntropyMetric>(weighted: bool) -> PairedConfig<M> {
    proptest_cosine_config::<M>().with_weighted(weighted)
}

fn run_paired_proptest_case_cosine<M>(
    spectra: &[GeneratedSpectrum],
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    cpu_score: impl Fn(&GenericSpectrum<f32>, &GenericSpectrum<f32>) -> f32,
    score_tolerance: f32,
) -> Result<(), TestCaseError>
where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    run_paired_proptest_case_with_config::<M>(
        spectra,
        mz_power,
        intensity_power,
        mz_tolerance,
        cpu_score,
        score_tolerance,
        proptest_cosine_config::<M>(),
    )
}

fn run_paired_proptest_case_entropy<M>(
    spectra: &[GeneratedSpectrum],
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    cpu_score: impl Fn(&GenericSpectrum<f32>, &GenericSpectrum<f32>) -> f32,
    score_tolerance: f32,
    weighted: bool,
) -> Result<(), TestCaseError>
where
    M: EntropyMetric,
    TestBackend: SpectralKernelBackend<M>,
{
    run_paired_proptest_case_with_config::<M>(
        spectra,
        mz_power,
        intensity_power,
        mz_tolerance,
        cpu_score,
        score_tolerance,
        proptest_entropy_config::<M>(weighted),
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    #[test]
    fn proptest_paired_linear_cosine(
        spectra in prop::collection::vec(arb_spectrum(), BATCH_SIZE..=BATCH_SIZE),
        mz_power in POWER_RANGE,
        intensity_power in POWER_RANGE,
        mz_tolerance in TOLERANCE_RANGE,
    ) {
        let scorer = LinearCosine::new(
            f64::from(mz_power),
            f64::from(intensity_power),
            f64::from(mz_tolerance),
        ).expect("CPU linear cosine config should be valid");
        run_paired_proptest_case_cosine::<LinearCosineMetric>(
            &spectra,
            mz_power,
            intensity_power,
            mz_tolerance,
            |left, right| scorer.similarity(left, right).map(|(s, _)| s as f32).unwrap_or(0.0),
            1.0e-4,
        )?;
    }

    #[test]
    fn proptest_paired_modified_linear_cosine(
        spectra in prop::collection::vec(arb_spectrum(), BATCH_SIZE..=BATCH_SIZE),
        mz_power in POWER_RANGE,
        intensity_power in POWER_RANGE,
        mz_tolerance in TOLERANCE_RANGE,
    ) {
        let scorer = ModifiedLinearCosine::new(
            f64::from(mz_power),
            f64::from(intensity_power),
            f64::from(mz_tolerance),
        ).expect("CPU modified linear cosine config should be valid");
        run_paired_proptest_case_cosine::<ModifiedLinearCosineMetric>(
            &spectra,
            mz_power,
            intensity_power,
            mz_tolerance,
            |left, right| scorer.similarity(left, right).map(|(s, _)| s as f32).unwrap_or(0.0),
            2.0e-4,
        )?;
    }

    #[test]
    fn proptest_paired_linear_entropy_unweighted(
        spectra in prop::collection::vec(arb_spectrum(), BATCH_SIZE..=BATCH_SIZE),
        mz_power in POWER_RANGE,
        intensity_power in POWER_RANGE,
        mz_tolerance in TOLERANCE_RANGE,
    ) {
        let scorer = LinearEntropy::new(
            f64::from(mz_power),
            f64::from(intensity_power),
            f64::from(mz_tolerance),
            false,
        ).expect("CPU linear entropy (unweighted) config should be valid");
        run_paired_proptest_case_entropy::<LinearEntropyMetric>(
            &spectra,
            mz_power,
            intensity_power,
            mz_tolerance,
            |left, right| scorer.similarity(left, right).map(|(s, _)| s as f32).unwrap_or(0.0),
            1.0e-4,
            false,
        )?;
    }

    #[test]
    fn proptest_paired_linear_entropy_weighted(
        spectra in prop::collection::vec(arb_spectrum(), BATCH_SIZE..=BATCH_SIZE),
        mz_power in POWER_RANGE,
        intensity_power in POWER_RANGE,
        mz_tolerance in TOLERANCE_RANGE,
    ) {
        let scorer = LinearEntropy::new(
            f64::from(mz_power),
            f64::from(intensity_power),
            f64::from(mz_tolerance),
            true,
        ).expect("CPU linear entropy (weighted) config should be valid");
        run_paired_proptest_case_entropy::<LinearEntropyMetric>(
            &spectra,
            mz_power,
            intensity_power,
            mz_tolerance,
            |left, right| scorer.similarity(left, right).map(|(s, _)| s as f32).unwrap_or(0.0),
            2.0e-4,
            true,
        )?;
    }

    #[test]
    fn proptest_paired_modified_linear_entropy_unweighted(
        spectra in prop::collection::vec(arb_spectrum(), BATCH_SIZE..=BATCH_SIZE),
        mz_power in POWER_RANGE,
        intensity_power in POWER_RANGE,
        mz_tolerance in TOLERANCE_RANGE,
    ) {
        let scorer = ModifiedLinearEntropy::new(
            f64::from(mz_power),
            f64::from(intensity_power),
            f64::from(mz_tolerance),
            false,
        ).expect("CPU modified linear entropy (unweighted) config should be valid");
        run_paired_proptest_case_entropy::<ModifiedLinearEntropyMetric>(
            &spectra,
            mz_power,
            intensity_power,
            mz_tolerance,
            |left, right| scorer.similarity(left, right).map(|(s, _)| s as f32).unwrap_or(0.0),
            2.0e-4,
            false,
        )?;
    }

    #[test]
    fn proptest_paired_modified_linear_entropy_weighted(
        spectra in prop::collection::vec(arb_spectrum(), BATCH_SIZE..=BATCH_SIZE),
        mz_power in POWER_RANGE,
        intensity_power in POWER_RANGE,
        mz_tolerance in TOLERANCE_RANGE,
    ) {
        let scorer = ModifiedLinearEntropy::new(
            f64::from(mz_power),
            f64::from(intensity_power),
            f64::from(mz_tolerance),
            true,
        ).expect("CPU modified linear entropy (weighted) config should be valid");
        run_paired_proptest_case_entropy::<ModifiedLinearEntropyMetric>(
            &spectra,
            mz_power,
            intensity_power,
            mz_tolerance,
            |left, right| scorer.similarity(left, right).map(|(s, _)| s as f32).unwrap_or(0.0),
            2.0e-4,
            true,
        )?;
    }
}
