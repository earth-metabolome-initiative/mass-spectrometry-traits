//! Pairwise spectral similarity timing: CPU (rayon-parallel) vs GPU (CUDA),
//! at both `f32` and `f64` spectrum-storage precision.
//!
//! For each of the four metrics, `LinearCosine`, `ModifiedLinearCosine`,
//! `LinearEntropy`, `ModifiedLinearEntropy`, the full N x N similarity matrix
//! is computed:
//!
//! * **f32 pass**: spectra stored as `GenericSpectrum<f32>`. CPU
//!   (rayon-parallel) compared against GPU (`Cuda<f32, i32>` backend). The
//!   GPU kernel runs entirely in f32.
//! * **f64 pass**: spectra stored as `GenericSpectrum<f64>`. CPU only. The
//!   GPU column is omitted because **burn-cuda 0.21 does not surface F64
//!   tensors on the CUDA backend**, `set_default_dtypes::<Cuda<f64, i64>>(
//!   &device, FloatDType::F64, _)` returns `UnsupportedDType`. The hardware
//!   supports f64 (at ~1/64 of f32 throughput on consumer NVIDIA parts), but
//!   the burn-cuda surface caps tensor storage at F32.
//!
//! The CPU reference implementations (`LinearCosine` etc.) upcast peaks to
//! f64 internally via `SpectrumFloat::to_f64`, so CPU arithmetic precision is
//! the same in both passes. What changes is only the spectrum storage /
//! iteration cost (one extra cast per peak in the f32 pass).
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example pairwise_timing --features burn,burn-cuda,rayon
//! ```

use std::time::{Duration, Instant};

use burn::backend::Cuda;
use burn::backend::cuda::CudaDevice;
use burn::tensor::{Tensor as BurnTensor, TensorData};
use geometric_traits::prelude::ScalarSimilarity;
use mass_spectrometry::burn::{
    KernelMetric, LinearCosineMetric, LinearEntropyMetric, ModifiedLinearCosineMetric,
    ModifiedLinearEntropyMetric, SpectralKernelBackend, SpectrumBatch, cross_kernel,
};
use mass_spectrometry::prelude::*;
use rayon::prelude::*;

type GpuBackend = Cuda<f32, i32>;

// 16384^2 = ~268M pair scores per metric. Firmly in the GPU-compute-bound
// steady state. Output matrix is ~1 GB at f32; well within a 24 GB RTX 4090.
// Smaller batches (e.g. 1024^2) are launch-overhead-bound for the cheaper
// metrics, LinearCosine in particular collapses to ~1x because the per-pair
// work is too small to amortize kernel launch + transfer.
const TARGET_BATCH: usize = 16384;
const MZ_POWER: f32 = 0.0;
const INTENSITY_POWER: f32 = 1.0;
const MZ_TOLERANCE: f32 = 0.02;
const EPSILON: f32 = 1.0e-8;
const MAX_PEAKS: usize = 128;
const TOLERANCE_LINEAR: f64 = 1.0e-4;
const TOLERANCE_MODIFIED: f64 = 2.0e-4;

fn main() {
    let sequence: Vec<usize> = (0..TARGET_BATCH).map(|i| i % reference_count()).collect();

    println!("Pairwise GPU<->CPU timing, mass-spectrometry-traits");
    println!("===================================================");
    println!(
        "Reference spectra: {} base entries, cycled to {} rows.",
        reference_count(),
        sequence.len()
    );
    println!(
        "Per metric: {n} x {n} = {} pair scores (CPU = rayon-parallel, GPU = CUDA).",
        sequence.len() * sequence.len(),
        n = sequence.len()
    );
    println!(
        "mz_power = {} | intensity_power = {} | mz_tolerance = {} | rayon threads = {}\n",
        MZ_POWER,
        INTENSITY_POWER,
        MZ_TOLERANCE,
        rayon::current_num_threads(),
    );

    let device = CudaDevice::default();

    // ----------------------------------------------------------------------
    // f32 pass: spectra stored at f32, CPU + GPU both run.
    // ----------------------------------------------------------------------
    let base_f32 = load_reference_spectra::<f32>();
    let row_width = base_f32
        .iter()
        .map(|s| s.len())
        .max()
        .expect("non-empty reference spectra");
    let packed_f32 = pack_rows(&base_f32, &sequence, row_width);

    println!("=== f32 (spectrum storage = f32, GPU = f32) ===");
    print_header_with_gpu();

    let cosine = LinearCosine::new(
        f64::from(MZ_POWER),
        f64::from(INTENSITY_POWER),
        f64::from(MZ_TOLERANCE),
    )
    .expect("CPU linear cosine config should be valid");
    bench_cpu_gpu::<f32, LinearCosineMetric, _>(
        "LinearCosine",
        |left, right| {
            cosine
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        TOLERANCE_LINEAR,
        &base_f32,
        &sequence,
        &packed_f32,
        row_width,
        &device,
    );

    let modified_cosine = ModifiedLinearCosine::new(
        f64::from(MZ_POWER),
        f64::from(INTENSITY_POWER),
        f64::from(MZ_TOLERANCE),
    )
    .expect("CPU modified linear cosine config should be valid");
    bench_cpu_gpu::<f32, ModifiedLinearCosineMetric, _>(
        "ModifiedLinearCosine",
        |left, right| {
            modified_cosine
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        TOLERANCE_MODIFIED,
        &base_f32,
        &sequence,
        &packed_f32,
        row_width,
        &device,
    );

    let entropy = LinearEntropy::new(
        f64::from(MZ_POWER),
        f64::from(INTENSITY_POWER),
        f64::from(MZ_TOLERANCE),
        false,
    )
    .expect("CPU linear entropy config should be valid");
    bench_cpu_gpu::<f32, LinearEntropyMetric, _>(
        "LinearEntropy",
        |left, right| {
            entropy
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        TOLERANCE_LINEAR,
        &base_f32,
        &sequence,
        &packed_f32,
        row_width,
        &device,
    );

    let modified_entropy = ModifiedLinearEntropy::new(
        f64::from(MZ_POWER),
        f64::from(INTENSITY_POWER),
        f64::from(MZ_TOLERANCE),
        false,
    )
    .expect("CPU modified linear entropy config should be valid");
    bench_cpu_gpu::<f32, ModifiedLinearEntropyMetric, _>(
        "ModifiedLinearEntropy",
        |left, right| {
            modified_entropy
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        TOLERANCE_MODIFIED,
        &base_f32,
        &sequence,
        &packed_f32,
        row_width,
        &device,
    );

    // ----------------------------------------------------------------------
    // f64 pass: spectra stored at f64, CPU only. GPU column omitted because
    // burn-cuda 0.21's CUDA backend does not support F64 tensors.
    // ----------------------------------------------------------------------
    let base_f64 = load_reference_spectra::<f64>();
    println!("\n=== f64 (spectrum storage = f64, GPU = unavailable, see note below) ===");
    print_header_cpu_only();

    bench_cpu_only(
        "LinearCosine",
        |left, right| {
            cosine
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        &base_f64,
        &sequence,
    );
    bench_cpu_only(
        "ModifiedLinearCosine",
        |left, right| {
            modified_cosine
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        &base_f64,
        &sequence,
    );
    bench_cpu_only(
        "LinearEntropy",
        |left, right| {
            entropy
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        &base_f64,
        &sequence,
    );
    bench_cpu_only(
        "ModifiedLinearEntropy",
        |left, right| {
            modified_entropy
                .similarity(left, right)
                .map(|(s, _)| s)
                .unwrap_or(0.0)
        },
        &base_f64,
        &sequence,
    );

    println!();
    println!("Notes:");
    println!("  * GPU outputs at f32 match the CPU reference within tolerance.");
    println!(
        "  * burn-cuda 0.21 reports `UnsupportedDType {{ dtype: F64 }}` for the CUDA backend, so"
    );
    println!("    the f64 pass only shows CPU times. The CPU side computes in f64 in both passes");
    println!(
        "    (the CPU scorers upcast peaks via `SpectrumFloat::to_f64`); the difference between"
    );
    println!("    passes is therefore the spectrum-storage / iteration cost.");
}

struct PackedRows<P> {
    mz: Vec<P>,
    intensity: Vec<P>,
    precursor: Vec<P>,
}

fn pack_rows<P: SpectrumFloat>(
    base: &[GenericSpectrum<P>],
    sequence: &[usize],
    row_width: usize,
) -> PackedRows<P> {
    let n = sequence.len();
    let zero = P::from_f64_lossy(0.0);
    let mut mz = Vec::with_capacity(n * row_width);
    let mut intensity = Vec::with_capacity(n * row_width);
    let mut precursor = Vec::with_capacity(n);
    for &index in sequence {
        let spectrum = &base[index];
        let start = mz.len();
        for (m, i) in spectrum.peaks() {
            mz.push(m);
            intensity.push(i);
        }
        mz.resize(start + row_width, zero);
        intensity.resize(start + row_width, zero);
        precursor.push(spectrum.precursor_mz());
    }
    PackedRows {
        mz,
        intensity,
        precursor,
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_cpu_gpu<P, M, Score>(
    label: &str,
    cpu_score: Score,
    tolerance: f64,
    base: &[GenericSpectrum<P>],
    sequence: &[usize],
    packed: &PackedRows<P>,
    row_width: usize,
    device: &CudaDevice,
) where
    P: SpectrumFloat + Sync + Send + burn::tensor::Element,
    GpuBackend: SpectralKernelBackend<M>,
    M: KernelMetric,
    Score: Fn(&GenericSpectrum<P>, &GenericSpectrum<P>) -> f64 + Sync + Send,
{
    let n = sequence.len();
    let config = M::cross_config()
        .with_mz_power(MZ_POWER)
        .with_intensity_power(INTENSITY_POWER)
        .with_mz_tolerance(MZ_TOLERANCE)
        .with_max_peaks(MAX_PEAKS)
        .with_epsilon(EPSILON);

    let make_batch = || -> SpectrumBatch<GpuBackend> {
        SpectrumBatch::new(
            BurnTensor::from_data(TensorData::new(packed.mz.clone(), [n, row_width]), device),
            BurnTensor::from_data(
                TensorData::new(packed.intensity.clone(), [n, row_width]),
                device,
            ),
            BurnTensor::from_data(TensorData::new(packed.precursor.clone(), [n]), device),
        )
    };

    // Warm-up: run at the FULL batch size so the CUDA memory pool gets
    // expanded to accommodate the 4*n^2 output allocation. JIT compilation
    // happens here too. (A small-batch warm-up doesn't fix the
    // memory-pool-expansion penalty, which manifests as a ~70% slowdown on
    // the first timed launch of the first metric tested.)
    {
        let _ = cross_kernel::<GpuBackend, M>(make_batch(), make_batch(), config)
            .into_data()
            .to_vec::<f32>()
            .expect("warm-up output");
    }

    // GPU timing, repeated runs to expose any sustained-workload effects
    // (clock ramp-up, L2 cache reuse, driver per-launch cost).
    const GPU_RUNS: usize = 5;
    let mut gpu_run_times: Vec<Duration> = Vec::with_capacity(GPU_RUNS);
    let mut gpu_scores: Vec<f32> = Vec::new();
    for _ in 0..GPU_RUNS {
        let run_start = Instant::now();
        let out: Vec<f32> = cross_kernel::<GpuBackend, M>(make_batch(), make_batch(), config)
            .into_data()
            .to_vec::<f32>()
            .expect("GPU kernel output should be f32");
        gpu_run_times.push(run_start.elapsed());
        gpu_scores = out;
    }
    // Use the median run for the headline number. Print all five so the
    // user can see whether the GPU "settles" with sustained launches.
    let mut sorted = gpu_run_times.clone();
    sorted.sort();
    let gpu_elapsed = sorted[GPU_RUNS / 2];
    assert_eq!(gpu_scores.len(), n * n);

    // CPU timing, rayon-parallel pair loop.
    let (cpu_scores, cpu_elapsed) = parallel_cpu_pairs(base, sequence, cpu_score);

    // Element-wise equivalence within tolerance.
    let mut max_delta = 0.0_f64;
    let mut max_index = 0usize;
    let mut failures = 0usize;
    for (idx, (&gpu, &cpu)) in gpu_scores.iter().zip(&cpu_scores).enumerate() {
        let delta = (f64::from(gpu) - cpu).abs();
        if delta > max_delta {
            max_delta = delta;
            max_index = idx;
        }
        if delta >= tolerance {
            failures += 1;
        }
    }
    assert!(
        failures == 0,
        "{label}: {failures} pair scores diverged beyond {tolerance}; \
         worst at index {max_index} delta={max_delta}",
    );

    let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64().max(1.0e-9);
    println!(
        "{label:<24} | {:>14} | {:>14} | {:>8.1}x | max delta {:.2e}",
        format_duration(cpu_elapsed),
        format_duration(gpu_elapsed),
        speedup,
        max_delta,
    );
    print!("  GPU runs:");
    for run_time in &gpu_run_times {
        print!(" {}", format_duration(*run_time));
    }
    println!();
}

fn bench_cpu_only<P, Score>(
    label: &str,
    cpu_score: Score,
    base: &[GenericSpectrum<P>],
    sequence: &[usize],
) where
    P: SpectrumFloat + Sync + Send,
    Score: Fn(&GenericSpectrum<P>, &GenericSpectrum<P>) -> f64 + Sync + Send,
{
    let (_scores, elapsed) = parallel_cpu_pairs(base, sequence, cpu_score);
    println!("{label:<24} | {:>14}", format_duration(elapsed));
}

fn parallel_cpu_pairs<P, Score>(
    base: &[GenericSpectrum<P>],
    sequence: &[usize],
    cpu_score: Score,
) -> (Vec<f64>, Duration)
where
    P: SpectrumFloat + Sync,
    Score: Fn(&GenericSpectrum<P>, &GenericSpectrum<P>) -> f64 + Sync + Send,
{
    let n = sequence.len();
    let start = Instant::now();
    let scores: Vec<f64> = (0..n * n)
        .into_par_iter()
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            cpu_score(&base[sequence[i]], &base[sequence[j]])
        })
        .collect();
    (scores, start.elapsed())
}

fn print_header_with_gpu() {
    println!(
        "{:<24} | {:>14} | {:>14} | {:>9} | Equivalence",
        "Metric", "CPU (rayon)", "GPU", "Speedup",
    );
    println!(
        "{:-<24}-+-{:-<14}-+-{:-<14}-+-{:-<9}-+-{:-<16}",
        "", "", "", "", "",
    );
}

fn print_header_cpu_only() {
    println!("{:<24} | {:>14}", "Metric", "CPU (rayon)");
    println!("{:-<24}-+-{:-<14}", "", "");
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.3} s")
    } else if secs >= 1.0e-3 {
        format!("{:.2} ms", secs * 1_000.0)
    } else {
        format!("{:.1} us", secs * 1_000_000.0)
    }
}

const fn reference_count() -> usize {
    74
}

fn load_reference_spectra<P: SpectrumFloat>() -> Vec<GenericSpectrum<P>> {
    let processor = SiriusMergeClosePeaks::<P>::new_with_precision(f64::from(MZ_TOLERANCE))
        .expect("reference-spectrum preprocess config should be valid");

    macro_rules! spectrum {
        ($method:ident) => {
            processor.process(
                &GenericSpectrum::<P>::$method()
                    .expect(concat!(stringify!($method), " spectrum should build"))
                    .top_k_peaks(MAX_PEAKS)
                    .expect(concat!(stringify!($method), " top-k should build")),
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
