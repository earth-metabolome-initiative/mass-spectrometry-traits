//! Criterion benchmarks for Flash index construction.
//!
//! These benchmarks compare the sequential constructors with the Rayon-backed
//! constructors for the three index types that can be built today:
//! `FlashCosineIndex`, `FlashCosineThresholdIndex`, and `FlashEntropyIndex`.
//!
//! Environment knobs:
//!
//! - `INDEX_CONSTRUCTION_SIZES=1000000`
//! - `INDEX_CONSTRUCTION_THRESHOLD=0.9`
//! - `INDEX_CONSTRUCTION_WARM_UP_MS=100`
//! - `INDEX_CONSTRUCTION_MEASUREMENT_SECS=1`
//! - `INDEX_CONSTRUCTION_SAMPLE_SIZE=10`

use std::{hint::black_box, time::Duration};

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineThresholdIndex, FlashEntropyIndex, GenericSpectrum,
    RandomSpectrumConfig, SpectrumAlloc,
};

const DEFAULT_LIBRARY_SIZES: &[usize] = &[1_000_000];
const PEAKS_PER_SPECTRUM: usize = 64;
const MZ_TOLERANCE: f64 = 0.1;
const MZ_POWER: f64 = 1.0;
const INTENSITY_POWER: f64 = 1.0;
const DEFAULT_SCORE_THRESHOLD: f64 = 0.9;
const RANDOM_BASE_SEED: u64 = 0xC0DE_1D5E_1A57_BA5E;

#[inline]
fn nonzero_seed(seed: u64) -> u64 {
    if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    }
}

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

#[inline]
fn next_unit_f64(state: &mut u64) -> f64 {
    const INV_2POW53: f64 = 1.0 / ((1_u64 << 53) as f64);
    ((next_u64(state) >> 11) as f64) * INV_2POW53
}

fn parse_library_sizes() -> Vec<usize> {
    let Ok(raw) = std::env::var("INDEX_CONSTRUCTION_SIZES") else {
        return DEFAULT_LIBRARY_SIZES.to_vec();
    };

    let parsed: Vec<usize> = raw
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .collect();

    if parsed.is_empty() {
        DEFAULT_LIBRARY_SIZES.to_vec()
    } else {
        parsed
    }
}

fn parse_score_threshold() -> f64 {
    std::env::var("INDEX_CONSTRUCTION_THRESHOLD")
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        .unwrap_or(DEFAULT_SCORE_THRESHOLD)
}

fn parse_u64_env(env_name: &str, default: u64) -> u64 {
    std::env::var(env_name)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn parse_usize_env(env_name: &str, default: usize) -> usize {
    std::env::var(env_name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&value| value >= 10)
        .unwrap_or(default)
}

fn random_spectrum_from_seed(seed: u64) -> GenericSpectrum {
    let mut state = nonzero_seed(seed);
    let precursor_mz = 650.0 + next_unit_f64(&mut state) * 550.0;
    let config = RandomSpectrumConfig {
        precursor_mz,
        n_peaks: PEAKS_PER_SPECTRUM,
        mz_min: 50.0,
        mz_max: 550.0,
        min_peak_gap: 0.25,
        intensity_min: 1.0,
        intensity_max: 1_000.0,
    };
    GenericSpectrum::random(config, seed).expect("benchmark spectrum should build")
}

fn build_random_spectra(count: usize) -> Vec<GenericSpectrum> {
    (0..count)
        .map(|index| {
            let seed =
                RANDOM_BASE_SEED.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            random_spectrum_from_seed(seed)
        })
        .collect()
}

#[cfg(feature = "rayon")]
fn bench_index_construction(c: &mut Criterion) {
    let library_sizes = parse_library_sizes();
    let score_threshold = parse_score_threshold();
    let warm_up_ms = parse_u64_env("INDEX_CONSTRUCTION_WARM_UP_MS", 100);
    let measurement_secs = parse_u64_env("INDEX_CONSTRUCTION_MEASUREMENT_SECS", 1);
    let sample_size = parse_usize_env("INDEX_CONSTRUCTION_SAMPLE_SIZE", 10);
    let mut group = c.benchmark_group("index_construction");
    group.sample_size(sample_size);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(warm_up_ms));
    group.measurement_time(Duration::from_secs(measurement_secs));

    for library_size in library_sizes {
        let spectra = build_random_spectra(library_size);
        group.throughput(Throughput::Elements(library_size as u64));

        group.bench_with_input(
            BenchmarkId::new("flash_cosine/sequential", library_size),
            &spectra,
            |b, spectra| {
                b.iter(|| {
                    let index = FlashCosineIndex::new(
                        black_box(MZ_POWER),
                        black_box(INTENSITY_POWER),
                        black_box(MZ_TOLERANCE),
                        black_box(spectra).iter(),
                    )
                    .expect("sequential cosine index should build");
                    black_box(index)
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("flash_cosine/rayon", library_size),
            &spectra,
            |b, spectra| {
                b.iter(|| {
                    let index = FlashCosineIndex::new_parallel(
                        black_box(MZ_POWER),
                        black_box(INTENSITY_POWER),
                        black_box(MZ_TOLERANCE),
                        black_box(spectra.as_slice()),
                    )
                    .expect("parallel cosine index should build");
                    black_box(index)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("flash_cosine_threshold/sequential", library_size),
            &spectra,
            |b, spectra| {
                b.iter(|| {
                    let index = FlashCosineThresholdIndex::new(
                        black_box(MZ_POWER),
                        black_box(INTENSITY_POWER),
                        black_box(MZ_TOLERANCE),
                        black_box(score_threshold),
                        black_box(spectra).iter(),
                    )
                    .expect("sequential threshold cosine index should build");
                    black_box(index)
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("flash_cosine_threshold/rayon", library_size),
            &spectra,
            |b, spectra| {
                b.iter(|| {
                    let index = FlashCosineThresholdIndex::new_parallel(
                        black_box(MZ_POWER),
                        black_box(INTENSITY_POWER),
                        black_box(MZ_TOLERANCE),
                        black_box(score_threshold),
                        black_box(spectra.as_slice()),
                    )
                    .expect("parallel threshold cosine index should build");
                    black_box(index)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("flash_entropy/sequential", library_size),
            &spectra,
            |b, spectra| {
                b.iter(|| {
                    let index = FlashEntropyIndex::new(
                        black_box(0.0),
                        black_box(1.0),
                        black_box(MZ_TOLERANCE),
                        black_box(true),
                        black_box(spectra).iter(),
                    )
                    .expect("sequential entropy index should build");
                    black_box(index)
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("flash_entropy/rayon", library_size),
            &spectra,
            |b, spectra| {
                b.iter(|| {
                    let index = FlashEntropyIndex::new_parallel(
                        black_box(0.0),
                        black_box(1.0),
                        black_box(MZ_TOLERANCE),
                        black_box(true),
                        black_box(spectra.as_slice()),
                    )
                    .expect("parallel entropy index should build");
                    black_box(index)
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "rayon"))]
fn bench_index_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_construction");
    group.bench_function("rayon_feature_required", |b| b.iter(|| black_box(())));
    group.finish();
}

criterion_group!(benches, bench_index_construction);
criterion_main!(benches);
