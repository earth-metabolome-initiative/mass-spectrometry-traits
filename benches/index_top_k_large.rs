//! Large Criterion benchmark for exact thresholded index top-k queries.
//!
//! This target is intentionally focused on the self-similarity workload where
//! the query spectrum is already present in the indexed library. It defaults to
//! one million spectra so improvements are measured on a scale closer to graph
//! construction workloads.
//!
//! Useful knobs:
//! - `INDEX_SEARCH_TOP_K_LIBRARY_SIZE=1000000`
//! - `INDEX_SEARCH_TOP_K_QUERY_COUNT=10000`
//! - `INDEX_SEARCH_TOP_K=16`
//! - `INDEX_SEARCH_TOP_K_THRESHOLDS=0.9` or `0.5,0.7,0.9`
//! - `INDEX_SEARCH_TOP_K_ENTROPY_THRESHOLDS=0.75,0.9`
//! - `INDEX_SEARCH_TOP_K_METRICS=cosine`, `entropy-weighted`,
//!   `entropy-unweighted`, `entropy`, or `all`
//! - `INDEX_SEARCH_TOP_K_PRECISIONS=f64`, `f32`, `f16`, or `all`
//! - `INDEX_SEARCH_TOP_K_CLUSTER_SIZE=256`
//! - `INDEX_SEARCH_TOP_K_SHUFFLE=1`
//! - `INDEX_SEARCH_TOP_K_SAMPLE_SIZE=10`
//! - `INDEX_SEARCH_TOP_K_PEPMASS_TOLERANCE=0.5`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use half::f16;
#[cfg(feature = "rayon")]
use mass_spectrometry::prelude::FlashCosineSelfSimilarityIndex;
use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineThresholdIndex, FlashEntropyIndex, FlashSearchDiagnostics,
    FlashSearchResult, RandomSpectrumConfig, SpectraIndexBuilder, Spectrum, SpectrumAlloc,
    SpectrumFloat, SpectrumMut, TopKSearchState,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

type BenchSpectrum = mass_spectrometry::prelude::GenericSpectrum;

const DEFAULT_LIBRARY_SIZE: usize = 1_000_000;
const DEFAULT_QUERY_COUNT: usize = 10_000;
const DEFAULT_TOP_K: usize = 16;
const DEFAULT_CLUSTER_SIZE: usize = 256;
const DEFAULT_SAMPLE_SIZE: usize = 10;
const RANDOM_BASE_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

#[derive(Clone, Copy, Debug, Default)]
struct DiagnosticTotals {
    product_postings_visited: usize,
    spectrum_block_bound_entries_visited: usize,
    candidates_marked: usize,
    candidates_rescored: usize,
    results_emitted: usize,
    spectrum_blocks_evaluated: usize,
    spectrum_blocks_allowed: usize,
    spectrum_blocks_pruned: usize,
}

#[derive(Clone, Copy, Debug)]
struct BenchmarkMetrics {
    cosine: bool,
    entropy_weighted: bool,
    entropy_unweighted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkPrecision {
    F64,
    F32,
    F16,
}

#[derive(Clone, Copy, Debug)]
struct EntropyTopKBenchConfig<'a> {
    library: &'a [BenchSpectrum],
    query_ids: &'a [usize],
    top_k: usize,
    sample_size: usize,
    thresholds: &'a [f64],
    pepmass_tolerance: Option<f64>,
    weighted: bool,
    variant_name: &'a str,
    precision_label: &'a str,
    order_label: &'a str,
}

#[derive(Clone, Copy, Debug)]
struct CosineTopKBenchConfig<'a> {
    library: &'a [BenchSpectrum],
    query_ids: &'a [usize],
    top_k: usize,
    sample_size: usize,
    thresholds: &'a [f64],
    pepmass_tolerance: Option<f64>,
    precision_label: &'a str,
    order_label: &'a str,
}

impl DiagnosticTotals {
    fn add(&mut self, diagnostics: FlashSearchDiagnostics) {
        self.product_postings_visited = self
            .product_postings_visited
            .saturating_add(diagnostics.product_postings_visited);
        self.spectrum_block_bound_entries_visited = self
            .spectrum_block_bound_entries_visited
            .saturating_add(diagnostics.spectrum_block_bound_entries_visited);
        self.candidates_marked = self
            .candidates_marked
            .saturating_add(diagnostics.candidates_marked);
        self.candidates_rescored = self
            .candidates_rescored
            .saturating_add(diagnostics.candidates_rescored);
        self.results_emitted = self
            .results_emitted
            .saturating_add(diagnostics.results_emitted);
        self.spectrum_blocks_evaluated = self
            .spectrum_blocks_evaluated
            .saturating_add(diagnostics.spectrum_blocks_evaluated);
        self.spectrum_blocks_allowed = self
            .spectrum_blocks_allowed
            .saturating_add(diagnostics.spectrum_blocks_allowed);
        self.spectrum_blocks_pruned = self
            .spectrum_blocks_pruned
            .saturating_add(diagnostics.spectrum_blocks_pruned);
    }

    fn is_empty(&self) -> bool {
        self.product_postings_visited == 0
            && self.spectrum_block_bound_entries_visited == 0
            && self.candidates_marked == 0
            && self.candidates_rescored == 0
            && self.results_emitted == 0
            && self.spectrum_blocks_evaluated == 0
            && self.spectrum_blocks_allowed == 0
            && self.spectrum_blocks_pruned == 0
    }
}

fn per_query(total: usize, query_count: usize) -> f64 {
    if query_count == 0 {
        return 0.0;
    }
    total as f64 / query_count as f64
}

fn print_diagnostic_summary(
    bench_name: &str,
    threshold_label: &str,
    query_count: usize,
    diagnostics: DiagnosticTotals,
) {
    if diagnostics.is_empty() {
        return;
    }

    eprintln!(
        concat!(
            "index_top_k_large diagnostics ",
            "bench={bench_name}/{threshold_label} queries={query_count} ",
            "product_postings/q={product_postings:.3} ",
            "block_bound_entries/q={block_bound_entries:.3} ",
            "candidates_marked/q={candidates_marked:.3} ",
            "candidates_rescored/q={candidates_rescored:.3} ",
            "results_emitted/q={results_emitted:.3} ",
            "spectrum_blocks_evaluated/q={spectrum_blocks_evaluated:.3} ",
            "spectrum_blocks_allowed/q={spectrum_blocks_allowed:.3} ",
            "spectrum_blocks_pruned/q={spectrum_blocks_pruned:.3}"
        ),
        bench_name = bench_name,
        threshold_label = threshold_label,
        query_count = query_count,
        product_postings = per_query(diagnostics.product_postings_visited, query_count),
        block_bound_entries = per_query(
            diagnostics.spectrum_block_bound_entries_visited,
            query_count
        ),
        candidates_marked = per_query(diagnostics.candidates_marked, query_count),
        candidates_rescored = per_query(diagnostics.candidates_rescored, query_count),
        results_emitted = per_query(diagnostics.results_emitted, query_count),
        spectrum_blocks_evaluated = per_query(diagnostics.spectrum_blocks_evaluated, query_count),
        spectrum_blocks_allowed = per_query(diagnostics.spectrum_blocks_allowed, query_count),
        spectrum_blocks_pruned = per_query(diagnostics.spectrum_blocks_pruned, query_count),
    );
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|raw| matches!(raw.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn parse_optional_f64_env(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn order_group_suffix(order_label: &str) -> String {
    if order_label == "clustered" {
        String::new()
    } else {
        format!("_{order_label}")
    }
}

fn parse_thresholds_env(name: &str) -> Vec<f64> {
    let thresholds = std::env::var(name)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|item| item.trim().parse::<f64>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty());

    thresholds.unwrap_or_else(|| vec![0.9])
}

fn parse_entropy_thresholds_env() -> Vec<f64> {
    let thresholds = std::env::var("INDEX_SEARCH_TOP_K_ENTROPY_THRESHOLDS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|item| item.trim().parse::<f64>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty());

    thresholds.unwrap_or_else(|| vec![0.75, 0.9])
}

fn parse_metrics_env() -> BenchmarkMetrics {
    let raw = std::env::var("INDEX_SEARCH_TOP_K_METRICS").unwrap_or_else(|_| "cosine".to_owned());
    let mut metrics = BenchmarkMetrics {
        cosine: false,
        entropy_weighted: false,
        entropy_unweighted: false,
    };

    for item in raw.split(',').map(str::trim) {
        match item {
            "all" => {
                metrics.cosine = true;
                metrics.entropy_weighted = true;
                metrics.entropy_unweighted = true;
            }
            "cosine" => metrics.cosine = true,
            "entropy" => {
                metrics.entropy_weighted = true;
                metrics.entropy_unweighted = true;
            }
            "entropy-weighted" | "weighted-entropy" => metrics.entropy_weighted = true,
            "entropy-unweighted" | "unweighted-entropy" => metrics.entropy_unweighted = true,
            _ => {}
        }
    }

    if !metrics.cosine && !metrics.entropy_weighted && !metrics.entropy_unweighted {
        metrics.cosine = true;
    }
    metrics
}

fn parse_precisions_env() -> Vec<BenchmarkPrecision> {
    let raw = std::env::var("INDEX_SEARCH_TOP_K_PRECISIONS").unwrap_or_else(|_| "f64".to_owned());
    let mut precisions = Vec::new();

    fn push_unique(precisions: &mut Vec<BenchmarkPrecision>, precision: BenchmarkPrecision) {
        if !precisions.contains(&precision) {
            precisions.push(precision);
        }
    }

    for item in raw.split(',').map(str::trim) {
        match item {
            "all" => {
                for precision in [
                    BenchmarkPrecision::F64,
                    BenchmarkPrecision::F32,
                    BenchmarkPrecision::F16,
                ] {
                    push_unique(&mut precisions, precision);
                }
            }
            "f64" | "double" => {
                push_unique(&mut precisions, BenchmarkPrecision::F64);
            }
            "f32" | "float" => {
                push_unique(&mut precisions, BenchmarkPrecision::F32);
            }
            "f16" | "half" => {
                push_unique(&mut precisions, BenchmarkPrecision::F16);
            }
            _ => {}
        }
    }

    if precisions.is_empty() {
        precisions.push(BenchmarkPrecision::F64);
    }
    precisions
}

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
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn next_unit_f64(state: &mut u64) -> f64 {
    const INV_2POW53: f64 = 1.0 / ((1u64 << 53) as f64);
    ((next_u64(state) >> 11) as f64) * INV_2POW53
}

fn random_spectrum_from_seed(seed: u64) -> BenchSpectrum {
    let mut state = nonzero_seed(seed);
    let n_peaks = 48 + (next_u64(&mut state) % 49) as usize;
    let precursor_mz = 650.0 + (next_unit_f64(&mut state) * 550.0);
    let config = RandomSpectrumConfig {
        precursor_mz,
        n_peaks,
        mz_min: 50.0,
        mz_max: 550.0,
        min_peak_gap: 0.25,
        intensity_min: 1.0,
        intensity_max: 1_000.0,
    };
    BenchSpectrum::random(config, seed).expect("random benchmark spectrum should build")
}

fn perturb_spectrum(template: &BenchSpectrum, seed: u64) -> BenchSpectrum {
    let mut state = nonzero_seed(seed);
    let precursor_jitter = (next_unit_f64(&mut state) - 0.5) * 0.02;
    let mut spectrum =
        BenchSpectrum::with_capacity(template.precursor_mz() + precursor_jitter, template.len())
            .expect("perturbed benchmark spectrum should allocate");

    for (mz, intensity) in template.peaks() {
        let mz_jitter = (next_unit_f64(&mut state) - 0.5) * 0.02;
        let intensity_scale = 0.9 + next_unit_f64(&mut state) * 0.2;
        spectrum
            .add_peak(mz + mz_jitter, intensity * intensity_scale)
            .expect("small perturbations should preserve sorted, well-separated peaks");
    }

    spectrum
}

fn build_clustered_spectra(count: usize, cluster_size: usize, seed: u64) -> Vec<BenchSpectrum> {
    let mut spectra = Vec::with_capacity(count);
    let mut cluster_index = 0usize;
    let cluster_size = cluster_size.max(1);

    while spectra.len() < count {
        let base_seed =
            seed.wrapping_add((cluster_index as u64).wrapping_mul(0xA076_1D64_78BD_642F));
        let base = random_spectrum_from_seed(base_seed);
        let remaining = count - spectra.len();
        let current_cluster_size = remaining.min(cluster_size);

        for variant_index in 0..current_cluster_size {
            let variant_seed =
                base_seed ^ (variant_index as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB);
            spectra.push(perturb_spectrum(&base, variant_seed));
        }

        cluster_index += 1;
    }

    spectra
}

fn shuffle_spectra(spectra: &mut [BenchSpectrum], seed: u64) {
    let mut state = nonzero_seed(seed);
    for index in (1..spectra.len()).rev() {
        let swap_index = (next_u64(&mut state) as usize) % (index + 1);
        spectra.swap(index, swap_index);
    }
}

fn spread_query_ids(library_size: usize, query_count: usize) -> Vec<usize> {
    let query_count = query_count.min(library_size);
    if query_count == 0 {
        return Vec::new();
    }

    (0..query_count)
        .map(|index| (index * library_size) / query_count)
        .collect()
}

fn ranked_top_k(mut results: Vec<FlashSearchResult>, k: usize) -> Vec<FlashSearchResult> {
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.n_matches.cmp(&left.n_matches))
            .then_with(|| left.spectrum_id.cmp(&right.spectrum_id))
    });
    results.truncate(k);
    results
}

fn summarize_results(results: &[FlashSearchResult]) -> (f64, usize) {
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for result in results {
        score_sum += result.score;
        match_sum += result.n_matches;
    }
    (score_sum, match_sum)
}

fn bench_large_entropy_top_k<P>(c: &mut Criterion, config: EntropyTopKBenchConfig<'_>)
where
    P: SpectrumFloat + Send + Sync,
{
    let mz_power = 0.0_f64;
    let intensity_power = 1.0_f64;
    let mz_tolerance = 0.1_f64;
    let mut builder = FlashEntropyIndex::<P>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .weighted(config.weighted);
    if let Some(tolerance) = config.pepmass_tolerance {
        builder = match builder.pepmass_tolerance(tolerance) {
            Ok(builder) => builder,
            Err(error) => {
                eprintln!(
                    "index_top_k_large skipped entropy precision={} weighted={} because pepmass filter failed: {error:?}",
                    config.precision_label, config.weighted
                );
                return;
            }
        };
    }
    let flash_entropy = match builder.build(config.library) {
        Ok(index) => index,
        Err(error) => {
            eprintln!(
                "index_top_k_large skipped entropy precision={} weighted={} because index build failed: {error:?}",
                config.precision_label, config.weighted
            );
            return;
        }
    };
    let mut group = c.benchmark_group(format!(
        "library_search_entropy_{}_{}_top_k_large{}",
        config.variant_name,
        config.precision_label,
        order_group_suffix(config.order_label)
    ));
    group.sample_size(config.sample_size.max(10));

    for &threshold in config.thresholds {
        let threshold_label = format!("{threshold:.3}");

        let mut state = flash_entropy.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("external_top_k_threshold", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        flash_entropy
                            .for_each_top_k_threshold_with_state(
                                black_box(&config.library[query_id]),
                                black_box(config.top_k),
                                black_box(threshold),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("entropy external top-k threshold search should succeed");
                        diagnostics_totals.add(state.diagnostics());
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            &format!("entropy_{}_external_top_k_threshold", config.variant_name),
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );

        let mut state = flash_entropy.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("indexed_top_k_threshold", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        flash_entropy
                            .for_each_top_k_threshold_indexed_with_state(
                                black_box(
                                    u32::try_from(query_id)
                                        .expect("benchmark query id should fit in u32"),
                                ),
                                black_box(config.top_k),
                                black_box(threshold),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("entropy indexed top-k threshold search should succeed");
                        diagnostics_totals.add(state.diagnostics());
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            &format!("entropy_{}_indexed_top_k_threshold", config.variant_name),
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );

        let mut state = flash_entropy.new_search_state();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("threshold_then_sort", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        let results = flash_entropy
                            .search_threshold_with_state(
                                black_box(&config.library[query_id]),
                                black_box(threshold),
                                &mut state,
                            )
                            .expect("entropy threshold search should succeed");
                        let results = ranked_top_k(results, config.top_k);
                        let (score, matches) = summarize_results(&results);
                        total_score += score;
                        total_matches += matches;
                        diagnostics_totals.add(state.diagnostics());
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            &format!("entropy_{}_threshold_then_sort", config.variant_name),
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );

        let mut state = flash_entropy.new_search_state();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("indexed_emit_then_sort", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        let mut results = Vec::new();
                        flash_entropy
                            .for_each_threshold_indexed_with_state(
                                black_box(
                                    u32::try_from(query_id)
                                        .expect("benchmark query id should fit in u32"),
                                ),
                                black_box(threshold),
                                &mut state,
                                |result| results.push(result),
                            )
                            .expect("entropy indexed threshold search should succeed");
                        let results = ranked_top_k(results, config.top_k);
                        let (score, matches) = summarize_results(&results);
                        total_score += score;
                        total_matches += matches;
                        diagnostics_totals.add(state.diagnostics());
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            &format!("entropy_{}_indexed_emit_then_sort", config.variant_name),
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );
    }

    group.finish();
}

fn bench_large_index_top_k(c: &mut Criterion) {
    let library_size = parse_usize_env("INDEX_SEARCH_TOP_K_LIBRARY_SIZE", DEFAULT_LIBRARY_SIZE);
    let query_count = parse_usize_env("INDEX_SEARCH_TOP_K_QUERY_COUNT", DEFAULT_QUERY_COUNT);
    let top_k = parse_usize_env("INDEX_SEARCH_TOP_K", DEFAULT_TOP_K);
    let cluster_size = parse_usize_env("INDEX_SEARCH_TOP_K_CLUSTER_SIZE", DEFAULT_CLUSTER_SIZE);
    let sample_size = parse_usize_env("INDEX_SEARCH_TOP_K_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE);
    let shuffle_library = parse_bool_env("INDEX_SEARCH_TOP_K_SHUFFLE");
    let thresholds = parse_thresholds_env("INDEX_SEARCH_TOP_K_THRESHOLDS");
    let entropy_thresholds = parse_entropy_thresholds_env();
    let metrics = parse_metrics_env();
    let precisions = parse_precisions_env();
    let pepmass_tolerance = parse_optional_f64_env("INDEX_SEARCH_TOP_K_PEPMASS_TOLERANCE");

    let mut library =
        build_clustered_spectra(library_size, cluster_size, RANDOM_BASE_SEED ^ 0xC05E_C05E);
    if shuffle_library {
        shuffle_spectra(&mut library, RANDOM_BASE_SEED ^ 0x51A7_E51A);
    }
    let order_label = if shuffle_library {
        "shuffled"
    } else {
        "clustered"
    };
    let query_ids = spread_query_ids(library.len(), query_count);

    for precision in &precisions {
        let precision_label = match precision {
            BenchmarkPrecision::F64 => "f64",
            BenchmarkPrecision::F32 => "f32",
            BenchmarkPrecision::F16 => "f16",
        };

        if metrics.entropy_weighted {
            match precision {
                BenchmarkPrecision::F64 => bench_large_entropy_top_k::<f64>(
                    c,
                    EntropyTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &entropy_thresholds,
                        pepmass_tolerance,
                        weighted: true,
                        variant_name: "weighted",
                        precision_label,
                        order_label,
                    },
                ),
                BenchmarkPrecision::F32 => bench_large_entropy_top_k::<f32>(
                    c,
                    EntropyTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &entropy_thresholds,
                        pepmass_tolerance,
                        weighted: true,
                        variant_name: "weighted",
                        precision_label,
                        order_label,
                    },
                ),
                BenchmarkPrecision::F16 => bench_large_entropy_top_k::<f16>(
                    c,
                    EntropyTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &entropy_thresholds,
                        pepmass_tolerance,
                        weighted: true,
                        variant_name: "weighted",
                        precision_label,
                        order_label,
                    },
                ),
            }
        }
        if metrics.entropy_unweighted {
            match precision {
                BenchmarkPrecision::F64 => bench_large_entropy_top_k::<f64>(
                    c,
                    EntropyTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &entropy_thresholds,
                        pepmass_tolerance,
                        weighted: false,
                        variant_name: "unweighted",
                        precision_label,
                        order_label,
                    },
                ),
                BenchmarkPrecision::F32 => bench_large_entropy_top_k::<f32>(
                    c,
                    EntropyTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &entropy_thresholds,
                        pepmass_tolerance,
                        weighted: false,
                        variant_name: "unweighted",
                        precision_label,
                        order_label,
                    },
                ),
                BenchmarkPrecision::F16 => bench_large_entropy_top_k::<f16>(
                    c,
                    EntropyTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &entropy_thresholds,
                        pepmass_tolerance,
                        weighted: false,
                        variant_name: "unweighted",
                        precision_label,
                        order_label,
                    },
                ),
            }
        }
        if metrics.cosine {
            match precision {
                BenchmarkPrecision::F64 => bench_large_cosine_top_k_for_precision::<f64>(
                    c,
                    CosineTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &thresholds,
                        pepmass_tolerance,
                        precision_label,
                        order_label,
                    },
                ),
                BenchmarkPrecision::F32 => bench_large_cosine_top_k_for_precision::<f32>(
                    c,
                    CosineTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &thresholds,
                        pepmass_tolerance,
                        precision_label,
                        order_label,
                    },
                ),
                BenchmarkPrecision::F16 => bench_large_cosine_top_k_for_precision::<f16>(
                    c,
                    CosineTopKBenchConfig {
                        library: &library,
                        query_ids: &query_ids,
                        top_k,
                        sample_size,
                        thresholds: &thresholds,
                        pepmass_tolerance,
                        precision_label,
                        order_label,
                    },
                ),
            }
        }
    }
}

fn bench_large_cosine_top_k_for_precision<P>(c: &mut Criterion, config: CosineTopKBenchConfig<'_>)
where
    P: SpectrumFloat + Send + Sync,
{
    let mz_power = 1.0_f64;
    let intensity_power = 1.0_f64;
    let mz_tolerance = 0.1_f64;
    let mut builder = FlashCosineIndex::<P>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance);
    if let Some(tolerance) = config.pepmass_tolerance {
        builder = match builder.pepmass_tolerance(tolerance) {
            Ok(builder) => builder,
            Err(error) => {
                eprintln!(
                    "index_top_k_large skipped cosine precision={} because pepmass filter failed: {error:?}",
                    config.precision_label
                );
                return;
            }
        };
    }
    let flash_cosine = match builder.build(config.library) {
        Ok(index) => index,
        Err(error) => {
            eprintln!(
                "index_top_k_large skipped cosine precision={} because direct index build failed: {error:?}",
                config.precision_label
            );
            return;
        }
    };

    let mut group = c.benchmark_group(format!(
        "library_search_cosine_{}_top_k_large{}",
        config.precision_label,
        order_group_suffix(config.order_label)
    ));
    group.sample_size(config.sample_size.max(10));

    for &threshold in config.thresholds {
        let mut threshold_builder = FlashCosineThresholdIndex::<P>::builder()
            .mz_power(mz_power)
            .intensity_power(intensity_power)
            .mz_tolerance(mz_tolerance)
            .score_threshold(threshold);
        if let Some(tolerance) = config.pepmass_tolerance {
            threshold_builder = match threshold_builder.pepmass_tolerance(tolerance) {
                Ok(builder) => builder,
                Err(error) => {
                    eprintln!(
                        "index_top_k_large skipped cosine threshold precision={} threshold={threshold:.3} because pepmass filter failed: {error:?}",
                        config.precision_label
                    );
                    continue;
                }
            };
        }
        let flash_cosine_threshold = match threshold_builder.build(config.library) {
            Ok(index) => index,
            Err(error) => {
                eprintln!(
                    "index_top_k_large skipped cosine threshold precision={} threshold={threshold:.3} because threshold index build failed: {error:?}",
                    config.precision_label
                );
                continue;
            }
        };
        let threshold_label = format!("{threshold:.3}");
        #[cfg(feature = "rayon")]
        let query_ids_u32: Vec<u32> = config
            .query_ids
            .iter()
            .map(|&query_id| u32::try_from(query_id).expect("benchmark query id should fit in u32"))
            .collect();

        let mut state = flash_cosine.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("direct_top_k_threshold", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        flash_cosine
                            .for_each_top_k_threshold_with_state(
                                black_box(&config.library[query_id]),
                                black_box(config.top_k),
                                black_box(threshold),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("direct top-k threshold search should succeed");
                        diagnostics_totals.add(state.diagnostics());
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            "direct_top_k_threshold",
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );

        let mut state = flash_cosine.new_search_state();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("threshold_then_sort", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        let results = flash_cosine
                            .search_threshold_with_state(
                                black_box(&config.library[query_id]),
                                black_box(threshold),
                                &mut state,
                            )
                            .expect("threshold search should succeed");
                        let results = ranked_top_k(results, config.top_k);
                        let (score, matches) = summarize_results(&results);
                        total_score += score;
                        total_matches += matches;
                        diagnostics_totals.add(state.diagnostics());
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            "threshold_then_sort",
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );

        let mut state = flash_cosine_threshold.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("threshold_index_indexed_top_k", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        flash_cosine_threshold
                            .for_each_top_k_indexed_with_state(
                                black_box(
                                    u32::try_from(query_id)
                                        .expect("benchmark query id should fit in u32"),
                                ),
                                black_box(config.top_k),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("indexed top-k threshold search should succeed");
                        diagnostics_totals.add(state.diagnostics());
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            "threshold_index_indexed_top_k",
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );

        #[cfg(feature = "rayon")]
        group.bench_function(
            BenchmarkId::new("threshold_index_indexed_top_k_rayon", &threshold_label),
            |b| {
                b.iter(|| {
                    let (total_score, total_matches, diagnostics_totals) = query_ids_u32
                        .par_iter()
                        .copied()
                        .map_init(
                            || {
                                (
                                    flash_cosine_threshold.new_search_state(),
                                    TopKSearchState::new(),
                                )
                            },
                            |(state, top_k_state), query_id| {
                                let mut total_score = 0.0;
                                let mut total_matches = 0usize;
                                flash_cosine_threshold
                                    .for_each_top_k_indexed_with_state(
                                        black_box(query_id),
                                        black_box(config.top_k),
                                        state,
                                        top_k_state,
                                        |result| {
                                            total_score += result.score;
                                            total_matches += result.n_matches;
                                        },
                                    )
                                    .expect("parallel indexed top-k threshold search should work");
                                let mut diagnostics = DiagnosticTotals::default();
                                diagnostics.add(state.diagnostics());
                                (total_score, total_matches, diagnostics)
                            },
                        )
                        .reduce(
                            || (0.0_f64, 0usize, DiagnosticTotals::default()),
                            |left, right| {
                                let mut diagnostics = left.2;
                                diagnostics.product_postings_visited = diagnostics
                                    .product_postings_visited
                                    .saturating_add(right.2.product_postings_visited);
                                diagnostics.spectrum_block_bound_entries_visited = diagnostics
                                    .spectrum_block_bound_entries_visited
                                    .saturating_add(right.2.spectrum_block_bound_entries_visited);
                                diagnostics.candidates_marked = diagnostics
                                    .candidates_marked
                                    .saturating_add(right.2.candidates_marked);
                                diagnostics.candidates_rescored = diagnostics
                                    .candidates_rescored
                                    .saturating_add(right.2.candidates_rescored);
                                diagnostics.results_emitted = diagnostics
                                    .results_emitted
                                    .saturating_add(right.2.results_emitted);
                                diagnostics.spectrum_blocks_evaluated = diagnostics
                                    .spectrum_blocks_evaluated
                                    .saturating_add(right.2.spectrum_blocks_evaluated);
                                diagnostics.spectrum_blocks_allowed = diagnostics
                                    .spectrum_blocks_allowed
                                    .saturating_add(right.2.spectrum_blocks_allowed);
                                diagnostics.spectrum_blocks_pruned = diagnostics
                                    .spectrum_blocks_pruned
                                    .saturating_add(right.2.spectrum_blocks_pruned);
                                (left.0 + right.0, left.1 + right.1, diagnostics)
                            },
                        );
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );

        let mut state = flash_cosine_threshold.new_search_state();
        let mut last_diagnostics = DiagnosticTotals::default();
        group.bench_function(
            BenchmarkId::new("threshold_index_emit_then_sort", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    let mut diagnostics_totals = DiagnosticTotals::default();
                    for &query_id in config.query_ids {
                        let mut results = Vec::new();
                        flash_cosine_threshold
                            .for_each_indexed_with_state(
                                black_box(
                                    u32::try_from(query_id)
                                        .expect("benchmark query id should fit in u32"),
                                ),
                                &mut state,
                                |result| results.push(result),
                            )
                            .expect("indexed threshold search should succeed");
                        let results = ranked_top_k(results, config.top_k);
                        let (score, matches) = summarize_results(&results);
                        total_score += score;
                        total_matches += matches;
                        let diagnostics = state.diagnostics();
                        diagnostics_totals.add(diagnostics);
                    }
                    last_diagnostics = diagnostics_totals;
                    black_box((total_score, total_matches, diagnostics_totals))
                })
            },
        );
        print_diagnostic_summary(
            "threshold_index_emit_then_sort",
            &threshold_label,
            config.query_ids.len(),
            last_diagnostics,
        );

        #[cfg(feature = "rayon")]
        if let Some(pepmass_tolerance) = config.pepmass_tolerance {
            let self_similarity_builder = match FlashCosineSelfSimilarityIndex::<P>::builder()
                .mz_power(mz_power)
                .intensity_power(intensity_power)
                .mz_tolerance(mz_tolerance)
                .score_threshold(threshold)
                .top_k(config.top_k)
                .pepmass_tolerance(pepmass_tolerance)
            {
                Ok(builder) => builder.parallel(),
                Err(error) => {
                    eprintln!(
                        "index_top_k_large skipped cosine self-similarity precision={} threshold={threshold:.3} because pepmass filter failed: {error:?}",
                        config.precision_label
                    );
                    continue;
                }
            };
            let self_similarity = match self_similarity_builder.build(config.library) {
                Ok(index) => index,
                Err(error) => {
                    eprintln!(
                        "index_top_k_large skipped cosine self-similarity precision={} threshold={threshold:.3} because one-shot index build failed: {error:?}",
                        config.precision_label
                    );
                    continue;
                }
            };
            let mut last_diagnostics = DiagnosticTotals::default();
            group.bench_function(
                BenchmarkId::new("one_shot_self_similarity_top_k_rows", &threshold_label),
                |b| {
                    b.iter(|| {
                        let query_ids = black_box(query_ids_u32.as_slice());
                        let (total_score, total_matches, diagnostics_totals) = self_similarity
                            .rows()
                            .ids(query_ids)
                            .with_diagnostics()
                            .into_par_iter()
                            .map(|row| {
                                let row = row.expect("one-shot self row should score");
                                let (score, matches) =
                                    row.hits.iter().fold((0.0_f64, 0usize), |acc, hit| {
                                        (acc.0 + hit.score, acc.1 + hit.n_matches)
                                    });
                                let mut diagnostics = DiagnosticTotals::default();
                                diagnostics.add(row.diagnostics);
                                (score, matches, diagnostics)
                            })
                            .reduce(
                                || (0.0_f64, 0usize, DiagnosticTotals::default()),
                                |left, right| {
                                    let mut diagnostics = left.2;
                                    diagnostics.product_postings_visited = diagnostics
                                        .product_postings_visited
                                        .saturating_add(right.2.product_postings_visited);
                                    diagnostics.spectrum_block_bound_entries_visited = diagnostics
                                        .spectrum_block_bound_entries_visited
                                        .saturating_add(
                                            right.2.spectrum_block_bound_entries_visited,
                                        );
                                    diagnostics.candidates_marked = diagnostics
                                        .candidates_marked
                                        .saturating_add(right.2.candidates_marked);
                                    diagnostics.candidates_rescored = diagnostics
                                        .candidates_rescored
                                        .saturating_add(right.2.candidates_rescored);
                                    diagnostics.results_emitted = diagnostics
                                        .results_emitted
                                        .saturating_add(right.2.results_emitted);
                                    diagnostics.spectrum_blocks_evaluated = diagnostics
                                        .spectrum_blocks_evaluated
                                        .saturating_add(right.2.spectrum_blocks_evaluated);
                                    diagnostics.spectrum_blocks_allowed = diagnostics
                                        .spectrum_blocks_allowed
                                        .saturating_add(right.2.spectrum_blocks_allowed);
                                    diagnostics.spectrum_blocks_pruned = diagnostics
                                        .spectrum_blocks_pruned
                                        .saturating_add(right.2.spectrum_blocks_pruned);
                                    (left.0 + right.0, left.1 + right.1, diagnostics)
                                },
                            );
                        last_diagnostics = diagnostics_totals;
                        black_box((total_score, total_matches, diagnostics_totals))
                    })
                },
            );
            print_diagnostic_summary(
                "one_shot_self_similarity_top_k_rows",
                &threshold_label,
                config.query_ids.len(),
                last_diagnostics,
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_large_index_top_k);
criterion_main!(benches);
