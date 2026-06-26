//! Criterion benchmark for the modified-cosine top-k self-similarity workload.
//!
//! This is the t-SNE neighbor-search hot path: every spectrum queries the index
//! for its modified-cosine top-k neighbors. The canonical config mirrors the
//! `spectral_tsne_timing` example (`mz_power = 0.0`, `intensity_power = 0.25`,
//! `mz_tolerance = 0.02`) and `top_k = 3 * perplexity` at perplexity 30.
//!
//! The library is a set of perturbed clusters so that real near-duplicate
//! neighbors exist (intra-cluster, precursor-shared, direct matches) alongside
//! unrelated spectra with shifted (neutral-loss) matches across clusters.
//!
//! Knobs:
//! - `MODIFIED_TOPK_LIBRARY_SIZE=20000`
//! - `MODIFIED_TOPK_QUERY_COUNT=256`
//! - `MODIFIED_TOPK=90`
//! - `MODIFIED_TOPK_CLUSTER_SIZE=64`
//! - `MODIFIED_TOPK_SAMPLE_SIZE=10`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::{
    FlashCosineIndex, RandomSpectrumConfig, SpectraIndexBuilder, Spectrum, SpectrumAlloc,
    SpectrumMut, TopKSearchState,
};
#[cfg(feature = "minhash")]
use mass_spectrometry::prelude::{FlashCosineSketchIndex, MinHash};

type BenchSpectrum = mass_spectrometry::prelude::GenericSpectrum;

const DEFAULT_LIBRARY_SIZE: usize = 20_000;
const DEFAULT_QUERY_COUNT: usize = 256;
const DEFAULT_TOP_K: usize = 90;
const DEFAULT_CLUSTER_SIZE: usize = 64;
const DEFAULT_SAMPLE_SIZE: usize = 10;
const RANDOM_BASE_SEED: u64 = 0x5151_5151_2323_2323;

const MZ_POWER: f64 = 0.0;
const INTENSITY_POWER: f64 = 0.25;
const MZ_TOLERANCE: f64 = 0.02;

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
        mz_max: 600.0,
        min_peak_gap: 0.25,
        intensity_min: 1.0,
        intensity_max: 1_000.0,
    };
    BenchSpectrum::random(config, seed).expect("random benchmark spectrum should build")
}

/// A near-duplicate of `template`: same precursor (direct matches dominate),
/// small m/z and intensity jitter that preserves sorted, well-separated peaks.
fn perturb_spectrum(template: &BenchSpectrum, seed: u64) -> BenchSpectrum {
    let mut state = nonzero_seed(seed);
    let mut spectrum = BenchSpectrum::with_capacity(template.precursor_mz(), template.len())
        .expect("perturbed benchmark spectrum should allocate");
    for (mz, intensity) in template.peaks() {
        let mz_jitter = (next_unit_f64(&mut state) - 0.5) * 0.01;
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

fn spread_query_ids(library_size: usize, query_count: usize) -> Vec<usize> {
    let query_count = query_count.min(library_size);
    if query_count == 0 {
        return Vec::new();
    }
    (0..query_count)
        .map(|index| (index * library_size) / query_count)
        .collect()
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

const DEFAULT_M_VALUES: [usize; 4] = [4, 8, 16, 32];

fn parse_m_values() -> Vec<usize> {
    match std::env::var("MODIFIED_TOPK_PEAKS") {
        Ok(raw) => {
            let parsed: Vec<usize> = raw
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
                .collect();
            if parsed.is_empty() {
                DEFAULT_M_VALUES.to_vec()
            } else {
                parsed
            }
        }
        Err(_) => DEFAULT_M_VALUES.to_vec(),
    }
}

/// Build a MinHash-LSH index of one `(word, permutations, bands)` config, print
/// its recall and candidate volume against the precomputed `truth`, and register
/// a timed search arm on `group`.
#[cfg(feature = "minhash")]
macro_rules! sweep_lsh {
    (
        $group:expr, $library:expr, $query_ids:expr, $truth:expr,
        $top_k:expr, $library_size:expr, $word:literal, $w:ty, $perms:literal, $bands:expr
    ) => {{
        let lsh = FlashCosineSketchIndex::<f64, MinHash<$w, $perms>>::build_with_bands(
            $library,
            MZ_POWER,
            INTENSITY_POWER,
            MZ_TOLERANCE,
            $bands,
        )
        .expect("LSH index build should succeed");

        let mut state = lsh.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        let mut total_truth = 0usize;
        let mut recovered = 0usize;
        let mut rescored_total = 0usize;
        for (query_index, &query_id) in $query_ids.iter().enumerate() {
            let mut found: Vec<u32> = Vec::new();
            lsh.for_each_modified_top_k_with_state(
                &$library[query_id],
                $top_k + 1,
                &mut state,
                &mut top_k_state,
                |result| found.push(result.spectrum_id),
            )
            .expect("LSH modified top-k should succeed");
            rescored_total += state.diagnostics().candidates_rescored;
            total_truth += $truth[query_index].len();
            recovered += $truth[query_index]
                .iter()
                .filter(|id| found.contains(id))
                .count();
        }
        let recall = recovered as f64 / total_truth.max(1) as f64;
        let candidates = rescored_total as f64 / $query_ids.len().max(1) as f64;
        eprintln!(
            "index_modified_top_k lsh word={} perms={} bands={} sig_bytes/spec={} recall={recall:.4} candidates/q={candidates:.1}",
            $word,
            $perms,
            $bands,
            $perms * core::mem::size_of::<$w>()
        );

        let mut state = lsh.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        $group.bench_function(
            BenchmarkId::new(
                format!("lsh_{}x{}_b{}", $word, $perms, $bands),
                $library_size,
            ),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for &query_id in $query_ids {
                        lsh.for_each_modified_top_k_with_state(
                            black_box(&$library[query_id]),
                            black_box($top_k + 1),
                            &mut state,
                            &mut top_k_state,
                            |result| {
                                total_score += result.score;
                                total_matches += result.n_matches;
                            },
                        )
                        .expect("LSH modified top-k search should succeed");
                    }
                    black_box((total_score, total_matches))
                })
            },
        );
    }};
}

fn bench_modified_top_k(c: &mut Criterion) {
    let library_size = parse_usize_env("MODIFIED_TOPK_LIBRARY_SIZE", DEFAULT_LIBRARY_SIZE);
    let query_count = parse_usize_env("MODIFIED_TOPK_QUERY_COUNT", DEFAULT_QUERY_COUNT);
    let top_k = parse_usize_env("MODIFIED_TOPK", DEFAULT_TOP_K);
    let cluster_size = parse_usize_env("MODIFIED_TOPK_CLUSTER_SIZE", DEFAULT_CLUSTER_SIZE);
    let sample_size = parse_usize_env("MODIFIED_TOPK_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE);
    let rerank = parse_usize_env("MODIFIED_TOPK_RERANK", (top_k * 4).max(128));

    let library = build_clustered_spectra(library_size, cluster_size, RANDOM_BASE_SEED);
    let query_ids = spread_query_ids(library_size, query_count);

    let index = FlashCosineIndex::<f64>::builder()
        .mz_power(MZ_POWER)
        .intensity_power(INTENSITY_POWER)
        .mz_tolerance(MZ_TOLERANCE)
        .parallel()
        .build(&library)
        .expect("modified-cosine index build should succeed");

    let m_values = parse_m_values();

    // Dense modified top-k is the ground truth for recall. Compute it once.
    let mut probe_state = index.new_search_state();
    let truth: Vec<Vec<u32>> = query_ids
        .iter()
        .map(|&query_id| {
            let results = index
                .search_modified_top_k_with_state(&library[query_id], top_k + 1, &mut probe_state)
                .expect("dense modified top-k should succeed");
            results.iter().map(|result| result.spectrum_id).collect()
        })
        .collect();

    // Candidate volume the dense path fully scores per query (the no-pruning work).
    let mut candidate_total = 0usize;
    for &query_id in &query_ids {
        let results = index
            .search_modified_with_state(&library[query_id], &mut probe_state)
            .expect("modified search should succeed");
        candidate_total += results.len();
    }
    let candidates_per_query = candidate_total as f64 / query_ids.len().max(1) as f64;
    eprintln!(
        "index_modified_top_k probe library={library_size} queries={} top_k={top_k} \
         dense_candidates_scored/q={candidates_per_query:.1}",
        query_ids.len(),
    );

    // Recall and re-rank volume for each peak budget M.
    for &m in &m_values {
        let mut approx_state = index.new_search_state();
        let mut total_truth = 0usize;
        let mut recovered = 0usize;
        let mut rescored_total = 0usize;
        for (query_index, &query_id) in query_ids.iter().enumerate() {
            let results = index
                .search_modified_top_k_approx_with_state(
                    &library[query_id],
                    top_k + 1,
                    m,
                    rerank,
                    &mut approx_state,
                )
                .expect("approximate modified top-k should succeed");
            rescored_total += approx_state.diagnostics().candidates_rescored;
            let found: Vec<u32> = results.iter().map(|result| result.spectrum_id).collect();
            total_truth += truth[query_index].len();
            recovered += truth[query_index]
                .iter()
                .filter(|id| found.contains(id))
                .count();
        }
        let recall = recovered as f64 / total_truth.max(1) as f64;
        let rescored_per_query = rescored_total as f64 / query_ids.len().max(1) as f64;
        eprintln!(
            "index_modified_top_k approx M={m} rerank={rerank} recall={recall:.4} candidates_rescored/q={rescored_per_query:.1}"
        );
    }

    let mut group = c.benchmark_group("modified_top_k_self_similarity");
    group.sample_size(sample_size.max(10));

    // Baseline: the current dense accumulate-then-rank modified top-k path.
    let mut state = index.new_search_state();
    let mut top_k_state = TopKSearchState::new();
    group.bench_function(BenchmarkId::new("baseline_dense", library_size), |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for &query_id in &query_ids {
                index
                    .for_each_modified_top_k_with_state(
                        black_box(&library[query_id]),
                        black_box(top_k + 1),
                        &mut state,
                        &mut top_k_state,
                        |result| {
                            total_score += result.score;
                            total_matches += result.n_matches;
                        },
                    )
                    .expect("modified top-k search should succeed");
            }
            black_box((total_score, total_matches))
        })
    });

    // Approximate two-stage path at each peak budget M.
    for &m in &m_values {
        let mut state = index.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        group.bench_function(
            BenchmarkId::new(format!("approx_m{m}"), library_size),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for &query_id in &query_ids {
                        index
                            .for_each_modified_top_k_approx_with_state(
                                black_box(&library[query_id]),
                                black_box(top_k + 1),
                                black_box(m),
                                black_box(rerank),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("approximate modified top-k search should succeed");
                    }
                    black_box((total_score, total_matches))
                })
            },
        );
    }

    #[cfg(feature = "minhash")]
    {
        sweep_lsh!(
            group,
            &library,
            &query_ids,
            &truth,
            top_k,
            library_size,
            "u64",
            u64,
            128,
            16
        );
        sweep_lsh!(
            group,
            &library,
            &query_ids,
            &truth,
            top_k,
            library_size,
            "u32",
            u32,
            128,
            16
        );
        sweep_lsh!(
            group,
            &library,
            &query_ids,
            &truth,
            top_k,
            library_size,
            "u32",
            u32,
            64,
            8
        );
        sweep_lsh!(
            group,
            &library,
            &query_ids,
            &truth,
            top_k,
            library_size,
            "u32",
            u32,
            256,
            32
        );
        sweep_lsh!(
            group,
            &library,
            &query_ids,
            &truth,
            top_k,
            library_size,
            "u64",
            u64,
            256,
            32
        );
    }

    group.finish();
}

criterion_group!(benches, bench_modified_top_k);
criterion_main!(benches);
