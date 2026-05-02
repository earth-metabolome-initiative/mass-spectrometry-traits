//! Criterion benchmarks comparing indexed library search versus full library scans.
//!
//! Compares:
//! - cosine: `FlashCosineIndex::search` vs `LinearCosine` over all library spectra
//! - modified cosine: `FlashCosineIndex::search_modified` vs `ModifiedLinearCosine`
//! - entropy: `FlashEntropyIndex::search` vs `LinearEntropy` over all library spectra
//! - modified entropy: `FlashEntropyIndex::search_modified` vs `ModifiedLinearEntropy`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineThresholdIndex, FlashEntropyIndex, FlashSearchResult,
    LinearCosine, LinearEntropy, ModifiedLinearCosine, ModifiedLinearEntropy, RandomSpectrumConfig,
    ScalarSimilarity, Spectrum, SpectrumAlloc, SpectrumMut, TopKSearchState,
};

type BenchSpectrum = mass_spectrometry::prelude::GenericSpectrum;

const LIBRARY_SIZE: usize = 1_000;
const QUERY_COUNT: usize = 32;
const TOP_K_LIBRARY_SIZE: usize = 50_000;
const TOP_K_QUERY_COUNT: usize = 32;
const TOP_K_CLUSTER_SIZE: usize = 256;
const RANDOM_BASE_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

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
    let n_peaks = 48 + (next_u64(&mut state) % 49) as usize; // [48, 96]
    let precursor_mz = 650.0 + (next_unit_f64(&mut state) * 550.0); // [650, 1200]
    let config = RandomSpectrumConfig {
        precursor_mz,
        n_peaks,
        mz_min: 50.0,
        mz_max: 550.0,
        min_peak_gap: 0.25, // > 2 * 0.1 tolerance
        intensity_min: 1.0,
        intensity_max: 1_000.0,
    };
    BenchSpectrum::random(config, seed).expect("random benchmark spectrum should build")
}

fn build_random_spectra(count: usize, seed: u64) -> Vec<BenchSpectrum> {
    let mut spectra = Vec::with_capacity(count);
    for i in 0..count {
        let item_seed = seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        spectra.push(random_spectrum_from_seed(item_seed));
    }
    spectra
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

fn build_clustered_spectra(count: usize, seed: u64) -> Vec<BenchSpectrum> {
    let mut spectra = Vec::with_capacity(count);
    let mut cluster_index = 0usize;

    while spectra.len() < count {
        let base_seed =
            seed.wrapping_add((cluster_index as u64).wrapping_mul(0xA076_1D64_78BD_642F));
        let base = random_spectrum_from_seed(base_seed);
        let remaining = count - spectra.len();
        let cluster_size = remaining.min(TOP_K_CLUSTER_SIZE);

        for variant_index in 0..cluster_size {
            let variant_seed =
                base_seed ^ (variant_index as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB);
            spectra.push(perturb_spectrum(&base, variant_seed));
        }

        cluster_index += 1;
    }

    spectra
}

fn cosine_without_index(
    scorer: &LinearCosine,
    query: &BenchSpectrum,
    library: &[BenchSpectrum],
) -> (f64, usize) {
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for library_spectrum in library {
        let (score, matches) = scorer
            .similarity(query, library_spectrum)
            .expect("cosine similarity should succeed");
        score_sum += score;
        match_sum += matches;
    }
    (score_sum, match_sum)
}

fn modified_cosine_without_index(
    scorer: &ModifiedLinearCosine,
    query: &BenchSpectrum,
    library: &[BenchSpectrum],
) -> (f64, usize) {
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for library_spectrum in library {
        let (score, matches) = scorer
            .similarity(query, library_spectrum)
            .expect("modified cosine similarity should succeed");
        score_sum += score;
        match_sum += matches;
    }
    (score_sum, match_sum)
}

fn entropy_without_index(
    scorer: &LinearEntropy,
    query: &BenchSpectrum,
    library: &[BenchSpectrum],
) -> (f64, usize) {
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for library_spectrum in library {
        let (score, matches) = scorer
            .similarity(query, library_spectrum)
            .expect("entropy similarity should succeed");
        score_sum += score;
        match_sum += matches;
    }
    (score_sum, match_sum)
}

fn modified_entropy_without_index(
    scorer: &ModifiedLinearEntropy,
    query: &BenchSpectrum,
    library: &[BenchSpectrum],
) -> (f64, usize) {
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for library_spectrum in library {
        let (score, matches) = scorer
            .similarity(query, library_spectrum)
            .expect("modified entropy similarity should succeed");
        score_sum += score;
        match_sum += matches;
    }
    (score_sum, match_sum)
}

fn cosine_with_index(index: &FlashCosineIndex, query: &BenchSpectrum) -> (f64, usize) {
    let results = index
        .search(query)
        .expect("flash cosine search should succeed");
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for result in &results {
        score_sum += result.score;
        match_sum += result.n_matches;
    }
    (score_sum, match_sum)
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

fn modified_cosine_with_index(index: &FlashCosineIndex, query: &BenchSpectrum) -> (f64, usize) {
    let results = index
        .search_modified(query)
        .expect("flash modified cosine search should succeed");
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for result in &results {
        score_sum += result.score;
        match_sum += result.n_matches;
    }
    (score_sum, match_sum)
}

fn entropy_with_index(index: &FlashEntropyIndex, query: &BenchSpectrum) -> (f64, usize) {
    let results = index
        .search(query)
        .expect("flash entropy search should succeed");
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for result in &results {
        score_sum += result.score;
        match_sum += result.n_matches;
    }
    (score_sum, match_sum)
}

fn modified_entropy_with_index(index: &FlashEntropyIndex, query: &BenchSpectrum) -> (f64, usize) {
    let results = index
        .search_modified(query)
        .expect("flash modified entropy search should succeed");
    let mut score_sum = 0.0;
    let mut match_sum = 0usize;
    for result in &results {
        score_sum += result.score;
        match_sum += result.n_matches;
    }
    (score_sum, match_sum)
}

fn bench_index_search(c: &mut Criterion) {
    let mz_power = 1.0_f64;
    let intensity_power = 1.0_f64;
    let mz_tolerance = 0.1_f64;

    let library = build_random_spectra(LIBRARY_SIZE, RANDOM_BASE_SEED);
    let queries = build_random_spectra(QUERY_COUNT, RANDOM_BASE_SEED ^ 0xA5A5_A5A5_1357_2468);

    let linear_cosine = LinearCosine::new(mz_power, intensity_power, mz_tolerance)
        .expect("valid linear cosine config");
    let modified_linear_cosine = ModifiedLinearCosine::new(mz_power, intensity_power, mz_tolerance)
        .expect("valid modified linear cosine config");
    let linear_entropy =
        LinearEntropy::weighted(mz_tolerance).expect("valid linear entropy config");
    let modified_linear_entropy = ModifiedLinearEntropy::weighted(mz_tolerance)
        .expect("valid modified linear entropy config");

    let flash_cosine =
        FlashCosineIndex::new(mz_power, intensity_power, mz_tolerance, library.iter())
            .expect("flash cosine index should build");
    let top_k = 8_usize;
    let flash_entropy =
        FlashEntropyIndex::new(0.0_f64, 1.0_f64, mz_tolerance, true, library.iter())
            .expect("flash entropy index should build");

    let mut cosine_group = c.benchmark_group("library_search_cosine");
    cosine_group.bench_function("with_index_flash", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) = cosine_with_index(&flash_cosine, black_box(query));
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    cosine_group.bench_function("without_index_linear_scan", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) =
                    cosine_without_index(&linear_cosine, black_box(query), black_box(&library));
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    cosine_group.finish();

    let top_k_library = build_clustered_spectra(TOP_K_LIBRARY_SIZE, RANDOM_BASE_SEED ^ 0xC05E_C05E);
    let flash_cosine_top_k = FlashCosineIndex::new(
        mz_power,
        intensity_power,
        mz_tolerance,
        top_k_library.iter(),
    )
    .expect("top-k flash cosine index should build");
    let flash_entropy_top_k =
        FlashEntropyIndex::new(0.0_f64, 1.0_f64, mz_tolerance, true, top_k_library.iter())
            .expect("top-k flash entropy index should build");

    let mut top_k_group = c.benchmark_group("library_search_cosine_top_k");
    top_k_group.sample_size(10);
    for top_k_threshold in [0.5_f64, 0.7_f64, 0.9_f64] {
        let flash_cosine_threshold_top_k = FlashCosineThresholdIndex::new(
            mz_power,
            intensity_power,
            mz_tolerance,
            top_k_threshold,
            top_k_library.iter(),
        )
        .expect("top-k flash threshold cosine index should build");

        let threshold_label = format!("{top_k_threshold:.1}");

        let mut state = flash_cosine_top_k.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        top_k_group.bench_function(
            BenchmarkId::new("direct_top_k_threshold", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for query in top_k_library.iter().take(TOP_K_QUERY_COUNT) {
                        flash_cosine_top_k
                            .for_each_top_k_threshold_with_state(
                                black_box(query),
                                black_box(top_k),
                                black_box(top_k_threshold),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("top-k threshold search should succeed");
                    }
                    black_box((total_score, total_matches))
                })
            },
        );

        let mut state = flash_cosine_top_k.new_search_state();
        top_k_group.bench_function(
            BenchmarkId::new("threshold_then_sort", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for query in top_k_library.iter().take(TOP_K_QUERY_COUNT) {
                        let results = flash_cosine_top_k
                            .search_threshold_with_state(
                                black_box(query),
                                top_k_threshold,
                                &mut state,
                            )
                            .expect("threshold search should succeed");
                        let results = ranked_top_k(results, top_k);
                        let (score, matches) = summarize_results(&results);
                        total_score += score;
                        total_matches += matches;
                    }
                    black_box((total_score, total_matches))
                })
            },
        );

        let mut state = flash_cosine_threshold_top_k.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        top_k_group.bench_function(
            BenchmarkId::new("threshold_index_indexed_top_k", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for query_id in 0..TOP_K_QUERY_COUNT {
                        flash_cosine_threshold_top_k
                            .for_each_top_k_indexed_with_state(
                                black_box(query_id as u32),
                                black_box(top_k),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("indexed top-k threshold search should succeed");
                    }
                    black_box((total_score, total_matches))
                })
            },
        );

        let mut state = flash_cosine_threshold_top_k.new_search_state();
        top_k_group.bench_function(
            BenchmarkId::new("threshold_index_emit_then_sort", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for query_id in 0..TOP_K_QUERY_COUNT {
                        let mut results = Vec::new();
                        flash_cosine_threshold_top_k
                            .for_each_indexed_with_state(
                                black_box(query_id as u32),
                                &mut state,
                                |hit| {
                                    results.push(hit);
                                },
                            )
                            .expect("indexed threshold search should succeed");
                        let results = ranked_top_k(results, top_k);
                        let (score, matches) = summarize_results(&results);
                        total_score += score;
                        total_matches += matches;
                    }
                    black_box((total_score, total_matches))
                })
            },
        );
    }
    top_k_group.finish();

    let mut entropy_top_k_group = c.benchmark_group("library_search_entropy_top_k");
    entropy_top_k_group.sample_size(10);
    for top_k_threshold in [0.5_f64, 0.7_f64, 0.9_f64] {
        let threshold_label = format!("{top_k_threshold:.1}");

        let mut state = flash_entropy_top_k.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        entropy_top_k_group.bench_function(
            BenchmarkId::new("direct_top_k_threshold", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for query in top_k_library.iter().take(TOP_K_QUERY_COUNT) {
                        flash_entropy_top_k
                            .for_each_top_k_threshold_with_state(
                                black_box(query),
                                black_box(top_k),
                                black_box(top_k_threshold),
                                &mut state,
                                &mut top_k_state,
                                |result| {
                                    total_score += result.score;
                                    total_matches += result.n_matches;
                                },
                            )
                            .expect("top-k entropy threshold search should succeed");
                    }
                    black_box((total_score, total_matches))
                })
            },
        );

        let mut state = flash_entropy_top_k.new_search_state();
        entropy_top_k_group.bench_function(
            BenchmarkId::new("threshold_then_sort", &threshold_label),
            |b| {
                b.iter(|| {
                    let mut total_score = 0.0;
                    let mut total_matches = 0usize;
                    for query in top_k_library.iter().take(TOP_K_QUERY_COUNT) {
                        let results = flash_entropy_top_k
                            .search_threshold_with_state(
                                black_box(query),
                                top_k_threshold,
                                &mut state,
                            )
                            .expect("entropy threshold search should succeed");
                        let results = ranked_top_k(results, top_k);
                        let (score, matches) = summarize_results(&results);
                        total_score += score;
                        total_matches += matches;
                    }
                    black_box((total_score, total_matches))
                })
            },
        );
    }
    entropy_top_k_group.finish();

    let mut modified_group = c.benchmark_group("library_search_modified_cosine");
    modified_group.bench_function("with_index_flash_modified", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) = modified_cosine_with_index(&flash_cosine, black_box(query));
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    modified_group.bench_function("without_index_modified_linear_scan", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) = modified_cosine_without_index(
                    &modified_linear_cosine,
                    black_box(query),
                    black_box(&library),
                );
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    modified_group.finish();

    let mut entropy_group = c.benchmark_group("library_search_entropy");
    entropy_group.bench_function("with_index_flash", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) = entropy_with_index(&flash_entropy, black_box(query));
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    entropy_group.bench_function("without_index_linear_scan", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) =
                    entropy_without_index(&linear_entropy, black_box(query), black_box(&library));
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    entropy_group.finish();

    let mut modified_entropy_group = c.benchmark_group("library_search_modified_entropy");
    modified_entropy_group.bench_function("with_index_flash_modified", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) =
                    modified_entropy_with_index(&flash_entropy, black_box(query));
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    modified_entropy_group.bench_function("without_index_modified_linear_scan", |b| {
        b.iter(|| {
            let mut total_score = 0.0;
            let mut total_matches = 0usize;
            for query in &queries {
                let (score, matches) = modified_entropy_without_index(
                    &modified_linear_entropy,
                    black_box(query),
                    black_box(&library),
                );
                total_score += score;
                total_matches += matches;
            }
            black_box((total_score, total_matches))
        })
    });
    modified_entropy_group.finish();
}

criterion_group!(benches, bench_index_search);
criterion_main!(benches);
