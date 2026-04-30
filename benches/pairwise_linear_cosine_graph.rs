//! Large thresholded graph benchmarks for linear cosine and entropy similarity.
//!
//! The graph use case is an upper-triangular all-pairs similarity scan where
//! only scores above a threshold become edges. For 24M nodes the full pair
//! space is ~288T pairs, so this benchmark reports logical pair throughput and
//! includes cosine indexed threshold search paths alongside all-pairs
//! baselines.
//!
//! Environment knobs:
//!
//! - `PAIRWISE_LINEAR_COSINE_NODES=1024,4096,16384`
//! - `PAIRWISE_LINEAR_COSINE_PUBLIC_API_NODES=256,512`
//! - `PAIRWISE_LINEAR_COSINE_THRESHOLDS=0.5,0.7,0.9`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineThresholdIndex, GenericSpectrum, LinearCosine, LinearEntropy,
    ScalarSimilarity, Spectrum, SpectrumMut,
};

const DEFAULT_NODE_COUNTS: &[usize] = &[1_024, 4_096];
const DEFAULT_PUBLIC_API_NODE_COUNTS: &[usize] = &[256];

const PEAKS_PER_SPECTRUM: usize = 64;
const TARGET_CLUSTER_SIZE: usize = 32;
const MZ_TOLERANCE: f64 = 0.05;
const MZ_POWER: f64 = 1.0;
const INTENSITY_POWER: f64 = 1.0;
const DEFAULT_SCORE_THRESHOLDS: &[f64] = &[0.5, 0.7, 0.9];
const RANDOM_BASE_SEED: u64 = 0xA4D1_0C05_1A57_5EED;

#[derive(Debug, Clone)]
struct PreparedSpectrum {
    mz: Vec<f64>,
    products: Vec<f64>,
    suffix_square_sum: Vec<f64>,
    norm: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct GraphStats {
    edges: usize,
    pruned_pairs: usize,
    matched_peaks: usize,
    score_sum: f64,
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

fn parse_node_counts(env_name: &str, default: &[usize]) -> Vec<usize> {
    let Ok(raw) = std::env::var(env_name) else {
        return default.to_vec();
    };

    let parsed: Vec<usize> = raw
        .split(',')
        .filter_map(|item| {
            let value = item.trim().parse::<usize>().ok()?;
            (value >= 2).then_some(value)
        })
        .collect();

    if parsed.is_empty() {
        default.to_vec()
    } else {
        parsed
    }
}

fn parse_score_thresholds(env_name: &str, default: &[f64]) -> Vec<f64> {
    let Ok(raw) = std::env::var(env_name) else {
        return default.to_vec();
    };

    let parsed: Vec<f64> = raw
        .split(',')
        .filter_map(|item| {
            let value = item.trim().parse::<f64>().ok()?;
            (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(value)
        })
        .collect();

    if parsed.is_empty() {
        default.to_vec()
    } else {
        parsed
    }
}

fn upper_triangle_pairs(n: usize) -> u64 {
    ((n as u64) * ((n - 1) as u64)) / 2
}

fn build_cluster_templates(cluster_count: usize) -> Vec<Vec<(f64, f64)>> {
    let mut templates = Vec::with_capacity(cluster_count);

    for cluster_id in 0..cluster_count {
        let mut state = nonzero_seed(
            RANDOM_BASE_SEED ^ ((cluster_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let mut mz = 50.0 + next_unit_f64(&mut state) * 10.0;
        let mut peaks = Vec::with_capacity(PEAKS_PER_SPECTRUM);

        for _ in 0..PEAKS_PER_SPECTRUM {
            mz += 3.0 + next_unit_f64(&mut state) * 4.0;
            let intensity = 100.0 + next_unit_f64(&mut state) * 9_900.0;
            peaks.push((mz, intensity));
        }

        templates.push(peaks);
    }

    templates
}

fn build_clustered_spectra(node_count: usize) -> Vec<GenericSpectrum> {
    let cluster_count = (node_count / TARGET_CLUSTER_SIZE).max(1);
    let templates = build_cluster_templates(cluster_count);
    let mut spectra = Vec::with_capacity(node_count);

    for spectrum_id in 0..node_count {
        let cluster_id = spectrum_id % cluster_count;
        let template = &templates[cluster_id];
        let mut state = nonzero_seed(
            RANDOM_BASE_SEED
                ^ ((spectrum_id as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
                ^ ((cluster_id as u64).wrapping_mul(0x94D0_49BB_1331_11EB)),
        );
        let precursor_mz = 750.0 + cluster_id as f64 + next_unit_f64(&mut state);
        let mut spectrum = GenericSpectrum::try_with_capacity(precursor_mz, template.len())
            .expect("benchmark spectrum precursor should be valid");

        for &(base_mz, base_intensity) in template {
            let mz_jitter = (next_unit_f64(&mut state) - 0.5) * MZ_TOLERANCE * 0.5;
            let intensity_scale = 0.8 + next_unit_f64(&mut state) * 0.4;
            spectrum
                .add_peak(base_mz + mz_jitter, base_intensity * intensity_scale)
                .expect("benchmark peaks should stay sorted and valid");
        }

        spectra.push(spectrum);
    }

    spectra
}

fn prepare_spectrum(spectrum: &GenericSpectrum) -> PreparedSpectrum {
    let mut mz = Vec::with_capacity(spectrum.len());
    let mut mz_components = Vec::with_capacity(spectrum.len());
    let mut intensity_components = Vec::with_capacity(spectrum.len());

    for (peak_mz, intensity) in spectrum.peaks() {
        mz.push(peak_mz);
        mz_components.push(peak_mz.powf(MZ_POWER));
        intensity_components.push(intensity.powf(INTENSITY_POWER));
    }

    let mz_max = mz_components.iter().copied().fold(0.0_f64, f64::max);
    let intensity_max = intensity_components.iter().copied().fold(0.0_f64, f64::max);

    if mz_max > 0.0 {
        for value in &mut mz_components {
            *value /= mz_max;
        }
    }
    if intensity_max > 0.0 {
        for value in &mut intensity_components {
            *value /= intensity_max;
        }
    }

    let mut products: Vec<f64> = mz_components
        .into_iter()
        .zip(intensity_components)
        .map(|(mz_component, intensity_component)| mz_component * intensity_component)
        .collect();

    let product_max = products.iter().copied().fold(0.0_f64, f64::max);
    if product_max > 0.0 {
        for product in &mut products {
            *product /= product_max;
        }
    }

    let norm = products
        .iter()
        .map(|&product| product * product)
        .sum::<f64>()
        .sqrt();

    let mut suffix_square_sum = vec![0.0_f64; products.len() + 1];
    for index in (0..products.len()).rev() {
        suffix_square_sum[index] = suffix_square_sum[index + 1] + products[index] * products[index];
    }

    PreparedSpectrum {
        mz,
        products,
        suffix_square_sum,
        norm,
    }
}

fn prepare_spectra(spectra: &[GenericSpectrum]) -> Vec<PreparedSpectrum> {
    spectra.iter().map(prepare_spectrum).collect()
}

#[inline]
fn thresholded_prepared_linear_cosine(
    left: &PreparedSpectrum,
    right: &PreparedSpectrum,
    mz_tolerance: f64,
    score_threshold: f64,
) -> Option<(f64, usize)> {
    let denominator = left.norm * right.norm;
    if denominator == 0.0 {
        return None;
    }

    let target_raw_score = score_threshold * denominator;
    let mut score_sum = 0.0_f64;
    let mut n_matches = 0usize;
    let mut right_index = 0usize;

    for (left_index, &left_mz) in left.mz.iter().enumerate() {
        while right_index < right.mz.len() && right.mz[right_index] < left_mz - mz_tolerance {
            right_index += 1;
        }

        if right_index < right.mz.len()
            && right.mz[right_index] >= left_mz - mz_tolerance
            && right.mz[right_index] <= left_mz + mz_tolerance
        {
            let product = left.products[left_index] * right.products[right_index];
            if product != 0.0 {
                score_sum += product;
                n_matches += 1;
            }
            right_index += 1;
        }

        let max_remaining_score =
            (left.suffix_square_sum[left_index + 1] * right.suffix_square_sum[right_index]).sqrt();
        if score_sum + max_remaining_score < target_raw_score {
            return None;
        }
    }

    let score = score_sum / denominator;
    if score >= score_threshold {
        Some((score, n_matches))
    } else {
        None
    }
}

fn thresholded_prepared_all_pairs(
    spectra: &[PreparedSpectrum],
    mz_tolerance: f64,
    score_threshold: f64,
) -> GraphStats {
    let mut stats = GraphStats::default();

    for left_index in 0..spectra.len() {
        let left = &spectra[left_index];
        for right in &spectra[(left_index + 1)..] {
            match thresholded_prepared_linear_cosine(left, right, mz_tolerance, score_threshold) {
                Some((score, matches)) => {
                    stats.edges += 1;
                    stats.matched_peaks += matches;
                    stats.score_sum += score;
                }
                None => {
                    stats.pruned_pairs += 1;
                }
            }
        }
    }

    stats
}

fn thresholded_public_api_all_pairs(
    spectra: &[GenericSpectrum],
    scorer: &LinearCosine,
    score_threshold: f64,
) -> GraphStats {
    let mut stats = GraphStats::default();

    for left_index in 0..spectra.len() {
        let left = &spectra[left_index];
        for right in &spectra[(left_index + 1)..] {
            let (score, matches) = scorer
                .similarity(left, right)
                .expect("benchmark spectra satisfy LinearCosine preconditions");
            if score >= score_threshold {
                stats.edges += 1;
                stats.matched_peaks += matches;
                stats.score_sum += score;
            }
        }
    }

    stats
}

fn thresholded_entropy_public_api_all_pairs(
    spectra: &[GenericSpectrum],
    scorer: &LinearEntropy,
    score_threshold: f64,
) -> GraphStats {
    let mut stats = GraphStats::default();

    for left_index in 0..spectra.len() {
        let left = &spectra[left_index];
        for right in &spectra[(left_index + 1)..] {
            let (score, matches) = scorer
                .similarity(left, right)
                .expect("benchmark spectra satisfy LinearEntropy preconditions");
            if score >= score_threshold {
                stats.edges += 1;
                stats.matched_peaks += matches;
                stats.score_sum += score;
            }
        }
    }

    stats
}

fn thresholded_flash_index_graph(
    spectra: &[GenericSpectrum],
    index: &FlashCosineIndex,
    score_threshold: f64,
) -> Result<GraphStats, mass_spectrometry::prelude::SimilarityComputationError> {
    let mut stats = GraphStats::default();
    let mut state = index.new_search_state();

    for (query_id, query) in spectra.iter().enumerate() {
        index.for_each_threshold_with_state(query, score_threshold, &mut state, |result| {
            if result.spectrum_id as usize > query_id {
                stats.edges += 1;
                stats.matched_peaks += result.n_matches;
                stats.score_sum += result.score;
            }
        })?;
    }

    Ok(stats)
}

fn thresholded_flash_threshold_index_graph(
    node_count: usize,
    index: &FlashCosineThresholdIndex,
) -> Result<GraphStats, mass_spectrometry::prelude::SimilarityComputationError> {
    let mut stats = GraphStats::default();
    let mut state = index.new_search_state();

    for query_id in 0..node_count {
        index.for_each_indexed_with_state(query_id as u32, &mut state, |result| {
            if result.spectrum_id as usize > query_id {
                stats.edges += 1;
                stats.matched_peaks += result.n_matches;
                stats.score_sum += result.score;
            }
        })?;
    }

    Ok(stats)
}

fn bench_pairwise_linear_cosine_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("thresholded_pairwise_linear_cosine_graph");
    group.sample_size(10);
    let score_thresholds = parse_score_thresholds(
        "PAIRWISE_LINEAR_COSINE_THRESHOLDS",
        DEFAULT_SCORE_THRESHOLDS,
    );

    for node_count in parse_node_counts("PAIRWISE_LINEAR_COSINE_NODES", DEFAULT_NODE_COUNTS) {
        let spectra = build_clustered_spectra(node_count);
        let prepared = prepare_spectra(&spectra);
        let logical_pairs = upper_triangle_pairs(node_count);

        let index = FlashCosineIndex::new(MZ_POWER, INTENSITY_POWER, MZ_TOLERANCE, spectra.iter())
            .expect("benchmark spectra should build a flash cosine index");

        for &score_threshold in &score_thresholds {
            group.throughput(Throughput::Elements(logical_pairs));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("prepared_all_pairs_threshold_{score_threshold:.2}"),
                    node_count,
                ),
                &prepared,
                |bench, prepared| {
                    bench.iter(|| {
                        thresholded_prepared_all_pairs(
                            black_box(prepared),
                            black_box(MZ_TOLERANCE),
                            black_box(score_threshold),
                        )
                    });
                },
            );

            group.throughput(Throughput::Elements(logical_pairs));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("flash_index_thresholded_{score_threshold:.2}"),
                    node_count,
                ),
                &(&spectra, &index),
                |bench, &(spectra, index)| {
                    bench.iter(|| {
                        thresholded_flash_index_graph(
                            black_box(spectra),
                            black_box(index),
                            black_box(score_threshold),
                        )
                        .expect("benchmark spectra satisfy FlashCosineIndex preconditions")
                    });
                },
            );

            let threshold_index = FlashCosineThresholdIndex::new(
                MZ_POWER,
                INTENSITY_POWER,
                MZ_TOLERANCE,
                score_threshold,
                spectra.iter(),
            )
            .expect("benchmark spectra should build a threshold flash cosine index");
            group.throughput(Throughput::Elements(logical_pairs));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("flash_threshold_index_by_id_{score_threshold:.2}"),
                    node_count,
                ),
                &threshold_index,
                |bench, threshold_index| {
                    bench.iter(|| {
                        thresholded_flash_threshold_index_graph(
                            black_box(node_count),
                            black_box(threshold_index),
                        )
                        .expect("benchmark spectra satisfy FlashCosineThresholdIndex preconditions")
                    });
                },
            );
        }
    }

    group.finish();

    let scorer = LinearCosine::new(MZ_POWER, INTENSITY_POWER, MZ_TOLERANCE)
        .expect("benchmark scorer config should be valid");
    let mut public_api_group =
        c.benchmark_group("thresholded_pairwise_linear_cosine_public_api_baseline");
    public_api_group.sample_size(10);

    for node_count in parse_node_counts(
        "PAIRWISE_LINEAR_COSINE_PUBLIC_API_NODES",
        DEFAULT_PUBLIC_API_NODE_COUNTS,
    ) {
        let spectra = build_clustered_spectra(node_count);
        for &score_threshold in &score_thresholds {
            public_api_group.throughput(Throughput::Elements(upper_triangle_pairs(node_count)));
            public_api_group.bench_with_input(
                BenchmarkId::new(format!("threshold_{score_threshold:.2}"), node_count),
                &spectra,
                |bench, spectra| {
                    bench.iter(|| {
                        thresholded_public_api_all_pairs(
                            black_box(spectra),
                            black_box(&scorer),
                            black_box(score_threshold),
                        )
                    });
                },
            );
        }
    }

    public_api_group.finish();

    let entropy_scorer =
        LinearEntropy::weighted(MZ_TOLERANCE).expect("benchmark entropy config should be valid");
    let mut entropy_public_api_group =
        c.benchmark_group("thresholded_pairwise_linear_entropy_public_api_baseline");
    entropy_public_api_group.sample_size(10);

    for node_count in parse_node_counts(
        "PAIRWISE_LINEAR_COSINE_PUBLIC_API_NODES",
        DEFAULT_PUBLIC_API_NODE_COUNTS,
    ) {
        let spectra = build_clustered_spectra(node_count);
        for &score_threshold in &score_thresholds {
            entropy_public_api_group
                .throughput(Throughput::Elements(upper_triangle_pairs(node_count)));
            entropy_public_api_group.bench_with_input(
                BenchmarkId::new(format!("threshold_{score_threshold:.2}"), node_count),
                &spectra,
                |bench, spectra| {
                    bench.iter(|| {
                        thresholded_entropy_public_api_all_pairs(
                            black_box(spectra),
                            black_box(&entropy_scorer),
                            black_box(score_threshold),
                        )
                    });
                },
            );
        }
    }

    entropy_public_api_group.finish();
}

criterion_group!(benches, bench_pairwise_linear_cosine_graph);
criterion_main!(benches);
