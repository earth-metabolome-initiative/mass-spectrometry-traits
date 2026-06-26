//! Quantifies how the LSH modified-cosine index's recall affects the t-SNE
//! embedding it feeds, on a synthetic dataset with known cluster labels.
//!
//! Raw recall@k counts every missed top-k neighbor equally, but t-SNE does not.
//! bhtsne calibrates a per-row Gaussian bandwidth so that `p_{j|i} propto
//! exp(-beta_i * d_ij^2)` with the row's Shannon entropy equal to
//! `ln(perplexity)`, where `d_ij = arccos(sim_ij)`. Neighbors are therefore
//! weighted by an exponentially decaying kernel: a missed neighbor in the
//! low-similarity tail carries almost no affinity mass, while a missed
//! high-similarity neighbor is expensive. The metric that matters is the
//! fraction of affinity mass preserved, not the fraction of the top-k preserved.
//!
//! This example builds an exact `FlashCosineIndex` and an LSH
//! `FlashCosineSketchIndex`, computes both neighbor sets for every spectrum, and
//! reports:
//!   - raw recall@k against the exact modified top-k,
//!   - recall split by similarity-rank band (head versus tail),
//!   - mean similarity of recovered versus missed neighbors,
//!   - affinity mass recovered: the share of each exact P row's mass that lands
//!     on neighbors the LSH row also keeps (replicating bhtsne's beta search),
//!   - P-row overlap: `sum_j min(p_exact_j, p_lsh_j)`, which also captures the
//!     bandwidth recalibration when LSH substitutes worse neighbors,
//!   - end-to-end embedding quality for both neighbor sets: cluster-label purity
//!     among the 2D nearest neighbors, layout agreement between the two
//!     embeddings, and how well each layout preserves the true high-D neighbors.
//!
//! Run:
//!
//! ```text
//! cargo run --release --example tsne_recall_effect --features bhtsne,minhash
//! LIBRARY_SIZE=6400 CLUSTER_SIZE=64 PERPLEXITY=30 EPOCHS=250 \
//!   cargo run --release --example tsne_recall_effect --features bhtsne,minhash
//! ```
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineSketchIndex, GenericSpectrum, LshSketcher, MinHash,
    ModifiedLinearCosine, RandomSpectrumConfig, SpectraIndexBuilder, SpectralTsne, Spectrum,
    SpectrumAlloc, SpectrumMut, TopKSearchState,
};
use rayon::prelude::*;

type Spec = GenericSpectrum;

const MZ_POWER: f64 = 0.0;
const INTENSITY_POWER: f64 = 0.25;
const MZ_TOLERANCE: f64 = 0.02;
const RANDOM_BASE_SEED: u64 = 0x5151_5151_2323_2323;
/// Same MinHash configuration as the index default (`u32` words, 128
/// permutations) with 16 bands.
const BANDS: usize = 16;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| *v > 0.0)
        .unwrap_or(default)
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

fn random_spectrum_from_seed(seed: u64) -> Spec {
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
    Spec::random(config, seed).expect("random spectrum should build")
}

/// A near-duplicate of `template`: same precursor, small m/z and intensity
/// jitter that preserves sorted, well-separated peaks.
fn perturb_spectrum(template: &Spec, seed: u64) -> Spec {
    let mut state = nonzero_seed(seed);
    let mut spectrum = Spec::with_capacity(template.precursor_mz(), template.len())
        .expect("perturbed spectrum should allocate");
    for (mz, intensity) in template.peaks() {
        let mz_jitter = (next_unit_f64(&mut state) - 0.5) * 0.01;
        let intensity_scale = 0.9 + next_unit_f64(&mut state) * 0.2;
        spectrum
            .add_peak(mz + mz_jitter, intensity * intensity_scale)
            .expect("small perturbations preserve sorted, well-separated peaks");
    }
    spectrum
}

/// Clustered spectra plus a per-spectrum cluster label. Each cluster is one
/// random template plus `cluster_size - 1` near-duplicates.
fn build_clustered_spectra(
    count: usize,
    cluster_size: usize,
    seed: u64,
) -> (Vec<Spec>, Vec<usize>) {
    let mut spectra = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);
    let mut cluster_index = 0usize;
    let cluster_size = cluster_size.max(1);

    while spectra.len() < count {
        let base_seed =
            seed.wrapping_add((cluster_index as u64).wrapping_mul(0xA076_1D64_78BD_642F));
        let base = random_spectrum_from_seed(base_seed);
        let remaining = count - spectra.len();
        let current = remaining.min(cluster_size);
        for variant_index in 0..current {
            let variant_seed =
                base_seed ^ (variant_index as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB);
            spectra.push(perturb_spectrum(&base, variant_seed));
            labels.push(cluster_index);
        }
        cluster_index += 1;
    }

    (spectra, labels)
}

/// Modified-cosine similarity to the metric distance bhtsne consumes.
#[inline]
fn metric_distance(similarity: f64) -> f64 {
    similarity.clamp(-1.0, 1.0).acos()
}

/// Faithful port of bhtsne 0.5.10 `search_beta`: find the Gaussian precision
/// `beta = 1 / (2 sigma^2)` whose conditional distribution over `distances` has
/// Shannon entropy `ln(perplexity)`, then return the row-normalized `p_{j|i}`.
fn perplexity_row(distances: &[f64], perplexity: f64) -> Vec<f64> {
    let len = distances.len();
    let mut p = vec![0.0f64; len];
    if len == 0 {
        return p;
    }
    let target = perplexity.ln();
    let tolerance = 1e-5f64;
    let mut beta = 1.0f64;
    let mut min_beta = 0.0f64;
    let mut max_beta = 0.0f64;
    let mut min_set = false;
    let mut max_set = false;
    let mut sum = 0.0f64;

    for _ in 0..200 {
        sum = 0.0;
        for (pj, &d) in p.iter_mut().zip(distances) {
            *pj = (-beta * d * d).exp();
            sum += *pj;
        }
        let safe_sum = if sum > 0.0 { sum } else { f64::MIN_POSITIVE };
        let mut entropy = 0.0f64;
        for (&pj, &d) in p.iter().zip(distances) {
            entropy += beta * pj * d * d;
        }
        entropy = entropy / safe_sum + safe_sum.ln();
        let diff = entropy - target;
        if diff.abs() < tolerance {
            break;
        }
        if diff > 0.0 {
            min_beta = beta;
            min_set = true;
            beta = if max_set {
                (beta + max_beta) / 2.0
            } else {
                beta * 2.0
            };
        } else {
            max_beta = beta;
            max_set = true;
            beta = if min_set {
                (beta + min_beta) / 2.0
            } else {
                beta / 2.0
            };
        }
    }

    let denom = sum + f64::EPSILON;
    for pj in &mut p {
        *pj /= denom;
    }
    p
}

/// Indices of the `m` nearest 2D points to each point (Euclidean), self
/// excluded.
fn nearest_2d(embedding: &[[f64; 2]], m: usize) -> Vec<Vec<usize>> {
    let m = m.min(embedding.len().saturating_sub(1));
    (0..embedding.len())
        .into_par_iter()
        .map(|i| {
            let here = embedding[i];
            let mut scored: Vec<(f64, usize)> = embedding
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, p)| {
                    let dx = here[0] - p[0];
                    let dy = here[1] - p[1];
                    (dx * dx + dy * dy, j)
                })
                .collect();
            if m < scored.len() {
                scored.select_nth_unstable_by(m, |a, b| a.0.total_cmp(&b.0));
                scored.truncate(m);
            }
            scored.into_iter().map(|(_, j)| j).collect()
        })
        .collect()
}

/// Mean over points of the fraction of the `m` nearest 2D neighbors sharing the
/// point's true cluster label.
fn label_purity(neighbors: &[Vec<usize>], labels: &[usize]) -> f64 {
    let total: f64 = neighbors
        .iter()
        .enumerate()
        .map(|(i, row)| {
            if row.is_empty() {
                return 0.0;
            }
            let same = row.iter().filter(|&&j| labels[j] == labels[i]).count();
            same as f64 / row.len() as f64
        })
        .sum();
    total / neighbors.len().max(1) as f64
}

/// Mean over points of the fraction of the `m` nearest 2D neighbors that also
/// belong to `truth_sets[i]` (the exact high-D top-k of point `i`).
fn high_d_preservation(neighbors: &[Vec<usize>], truth_sets: &[HashSet<u32>]) -> f64 {
    let total: f64 = neighbors
        .iter()
        .enumerate()
        .map(|(i, row)| {
            if row.is_empty() {
                return 0.0;
            }
            let kept = row
                .iter()
                .filter(|&&j| truth_sets[i].contains(&(j as u32)))
                .count();
            kept as f64 / row.len() as f64
        })
        .sum();
    total / neighbors.len().max(1) as f64
}

/// Mean over points of the overlap between two layouts' `m` nearest 2D
/// neighbors.
fn layout_agreement(a: &[Vec<usize>], b: &[Vec<usize>]) -> f64 {
    let total: f64 = a
        .iter()
        .zip(b)
        .map(|(ra, rb)| {
            if ra.is_empty() {
                return 0.0;
            }
            let set: HashSet<usize> = rb.iter().copied().collect();
            let shared = ra.iter().filter(|&&j| set.contains(&j)).count();
            shared as f64 / ra.len() as f64
        })
        .sum();
    total / a.len().max(1) as f64
}

/// Mean recall@k of the LSH neighbor sets against the exact modified top-k.
fn raw_recall(exact_neighbors: &[Vec<(u32, f64)>], lsh_sets: &[HashSet<u32>]) -> f64 {
    let mut sum = 0.0;
    for (exact_row, lsh_set) in exact_neighbors.iter().zip(lsh_sets) {
        if exact_row.is_empty() {
            continue;
        }
        let recovered = exact_row
            .iter()
            .filter(|(id, _)| lsh_set.contains(id))
            .count();
        sum += recovered as f64 / exact_row.len() as f64;
    }
    sum / exact_neighbors.len().max(1) as f64
}

/// Replicates the bhtsne P rows for both neighbor sets and returns the affinity
/// mass of the exact row recovered by LSH and the full P-row overlap
/// (`sum_j min(p_exact, p_lsh)`, which also captures the bandwidth
/// recalibration when LSH substitutes weaker neighbors).
fn affinity_metrics(
    exact_neighbors: &[Vec<(u32, f64)>],
    lsh_neighbors: &[Vec<(u32, f64)>],
    lsh_sets: &[HashSet<u32>],
    k: usize,
    perplexity: f64,
) -> (f64, f64) {
    let neutral = std::f64::consts::FRAC_PI_2 * 1.0e3;
    let n = exact_neighbors.len();
    let (mass_sum, overlap_sum): (f64, f64) = (0..n)
        .into_par_iter()
        .map(|i| {
            let exact_row = &exact_neighbors[i];
            if exact_row.is_empty() {
                return (0.0, 0.0);
            }
            // Fixed-length-k rows exactly as build_neighbor_rows feeds bhtsne:
            // real neighbors at arccos(sim), missing slots at the neutral
            // distance (affinity underflows to zero there).
            let mut exact_dist: Vec<f64> =
                exact_row.iter().map(|&(_, s)| metric_distance(s)).collect();
            exact_dist.resize(k, neutral);
            let lsh_row = &lsh_neighbors[i];
            let mut lsh_dist: Vec<f64> = lsh_row.iter().map(|&(_, s)| metric_distance(s)).collect();
            lsh_dist.resize(k, neutral);

            let p_exact = perplexity_row(&exact_dist, perplexity);
            let p_lsh = perplexity_row(&lsh_dist, perplexity);

            let lsh_set = &lsh_sets[i];
            let mass: f64 = exact_row
                .iter()
                .zip(&p_exact)
                .filter(|((id, _), _)| lsh_set.contains(id))
                .map(|(_, &p)| p)
                .sum();
            let lsh_p: HashMap<u32, f64> = lsh_row
                .iter()
                .zip(&p_lsh)
                .map(|(&(id, _), &p)| (id, p))
                .collect();
            let overlap: f64 = exact_row
                .iter()
                .zip(&p_exact)
                .filter_map(|(&(id, _), &pe)| lsh_p.get(&id).map(|&pl| pe.min(pl)))
                .sum();
            (mass, overlap)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (mass_sum / n as f64, overlap_sum / n as f64)
}

/// Shared, type-agnostic context for evaluating one LSH config.
struct EvalContext<'a> {
    library: &'a [Spec],
    k: usize,
    perplexity: f64,
    m_eval: usize,
    exact_neighbors: &'a [Vec<(u32, f64)>],
    labels: &'a [usize],
    truth_sets: &'a [HashSet<u32>],
    nn_exact: &'a [Vec<usize>],
    scorer: &'a ModifiedLinearCosine,
    exact_search_secs: f64,
    tsne: &'a SpectralTsne,
}

/// Builds one MinHash-LSH config, computes its neighbor sets, runs the full
/// embedding, and prints recall, affinity mass, P-row overlap, candidate volume,
/// and embedding quality (cluster-label purity, high-D preservation, layout
/// agreement versus the exact embedding).
fn eval_lsh<K: LshSketcher + Sync>(
    ctx: &EvalContext<'_>,
    word: &str,
    perms: usize,
    bands: usize,
    word_bytes: usize,
) {
    let library = ctx.library;
    let k = ctx.k;
    let n = library.len();
    let build_start = Instant::now();
    let lsh = FlashCosineSketchIndex::<f64, K>::build_with_bands(
        library,
        MZ_POWER,
        INTENSITY_POWER,
        MZ_TOLERANCE,
        bands,
    )
    .expect("LSH index should build");
    let build_secs = build_start.elapsed().as_secs_f64();
    let search_start = Instant::now();
    let pairs: Vec<(Vec<(u32, f64)>, usize)> = (0..n)
        .into_par_iter()
        .map_init(
            || (lsh.new_search_state(), TopKSearchState::new()),
            |(state, top_k_state), i| {
                let mut hits: Vec<(u32, f64)> = Vec::new();
                lsh.for_each_modified_top_k_with_state(
                    &library[i],
                    k + 1,
                    state,
                    top_k_state,
                    |result| hits.push((result.spectrum_id, result.score)),
                )
                .expect("LSH search should succeed");
                hits.retain(|&(id, _)| id as usize != i);
                hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                hits.truncate(k);
                let candidates = state.diagnostics().candidates_rescored;
                (hits, candidates)
            },
        )
        .collect();
    let search_secs = search_start.elapsed().as_secs_f64();
    let candidates = pairs.iter().map(|(_, c)| *c as f64).sum::<f64>() / n as f64;
    let neighbors: Vec<Vec<(u32, f64)>> = pairs.into_iter().map(|(row, _)| row).collect();
    let sets: Vec<HashSet<u32>> = neighbors
        .iter()
        .map(|row| row.iter().map(|&(id, _)| id).collect())
        .collect();
    let recall = raw_recall(ctx.exact_neighbors, &sets);
    let (mass, overlap) =
        affinity_metrics(ctx.exact_neighbors, &neighbors, &sets, k, ctx.perplexity);
    let embed = ctx
        .tsne
        .embed_from_neighbors(&neighbors, ctx.scorer)
        .expect("embedding should run");
    let nn = nearest_2d(&embed, ctx.m_eval);
    let purity = label_purity(&nn, ctx.labels);
    let preserve = high_d_preservation(&nn, ctx.truth_sets);
    let agree = layout_agreement(ctx.nn_exact, &nn);
    let bytes = perms * word_bytes;
    let speedup = ctx.exact_search_secs / search_secs.max(1e-9);
    let query_us = search_secs * 1.0e6 / n as f64;
    println!(
        "{word:<4} {perms:>3}p/{bands:<2}b {bytes:>5}B  recall={recall:.3} aff_mass={mass:.3} overlap={overlap:.3} purity={purity:.3} preserve={preserve:.3} agree={agree:.3}  cand/q={candidates:.1} build={build_secs:.2}s search={search_secs:.3}s {query_us:.1}us/q {speedup:.0}x"
    );
}

/// Thin wrapper so each call names only the sketch type and config. `$ctx` is
/// forwarded as an expression so it resolves at the call site.
macro_rules! sweep {
    ($ctx:expr, $word:literal, $w:ty, $perms:literal, $bands:literal) => {
        eval_lsh::<MinHash<$w, $perms>>($ctx, $word, $perms, $bands, size_of::<$w>());
    };
}

fn main() {
    let library_size = env_usize("LIBRARY_SIZE", 6_400);
    let cluster_size = env_usize("CLUSTER_SIZE", 64);
    let perplexity = env_f64("PERPLEXITY", 30.0);
    let epochs = env_usize("EPOCHS", 250);
    let m_eval = env_usize("M_EVAL", 10);
    let k = ((3.0 * perplexity) as usize).clamp(1, library_size - 1);

    println!(
        "config: library={library_size} clusters_of={cluster_size} perplexity={perplexity} \
         k={k} epochs={epochs} eval_m={m_eval} sketch=MinHash<u32,128> bands={BANDS}"
    );

    let (library, labels) = build_clustered_spectra(library_size, cluster_size, RANDOM_BASE_SEED);
    let n = library.len();

    // Exact modified-cosine index (the ground truth) and the LSH sketch index.
    let exact = FlashCosineIndex::<f64>::builder()
        .mz_power(MZ_POWER)
        .intensity_power(INTENSITY_POWER)
        .mz_tolerance(MZ_TOLERANCE)
        .parallel()
        .build(&library)
        .expect("exact index should build");
    let lsh = FlashCosineSketchIndex::<f64, MinHash<u32, 128>>::build_with_bands(
        &library,
        MZ_POWER,
        INTENSITY_POWER,
        MZ_TOLERANCE,
        BANDS,
    )
    .expect("LSH index should build");

    // Per-spectrum neighbor rows as (id, similarity), descending, self dropped.
    let exact_search_start = Instant::now();
    let exact_neighbors: Vec<Vec<(u32, f64)>> = (0..n)
        .into_par_iter()
        .map_init(
            || exact.new_search_state(),
            |state, i| {
                let mut hits = exact
                    .search_modified_top_k_with_state(&library[i], k + 1, state)
                    .expect("exact search should succeed");
                hits.retain(|h| h.spectrum_id as usize != i);
                hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
                hits.truncate(k);
                hits.into_iter().map(|h| (h.spectrum_id, h.score)).collect()
            },
        )
        .collect();
    let exact_search_secs = exact_search_start.elapsed().as_secs_f64();

    let lsh_neighbors: Vec<Vec<(u32, f64)>> = (0..n)
        .into_par_iter()
        .map_init(
            || (lsh.new_search_state(), TopKSearchState::new()),
            |(state, top_k_state), i| {
                let mut hits: Vec<(u32, f64)> = Vec::new();
                lsh.for_each_modified_top_k_with_state(
                    &library[i],
                    k + 1,
                    state,
                    top_k_state,
                    |result| hits.push((result.spectrum_id, result.score)),
                )
                .expect("LSH search should succeed");
                hits.retain(|&(id, _)| id as usize != i);
                hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                hits.truncate(k);
                hits
            },
        )
        .collect();

    // ---- Neighbor-level recall and its similarity-rank structure ----
    let lsh_sets: Vec<HashSet<u32>> = lsh_neighbors
        .iter()
        .map(|row| row.iter().map(|&(id, _)| id).collect())
        .collect();

    // Rank bands over the exact top-k: head to tail.
    let band_edges = [0usize, 10, 30, 60, usize::MAX];
    let band_label = ["rank 0-9", "rank 10-29", "rank 30-59", "rank 60+"];
    let n_bands = band_label.len();
    let mut band_total = [0u64; 4];
    let mut band_recovered = [0u64; 4];

    let mut recall_sum = 0.0f64;
    let mut recovered_sim_sum = 0.0f64;
    let mut recovered_sim_count = 0u64;
    let mut missed_sim_sum = 0.0f64;
    let mut missed_sim_count = 0u64;

    for (i, exact_row) in exact_neighbors.iter().enumerate() {
        if exact_row.is_empty() {
            continue;
        }
        let lsh_set = &lsh_sets[i];
        let mut recovered = 0usize;
        for (rank, &(id, sim)) in exact_row.iter().enumerate() {
            let band = (0..n_bands)
                .find(|&b| rank >= band_edges[b] && rank < band_edges[b + 1])
                .unwrap_or(n_bands - 1);
            band_total[band] += 1;
            if lsh_set.contains(&id) {
                recovered += 1;
                band_recovered[band] += 1;
                recovered_sim_sum += sim;
                recovered_sim_count += 1;
            } else {
                missed_sim_sum += sim;
                missed_sim_count += 1;
            }
        }
        recall_sum += recovered as f64 / exact_row.len() as f64;
    }
    let raw_recall = recall_sum / n as f64;

    // ---- Affinity-level recall: replicate the bhtsne P rows ----
    let (affinity_mass, row_overlap) =
        affinity_metrics(&exact_neighbors, &lsh_neighbors, &lsh_sets, k, perplexity);

    // ---- End-to-end embeddings from each neighbor set ----
    let scorer = ModifiedLinearCosine::new(MZ_POWER, INTENSITY_POWER, MZ_TOLERANCE)
        .expect("scorer config should build");
    let tsne = SpectralTsne::new()
        .perplexity(perplexity)
        .epochs(epochs)
        .mz_tolerance(MZ_TOLERANCE);
    let embed_exact = tsne
        .embed_from_neighbors(&exact_neighbors, &scorer)
        .expect("exact embedding should run");
    let embed_lsh = tsne
        .embed_from_neighbors(&lsh_neighbors, &scorer)
        .expect("LSH embedding should run");

    let nn_exact = nearest_2d(&embed_exact, m_eval);
    let nn_lsh = nearest_2d(&embed_lsh, m_eval);
    let truth_sets: Vec<HashSet<u32>> = exact_neighbors
        .iter()
        .map(|row| row.iter().map(|&(id, _)| id).collect())
        .collect();

    let purity_exact = label_purity(&nn_exact, &labels);
    let purity_lsh = label_purity(&nn_lsh, &labels);
    let preserve_exact = high_d_preservation(&nn_exact, &truth_sets);
    let preserve_lsh = high_d_preservation(&nn_lsh, &truth_sets);
    let agreement = layout_agreement(&nn_exact, &nn_lsh);

    // Within a tight cluster of near-duplicates, the "m nearest in 2D" are an
    // arbitrary m of the cluster's mates, so two independent layouts overlap
    // only near this chance level even when both recover the cluster perfectly.
    let mut cluster_sizes: HashMap<usize, usize> = HashMap::new();
    for &label in &labels {
        *cluster_sizes.entry(label).or_insert(0) += 1;
    }
    let chance_agreement: f64 = labels
        .iter()
        .map(|label| {
            let avail = cluster_sizes[label].saturating_sub(1);
            if avail == 0 {
                0.0
            } else {
                m_eval.min(avail) as f64 / avail as f64
            }
        })
        .sum::<f64>()
        / n.max(1) as f64;

    // ---- Report ----
    let recovered_sim = recovered_sim_sum / recovered_sim_count.max(1) as f64;
    let missed_sim = missed_sim_sum / missed_sim_count.max(1) as f64;

    println!("\n--- neighbor recall ---");
    println!("raw recall@k:                 {raw_recall:.4}");
    println!("mean similarity recovered:    {recovered_sim:.4}  (n={recovered_sim_count})");
    println!("mean similarity missed:       {missed_sim:.4}  (n={missed_sim_count})");
    println!("recall by exact-rank band (head is highest similarity):");
    for b in 0..n_bands {
        let total = band_total[b];
        let recall = band_recovered[b] as f64 / total.max(1) as f64;
        println!(
            "  {:<11} recall={recall:.4}  (truth n={total})",
            band_label[b]
        );
    }

    println!("\n--- affinity (the t-SNE P matrix) ---");
    println!(
        "affinity mass recovered:      {affinity_mass:.4}  (exact-row P mass on kept neighbors)"
    );
    println!(
        "P-row overlap:                {row_overlap:.4}  (sum_j min(p_exact, p_lsh), with beta recalibration)"
    );

    println!("\n--- embedding quality (lower-bound exact vs LSH) ---");
    println!("cluster-label purity @{m_eval}:    exact={purity_exact:.4}  lsh={purity_lsh:.4}");
    println!("high-D neighbor preservation: exact={preserve_exact:.4}  lsh={preserve_lsh:.4}");
    println!("layout agreement (exact vs lsh) @{m_eval}: {agreement:.4}");
    println!(
        "layout agreement chance baseline:        {chance_agreement:.4}  (arbitrary within-cluster ordering)"
    );

    println!("\n--- sweep: how low can the sketch go (full embedding per config) ---");
    println!("word perms/bands bytes  |  recall aff_mass overlap cand/q  |  purity preserve agree");
    println!(
        "exact baseline search: {exact_search_secs:.3}s ({:.1}us/q over {n} queries)",
        exact_search_secs * 1.0e6 / n as f64
    );
    let ctx = EvalContext {
        library: &library,
        k,
        perplexity,
        m_eval,
        exact_neighbors: &exact_neighbors,
        labels: &labels,
        truth_sets: &truth_sets,
        nn_exact: &nn_exact,
        scorer: &scorer,
        tsne: &tsne,
        exact_search_secs,
    };
    sweep!(&ctx, "u32", u32, 256, 32);
    sweep!(&ctx, "u32", u32, 128, 16);
    sweep!(&ctx, "u32", u32, 64, 8);
    sweep!(&ctx, "u32", u32, 32, 4);
    sweep!(&ctx, "u32", u32, 16, 2);
    sweep!(&ctx, "u16", u16, 128, 16);
    sweep!(&ctx, "u16", u16, 64, 8);
    sweep!(&ctx, "u16", u16, 32, 4);
    sweep!(&ctx, "u16", u16, 16, 2);

    // Optional CSV for visual inspection.
    if let Ok(path) = std::env::var("TSNE_RECALL_CSV") {
        let mut csv = String::from("index,label,x_exact,y_exact,x_lsh,y_lsh\n");
        for i in 0..n {
            csv.push_str(&format!(
                "{i},{},{},{},{},{}\n",
                labels[i], embed_exact[i][0], embed_exact[i][1], embed_lsh[i][0], embed_lsh[i][1]
            ));
        }
        std::fs::write(&path, csv).expect("CSV write should succeed");
        println!("\nwrote embeddings to {path}");
    }
}
