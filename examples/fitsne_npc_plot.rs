//! Embeds the harmonized MS2 dataset with the GPU `fitsne` (CubeCL CUDA runtime, O(N) FFT
//! repulsion), from a Flash-cosine k-NN graph, and writes `x,y,npc_pathway,npc_class` to CSV so the
//! layout can be plotted colored by NPClassifier pathway or class.
//!
//! Pipeline: load harmonized top-60 via mascot-rs, optionally take a stratified subsample (balanced
//! across NPC pathways so one pathway does not dominate), clean, build a k-NN graph with the Flash
//! index (direct or modified cosine), convert each similarity to a geodesic distance `arccos(sim)`,
//! then run `fitsne::FitSne::<CudaRuntime>` with `fft_repulsion` (the O(N) interpolation-plus-FFT
//! path).
//!
//! ```text
//! METRIC=modified SUBSAMPLE=40000 INTENSITY_POWER=0.25 LR=4000 EPOCHS=1000 \
//!   cargo run --release --example fitsne_npc_plot --features rayon
//! ```
//!
//! Env knobs: `METRIC` (`direct` or `modified`, default `direct`), `SUBSAMPLE` (stratified target
//! size, 0 = full set, default 0), `STRATIFY_BY` (`pathway` or `class`, default `pathway`), `SEED`
//! (subsample RNG, default 1), `LIMIT` (raw spectra cap before subsampling, 0 = all, default 0),
//! `EPOCHS` (default 500), `PERPLEXITY` (default 30), `K` (neighbors per point, default
//! 3*perplexity), `BOXES` (FFT boxes per dim, default 128), `MZ_TOLERANCE` (default 0.05),
//! `INTENSITY_POWER` (cosine intensity exponent, default 1.0), `LR` (learning rate, default 200),
//! `CACHE` (1/0, default 1), `OUT` (csv path, default `examples/fitsne_npc.csv`).

use std::io::Write;
use std::time::Instant;

use fitsne::cubecl::cuda::CudaRuntime;
use fitsne::{FitSne, Neighbor};
use mascot_rs::prelude::{AnnotatedMs2Builder, MascotGenericFormat};
use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashEntropyIndex, GenericSpectrum, SiriusMergeClosePeaks, SpectraIndex,
    SpectraIndexBuilder, SpectralProcessor, Spectrum, SpectrumFloat, SpectrumMut,
};
use rayon::prelude::*;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn fmt(d: f64) -> String {
    if d >= 1.0 {
        format!("{d:.1} s")
    } else {
        format!("{:.1} ms", d * 1_000.0)
    }
}

/// SplitMix64, for deterministic per-stratum ordering.
fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Primary NPClassifier label of a spectrum from an MGF header field (first of a `A|B|C` list).
fn primary_label(spectrum: &MascotGenericFormat<f32>, key: &str) -> String {
    spectrum
        .metadata()
        .arbitrary_metadata_value(key)
        .unwrap_or("unknown")
        .split('|')
        .next()
        .unwrap_or("unknown")
        .trim()
        .to_string()
}

/// Stratified subsample of `0..all.len()`: group indices by `strata[i]`, then take up to
/// `ceil(target / n_strata)` from each group, picked deterministically by a seeded hash so big
/// strata (e.g. Alkaloids) are capped and small ones kept whole. Returns selected indices, sorted.
fn stratified_indices(strata: &[String], target: usize, seed: u64) -> Vec<usize> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, s) in strata.iter().enumerate() {
        groups.entry(s.as_str()).or_default().push(i);
    }
    let quota = target.div_ceil(groups.len().max(1));
    let mut selected = Vec::with_capacity(target);
    for (_, mut idxs) in groups {
        idxs.sort_by_key(|&i| mix(i as u64 ^ seed));
        idxs.truncate(quota);
        selected.extend(idxs);
    }
    selected.sort_unstable();
    selected
}

/// Serializes the neighbor graph to a compact little-endian binary.
fn save_graph(path: &str, graph: &[Vec<Neighbor<f32>>]) {
    let file = std::fs::File::create(path).expect("should create graph cache");
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&(graph.len() as u64).to_le_bytes()).unwrap();
    for row in graph {
        w.write_all(&(row.len() as u32).to_le_bytes()).unwrap();
        for nb in row {
            w.write_all(&(nb.index as u32).to_le_bytes()).unwrap();
            w.write_all(&nb.distance.to_le_bytes()).unwrap();
        }
    }
    w.flush().expect("flush graph cache");
}

/// Reads back a graph written by [`save_graph`].
fn load_graph(path: &str) -> Vec<Vec<Neighbor<f32>>> {
    let bytes = std::fs::read(path).expect("should read graph cache");
    let mut o = 0usize;
    let mut take = |len: usize| {
        let s = &bytes[o..o + len];
        o += len;
        s
    };
    let n = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
    let mut graph = Vec::with_capacity(n);
    for _ in 0..n {
        let count = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
        let mut row = Vec::with_capacity(count);
        for _ in 0..count {
            let index = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
            let distance = f32::from_le_bytes(take(4).try_into().unwrap());
            row.push(Neighbor::new(index, distance));
        }
        graph.push(row);
    }
    graph
}

/// Builds the k-NN graph from any Flash index (cosine or entropy). For each spectrum, the top-k
/// non-self hits, with each similarity mapped to a distance: `arccos(sim)` for cosine (the angular
/// metric) or `sqrt(1 - sim)` for entropy (the Jensen-Shannon metric), matching the crate's
/// `SpectralDistanceMetric`. `modified` selects the neutral-loss search.
fn build_graph<I: SpectraIndex + Sync>(
    index: &I,
    cleaned: &[GenericSpectrum<f32>],
    k: usize,
    modified: bool,
    entropy: bool,
) -> Vec<Vec<Neighbor<f32>>> {
    (0..cleaned.len())
        .into_par_iter()
        .map(|i| {
            let hits = if modified {
                index.search_modified_top_k(&cleaned[i], k + 1)
            } else {
                index.search_top_k(&cleaned[i], k + 1)
            }
            .expect("search should succeed");
            hits.into_iter()
                .filter(|hit| hit.spectrum_id as usize != i)
                .take(k)
                .map(|hit| {
                    let sim = hit.score.clamp(0.0, 1.0);
                    let distance = if entropy {
                        (1.0 - sim).sqrt()
                    } else {
                        sim.acos()
                    } as f32;
                    Neighbor::new(hit.spectrum_id as usize, distance)
                })
                .collect()
        })
        .collect()
}

/// Owns any spectrum as a `GenericSpectrum<f32>`.
fn to_generic<S: Spectrum<Precision = f32>>(spectrum: &S) -> GenericSpectrum<f32> {
    let mut generic =
        GenericSpectrum::<f32>::try_with_capacity(spectrum.precursor_mz().to_f64(), spectrum.len())
            .expect("spectrum capacity should be representable");
    for (mz, intensity) in spectrum.peaks() {
        generic
            .add_peak(mz, intensity)
            .expect("peak should be representable");
    }
    generic
}

fn main() {
    let metric = env_str("METRIC", "cosine");
    let (entropy, modified) = match metric.as_str() {
        "cosine" | "direct" => (false, false),
        "modified-cosine" | "modified" => (false, true),
        "entropy" => (true, false),
        "modified-entropy" => (true, true),
        other => {
            panic!("METRIC must be cosine|modified-cosine|entropy|modified-entropy, got '{other}'")
        }
    };
    let weighted = env_usize("ENTROPY_WEIGHTED", 1) != 0;
    let subsample = env_usize("SUBSAMPLE", 0);
    let stratify_by = env_str("STRATIFY_BY", "pathway");
    let stratify_key = match stratify_by.as_str() {
        "pathway" => "NPC_PATHWAYS",
        "class" => "NPC_CLASSES",
        other => panic!("STRATIFY_BY must be 'pathway' or 'class', got '{other}'"),
    };
    let seed = env_usize("SEED", 1) as u64;
    let limit = env_usize("LIMIT", 0);
    let epochs = env_usize("EPOCHS", 500);
    let perplexity = env_f64("PERPLEXITY", 30.0);
    let k = env_usize("K", (3.0 * perplexity).ceil() as usize);
    let boxes = env_usize("BOXES", 128);
    let mz_tolerance = env_f64("MZ_TOLERANCE", 0.05);
    let intensity_power = env_f64("INTENSITY_POWER", 1.0);
    let learning_rate = env_f64("LR", 200.0) as f32;
    let use_cache = env_usize("CACHE", 1) != 0;
    let out = std::env::var("OUT").unwrap_or_else(|_| "examples/fitsne_npc.csv".to_string());

    // 1. Load the harmonized MS2 dataset (top-60), cached via mascot-rs.
    let load_start = Instant::now();
    let dataset = pollster::block_on(AnnotatedMs2Builder::<f32>::default().top_60_peaks().load())
        .expect("loading the harmonized MS2 dataset should succeed");
    let all = dataset.spectra().as_ref();
    let pool = if limit == 0 || limit >= all.len() {
        all
    } else {
        &all[..limit]
    };
    println!(
        "loaded {} spectra in {}",
        all.len(),
        fmt(load_start.elapsed().as_secs_f64())
    );

    // 2. Choose the working set: full pool, or a stratified subsample balanced across `stratify_key`.
    let selected: Vec<usize> = if subsample == 0 || subsample >= pool.len() {
        (0..pool.len()).collect()
    } else {
        let strata: Vec<String> = pool
            .iter()
            .map(|s| primary_label(s, stratify_key))
            .collect();
        let idx = stratified_indices(&strata, subsample, seed);
        println!(
            "stratified subsample by {stratify_by}: {} of {} spectra across strata",
            idx.len(),
            pool.len(),
        );
        idx
    };
    let spectra: Vec<&MascotGenericFormat<f32>> = selected.iter().map(|&i| &pool[i]).collect();
    let n = spectra.len();
    assert!(n >= 4, "need at least 4 spectra");

    // Per-spectrum pathway and class labels, aligned to the working set.
    let pathways: Vec<String> = spectra
        .iter()
        .map(|s| primary_label(s, "NPC_PATHWAYS"))
        .collect();
    let classes: Vec<String> = spectra
        .iter()
        .map(|s| primary_label(s, "NPC_CLASSES"))
        .collect();

    // 3. Flash-cosine k-NN graph (direct or modified). Cached by everything that determines it.
    let cache_path = format!(
        "/tmp/fitsne_knn_{metric}_n{n}_k{k}_ip{intensity_power}_mz{mz_tolerance}_sub{subsample}_{stratify_by}_s{seed}.bin"
    );
    let neighbors: Vec<Vec<Neighbor<f32>>> =
        if use_cache && std::path::Path::new(&cache_path).exists() {
            let load_start = Instant::now();
            let graph = load_graph(&cache_path);
            assert_eq!(
                graph.len(),
                n,
                "cached graph row count must match sample count"
            );
            println!(
                "loaded cached {metric} k-NN graph from {cache_path} in {}",
                fmt(load_start.elapsed().as_secs_f64())
            );
            graph
        } else {
            let clean_start = Instant::now();
            let merger = SiriusMergeClosePeaks::<f32>::new_with_precision(mz_tolerance)
                .expect("valid merge tolerance");
            let cleaned: Vec<GenericSpectrum<f32>> = spectra
                .par_iter()
                .map(|s| merger.process(&to_generic(*s)))
                .collect();
            println!(
                "cleaned {n} spectra in {}",
                fmt(clean_start.elapsed().as_secs_f64())
            );

            let build_start = Instant::now();
            let search_start;
            let graph: Vec<Vec<Neighbor<f32>>> = if entropy {
                let index = FlashEntropyIndex::<f32>::builder()
                    .mz_power(0.0)
                    .intensity_power(intensity_power)
                    .mz_tolerance(mz_tolerance)
                    .weighted(weighted)
                    .parallel()
                    .build(&cleaned)
                    .expect("entropy index build should succeed");
                println!(
                    "built entropy index (weighted={weighted}) in {}",
                    fmt(build_start.elapsed().as_secs_f64())
                );
                search_start = Instant::now();
                build_graph(&index, &cleaned, k, modified, entropy)
            } else {
                let index = FlashCosineIndex::<f32>::builder()
                    .mz_power(0.0)
                    .intensity_power(intensity_power)
                    .mz_tolerance(mz_tolerance)
                    .parallel()
                    .build(&cleaned)
                    .expect("cosine index build should succeed");
                println!(
                    "built cosine index in {}",
                    fmt(build_start.elapsed().as_secs_f64())
                );
                search_start = Instant::now();
                build_graph(&index, &cleaned, k, modified, entropy)
            };
            let total_edges: usize = graph.iter().map(Vec::len).sum();
            println!(
                "{metric} k-NN graph (k={k}) in {}, {:.1} avg neighbors",
                fmt(search_start.elapsed().as_secs_f64()),
                total_edges as f64 / n as f64,
            );
            if use_cache {
                save_graph(&cache_path, &graph);
                println!("cached {metric} k-NN graph to {cache_path}");
            }
            graph
        };

    // 4. GPU fitsne on the CUDA runtime, O(N) FFT repulsion.
    println!(
        "fitsne on CUDA: n={n}, metric={metric}, perplexity={perplexity}, epochs={epochs}, boxes={boxes}"
    );
    let fit_start = Instant::now();
    let mut tsne = FitSne::<CudaRuntime>::new(n);
    tsne.embedding_dim(2)
        .perplexity(perplexity as f32)
        .epochs(epochs)
        .learning_rate(learning_rate)
        .fft_repulsion(boxes)
        .epoch_callback(move |epoch, _y| {
            if epoch % 25 == 0 || epoch + 1 == epochs {
                eprintln!("  epoch {}/{epochs}", epoch + 1);
            }
        });
    let coords = tsne.fit_with_neighbors(&neighbors, &Default::default());
    assert_eq!(coords.len(), n * 2);
    println!(
        "embedded {n} spectra in {}",
        fmt(fit_start.elapsed().as_secs_f64())
    );
    if let Some(kl) = tsne.kl_divergence() {
        println!("final KL (O(N) FFT estimate): {kl:.4}");
    }

    // 5. Write x,y,npc_pathway,npc_class.
    let file = std::fs::File::create(&out).expect("should create output csv");
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "x,y,npc_pathway,npc_class").expect("write header");
    for ((point, pathway), class) in coords
        .chunks_exact(2)
        .zip(pathways.iter())
        .zip(classes.iter())
    {
        writeln!(
            writer,
            "{},{},\"{pathway}\",\"{class}\"",
            point[0], point[1]
        )
        .expect("write row");
    }
    writer.flush().expect("flush csv");
    println!("wrote {out}");
}
