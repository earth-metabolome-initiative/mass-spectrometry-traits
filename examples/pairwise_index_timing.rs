//! Times the specialized direct-cosine self-similarity index against the general per-query Flash
//! search, on the harmonized MS2 dataset, to see how much the block-pruned pairwise path wins.
//!
//! `SpectralTsne` builds neighbors with the general `FlashCosineIndex` and one `search_*_top_k` call
//! per spectrum, which is the unpruned O(n^2) path the `spectral_tsne_timing` example measured. This
//! crate also ships `FlashCosineSelfSimilarityIndex`: a one-shot direct-cosine index that bakes the
//! score threshold, `k`, and a precursor-mass filter in at build time and prunes whole spectrum
//! blocks before exact scoring. This example builds both at matching parameters and times them.
//!
//! Both paths here are direct cosine (the specialized index has no modified/neutral-loss variant).
//!
//! ```text
//! cargo run --release --example pairwise_index_timing --features bhtsne
//! LIMIT=50000 K=30 THRESHOLD=0.5 PEPMASS=0.5 cargo run --release --example pairwise_index_timing --features bhtsne
//! ```
//!
//! Env knobs: `LIMIT` (spectra cap, 0 = all, default 10000), `K` (neighbors per row, default 30),
//! `THRESHOLD` (score cutoff baked into the pruned index, default 0.5), `PEPMASS` (precursor-mass Da
//! tolerance, default 0.5), `MZ_TOLERANCE` (peak match Da, default 0.02), `INTENSITY_POWER`
//! (default 1.0). `BASELINE=0` skips the slow general-index path.

use std::time::Instant;

use mascot_rs::prelude::AnnotatedMs2Builder;
use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineSelfSimilarityIndex, GenericSpectrum, SiriusMergeClosePeaks,
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

fn fmt(d: f64) -> String {
    if d >= 1.0 {
        format!("{d:.2} s")
    } else {
        format!("{:.1} ms", d * 1_000.0)
    }
}

/// Owns any spectrum as a `GenericSpectrum<f32>` of the same precision.
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

/// Mean retained neighbors per row, to confirm the two paths build comparable graphs.
fn avg_degree(rows: &[Vec<u32>]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().map(Vec::len).sum::<usize>() as f64 / rows.len() as f64
}

fn main() {
    let limit = env_usize("LIMIT", 10_000);
    let k = env_usize("K", 30);
    let threshold = env_f64("THRESHOLD", 0.5);
    let pepmass = env_f64("PEPMASS", 0.5);
    let mz_tolerance = env_f64("MZ_TOLERANCE", 0.02);
    let intensity_power = env_f64("INTENSITY_POWER", 1.0);
    let run_baseline = env_usize("BASELINE", 1) != 0;

    // 1. Load and slice the harmonized MS2 dataset (top-60 variant), cached via mascot-rs.
    let load_start = Instant::now();
    let dataset = pollster::block_on(AnnotatedMs2Builder::<f32>::default().top_60_peaks().load())
        .expect("loading the harmonized MS2 dataset should succeed");
    let all = dataset.spectra().as_ref();
    let raw = if limit == 0 || limit >= all.len() {
        all
    } else {
        &all[..limit]
    };
    let n = raw.len();
    println!(
        "loaded {} spectra in {}, using {n} (LIMIT={limit})",
        all.len(),
        fmt(load_start.elapsed().as_secs_f64()),
    );
    assert!(n >= 4, "need at least 4 spectra");

    // 2. Clean to GenericSpectrum<f32> with the same merger SpectralTsne uses.
    let clean_start = Instant::now();
    let merger = SiriusMergeClosePeaks::<f32>::new_with_precision(mz_tolerance)
        .expect("valid merge tolerance");
    let cleaned: Vec<GenericSpectrum<f32>> = raw
        .par_iter()
        .map(|s| merger.process(&to_generic(s)))
        .collect();
    println!(
        "cleaned {n} spectra in {}",
        fmt(clean_start.elapsed().as_secs_f64())
    );
    println!(
        "params: k={k}, score_threshold={threshold}, pepmass_tolerance={pepmass} Da, \
         mz_tolerance={mz_tolerance} Da, intensity_power={intensity_power}\n",
    );

    // 3. Specialized direct-cosine self-similarity index: block-pruned, threshold + k + pepmass fixed.
    let build_start = Instant::now();
    let pruned = FlashCosineSelfSimilarityIndex::<f32>::builder()
        .mz_power(0.0)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .score_threshold(threshold)
        .top_k(k)
        .pepmass_tolerance(pepmass)
        .expect("valid pepmass tolerance")
        .parallel()
        .build(&cleaned)
        .expect("self-similarity index build should succeed");
    let pruned_build = build_start.elapsed().as_secs_f64();

    let rows_start = Instant::now();
    let mut pruned_rows: Vec<(u32, Vec<u32>)> = (&pruned)
        .into_par_iter()
        .map(|row| {
            let (id, hits) = row.expect("self-similarity row should succeed");
            (id, hits.into_iter().map(|hit| hit.spectrum_id).collect())
        })
        .collect();
    let pruned_search = rows_start.elapsed().as_secs_f64();
    pruned_rows.sort_by_key(|row| row.0);
    let pruned_graph: Vec<Vec<u32>> = pruned_rows.into_iter().map(|(_, hits)| hits).collect();

    println!("specialized FlashCosineSelfSimilarityIndex (direct cosine, block-pruned)");
    println!("  build           {:>10}", fmt(pruned_build));
    println!("  graph (rows)    {:>10}", fmt(pruned_search));
    println!(
        "  build + graph   {:>10}",
        fmt(pruned_build + pruned_search)
    );
    println!("  avg neighbors   {:>10.1}\n", avg_degree(&pruned_graph));

    if !run_baseline {
        return;
    }

    // 4. General FlashCosineIndex baseline: one thresholded direct top-k search per spectrum, the
    //    same shape SpectralTsne uses (minus the modified path), with the same pepmass filter.
    let gbuild_start = Instant::now();
    let general = FlashCosineIndex::<f32>::builder()
        .mz_power(0.0)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .pepmass_tolerance(pepmass)
        .expect("valid pepmass tolerance")
        .parallel()
        .build(&cleaned)
        .expect("general index build should succeed");
    let general_build = gbuild_start.elapsed().as_secs_f64();

    let gsearch_start = Instant::now();
    let general_graph: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            general
                .search_top_k_threshold(&cleaned[i], k + 1, threshold)
                .expect("search should succeed")
                .into_iter()
                .filter(|hit| hit.spectrum_id as usize != i)
                .take(k)
                .map(|hit| hit.spectrum_id)
                .collect()
        })
        .collect();
    let general_search = gsearch_start.elapsed().as_secs_f64();

    println!("general FlashCosineIndex (direct cosine, per-query thresholded top-k)");
    println!("  build           {:>10}", fmt(general_build));
    println!("  search          {:>10}", fmt(general_search));
    println!(
        "  build + search  {:>10}",
        fmt(general_build + general_search)
    );
    println!("  avg neighbors   {:>10.1}\n", avg_degree(&general_graph));

    let pruned_total = pruned_build + pruned_search;
    let general_total = general_build + general_search;
    let speedup = general_total / pruned_total.max(1e-9);
    println!(
        "specialized index is {speedup:.1}x {} than the general per-query path \
         ({} vs {})",
        if speedup >= 1.0 { "faster" } else { "slower" },
        fmt(pruned_total),
        fmt(general_total),
    );
}
