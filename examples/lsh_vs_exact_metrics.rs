//! Ground-truth comparison of the LSH sketch index against the exact
//! modified-cosine FLASH index on the real harmonized MS2 dataset, sweeping
//! rows-per-band (via the band count) in a single run.
//!
//! The exact modified top-k is the ground truth, computed once. For each band
//! config the LSH top-k is computed over the same cleaned library and compared,
//! reporting recall and candidate-volume distributions. More bands means fewer
//! rows per band, a looser collision threshold, more candidates, and (the
//! question) higher recall.
//!
//! ```text
//! LIBRARY=0 QUERIES=2000 K=90 BANDS=16,32,64,128 cargo run --release \
//!   --example lsh_vs_exact_metrics --features rayon,minhash
//! ```
//!
//! Env: `LIBRARY` (library cap, 0 = all, default 0), `QUERIES` (sampled queries,
//! default 2000), `K` (neighbors, default 90), `BANDS` (comma list of band
//! counts to sweep, default `16,32,64,128`).

use std::collections::HashSet;
use std::time::Instant;

use mascot_rs::prelude::AnnotatedMs2Builder;
use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineSketchIndex, GenericSpectrum, MinHash, SiriusMergeClosePeaks,
    SpectraIndexBuilder, SpectralProcessor, Spectrum, SpectrumAlloc, SpectrumMut, TopKSearchState,
};
use rayon::prelude::*;

const MZ_POWER: f64 = 0.0;
const INTENSITY_POWER: f64 = 0.25;
const MZ_TOLERANCE: f64 = 0.02;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bands() -> Vec<usize> {
    std::env::var("BANDS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .filter(|&b| b > 0)
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![16, 32, 64, 128])
}

fn fmt(d: f64) -> String {
    if d >= 1.0 {
        format!("{d:.1} s")
    } else {
        format!("{:.0} ms", d * 1_000.0)
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    (
        mean,
        pct(&v, 50.0),
        pct(&v, 90.0),
        v.last().copied().unwrap_or(f64::NAN),
    )
}

fn main() {
    let lib_cap = env_usize("LIBRARY", 0);
    let n_queries = env_usize("QUERIES", 2000);
    let k = env_usize("K", 90);
    let band_sweep = env_bands();

    let load_start = Instant::now();
    let dataset = pollster::block_on(AnnotatedMs2Builder::<f32>::default().top_60_peaks().load())
        .expect("loading the harmonized MS2 dataset should succeed");
    let all = dataset.spectra().as_ref();
    let library = if lib_cap == 0 || lib_cap >= all.len() {
        all
    } else {
        &all[..lib_cap]
    };
    let n = library.len();
    eprintln!(
        "loaded {} spectra in {}, library = {n}",
        all.len(),
        fmt(load_start.elapsed().as_secs_f64())
    );

    let nq = n_queries.min(n);
    let query_ids: Vec<usize> = (0..nq).map(|i| i * n / nq).collect();
    println!(
        "config: library={n} queries={nq} k={k} sketch=MinHash<u32,128> bands sweep={band_sweep:?}\n"
    );

    // Clean to match the embedding pipeline.
    let merger = SiriusMergeClosePeaks::<f32>::new_with_precision(MZ_TOLERANCE)
        .expect("merger config should build");
    let cleaned: Vec<GenericSpectrum<f32>> = library
        .par_iter()
        .map(|s| {
            let mut g = GenericSpectrum::<f32>::with_capacity(f64::from(s.precursor_mz()), s.len())
                .expect("spectrum should allocate");
            g.add_peaks(s.peaks()).expect("sorted peaks copy");
            merger.process(&g)
        })
        .collect();

    // Exact ground truth, computed once.
    eprintln!("building exact index and computing ground truth for {nq} queries...");
    let t = Instant::now();
    let exact = FlashCosineIndex::<f32>::builder()
        .mz_power(MZ_POWER)
        .intensity_power(INTENSITY_POWER)
        .mz_tolerance(MZ_TOLERANCE)
        .parallel()
        .build(&cleaned)
        .expect("exact index should build");
    let truth: Vec<HashSet<u32>> = query_ids
        .par_iter()
        .map_init(
            || exact.new_search_state(),
            |es, &qi| {
                exact
                    .search_modified_top_k_with_state(&cleaned[qi], k + 1, es)
                    .expect("exact search should succeed")
                    .iter()
                    .filter(|r| r.spectrum_id as usize != qi)
                    .take(k)
                    .map(|r| r.spectrum_id)
                    .collect()
            },
        )
        .collect();
    let exact_counts: Vec<f64> = truth.iter().map(|t| t.len() as f64).collect();
    let (ec_mean, ec_med, _, _) = stats(exact_counts);
    eprintln!(
        "  exact truth ready in {} (exact neighbors/query: mean={ec_mean:.1} median={ec_med:.0})",
        fmt(t.elapsed().as_secs_f64())
    );

    println!(
        "bands  rows/band  recall(mean/median/p90)  recall>=0.5  candidates/q(mean/median/p90/max)  build  query"
    );
    for &bands in &band_sweep {
        let tb = Instant::now();
        let lsh = FlashCosineSketchIndex::<f32, MinHash<u32, 128>>::build_with_bands(
            &cleaned,
            MZ_POWER,
            INTENSITY_POWER,
            MZ_TOLERANCE,
            bands,
        )
        .expect("LSH index should build");
        let build_s = tb.elapsed().as_secs_f64();

        let tq = Instant::now();
        let records: Vec<(f64, usize)> = query_ids
            .par_iter()
            .enumerate()
            .map_init(
                || (lsh.new_search_state(), TopKSearchState::new()),
                |(ls, tk), (qix, &qi)| {
                    let mut hits: Vec<(u32, f64)> = Vec::new();
                    lsh.for_each_modified_top_k_with_state(&cleaned[qi], k + 1, ls, tk, |r| {
                        hits.push((r.spectrum_id, r.score));
                    })
                    .expect("LSH search should succeed");
                    let candidates = ls.diagnostics().candidates_rescored;
                    hits.retain(|&(id, _)| id as usize != qi);
                    hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    hits.truncate(k);
                    let lsh_ids: HashSet<u32> = hits.iter().map(|&(id, _)| id).collect();
                    let exact_set = &truth[qix];
                    let recall = if exact_set.is_empty() {
                        f64::NAN
                    } else {
                        let hit = exact_set.iter().filter(|id| lsh_ids.contains(id)).count();
                        hit as f64 / exact_set.len() as f64
                    };
                    (recall, candidates)
                },
            )
            .collect();
        let query_s = tq.elapsed().as_secs_f64();

        let recalls: Vec<f64> = records
            .iter()
            .map(|r| r.0)
            .filter(|r| !r.is_nan())
            .collect();
        let ge_half = recalls.iter().filter(|&&r| r >= 0.5).count();
        let cands: Vec<f64> = records.iter().map(|r| r.1 as f64).collect();
        let (r_mean, r_med, r_p90, _) = stats(recalls.clone());
        let (c_mean, c_med, c_p90, c_max) = stats(cands);
        let rows = 128 / bands.clamp(1, 128);
        println!(
            "{bands:<5}  {rows:<9}  {r_mean:.3}/{r_med:.3}/{r_p90:.3}        {:>5.1}%      {c_mean:.0}/{c_med:.0}/{c_p90:.0}/{c_max:.0}      {}  {}",
            100.0 * ge_half as f64 / recalls.len().max(1) as f64,
            fmt(build_s),
            fmt(query_s),
        );
    }
}
