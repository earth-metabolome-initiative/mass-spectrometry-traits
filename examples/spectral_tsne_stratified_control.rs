//! Control run: a stratified subset of the harmonized MS2 dataset, embedded with
//! the exact modified-cosine FLASH index (and optionally the LSH index on the
//! identical subset) so the only variable is exact versus approximate neighbors.
//!
//! The subset is proportionally stratified by NPClassifier primary class and
//! seeded, so it is representative of the full distribution rather than the
//! first-N slice. Writes `x,y,npc_class` per mode.
//!
//! ```text
//! MODE=both LIMIT=50000 EPOCHS=500 cargo run --release \
//!   --example spectral_tsne_stratified_control --features bhtsne,minhash
//! ```
//!
//! Env knobs: `LIMIT` (subset size, default 50000), `EPOCHS` (default 500),
//! `PERPLEXITY` (default 30), `BANDS` (LSH bands, default 16), `SEED`
//! (default fixed), `MODE` (`exact`, `lsh`, or `both`, default `both`),
//! `OUT_PREFIX` (default `/tmp/strat`).

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use mascot_rs::prelude::AnnotatedMs2Builder;
use mass_spectrometry::prelude::{
    GenericSpectrum, ModifiedLinearCosine, NeighborSearch, SpectralTsne, SpectralTsnePhase,
    Spectrum, SpectrumAlloc, SpectrumMut,
};

const MZ_POWER: f64 = 0.0;
const INTENSITY_POWER: f64 = 0.25;
const MZ_TOLERANCE: f64 = 0.02;

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

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn fmt(d: f64) -> String {
    if d >= 1.0 {
        format!("{d:.1} s")
    } else {
        format!("{:.1} ms", d * 1_000.0)
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

/// Run the embedding for one neighbor mode, time it, and write `x,y,npc_class`.
fn run(
    label: &str,
    search: NeighborSearch,
    epochs: usize,
    perplexity: f64,
    subset: &[GenericSpectrum<f32>],
    labels: &[&str],
    out: &str,
) {
    let scorer = ModifiedLinearCosine::new(MZ_POWER, INTENSITY_POWER, MZ_TOLERANCE)
        .expect("valid scorer configuration");
    let tsne = SpectralTsne::new()
        .perplexity(perplexity)
        .epochs(epochs)
        .mz_tolerance(MZ_TOLERANCE)
        .neighbor_search(search);

    let names = ["clean", "index", "search", "fit"];
    let mut per_phase = [0.0f64; 4];
    let mut last: Option<usize> = None;
    let start = Instant::now();
    let mut last_t = start;
    let mut phase_start = start;
    let mut next_mark = 0usize;
    let coords = tsne
        .embed_with_progress(subset, &scorer, &mut |phase, current, total| {
            let now = Instant::now();
            let idx = match phase {
                SpectralTsnePhase::Cleaning => 0,
                SpectralTsnePhase::Indexing => 1,
                SpectralTsnePhase::Searching => 2,
                SpectralTsnePhase::Fitting => 3,
            };
            if let Some(prev) = last {
                per_phase[prev] += now.duration_since(last_t).as_secs_f64();
            }
            last_t = now;
            if last != Some(idx) {
                if let Some(prev) = last {
                    eprintln!(
                        "[{label}] {} done in {}",
                        names[prev],
                        fmt(now.duration_since(phase_start).as_secs_f64())
                    );
                }
                eprintln!("[{label}] {} ({total} steps)...", names[idx]);
                phase_start = now;
                next_mark = total / 10;
            }
            last = Some(idx);
            if total >= 10 && current >= next_mark {
                eprintln!(
                    "[{label}]   {} {current}/{total} ({})",
                    names[idx],
                    fmt(now.duration_since(phase_start).as_secs_f64())
                );
                while current >= next_mark {
                    next_mark += (total / 10).max(1);
                }
            }
        })
        .expect("embedding should succeed");
    if let Some(prev) = last {
        per_phase[prev] += Instant::now().duration_since(last_t).as_secs_f64();
    }
    println!(
        "[{label:<5}] {} spectra | clean {} | index {} | search {} | fit {} | total {}",
        coords.len(),
        fmt(per_phase[0]),
        fmt(per_phase[1]),
        fmt(per_phase[2]),
        fmt(per_phase[3]),
        fmt(start.elapsed().as_secs_f64()),
    );

    let file = std::fs::File::create(out).expect("create csv");
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "x,y,npc_class").expect("header");
    for ([x, y], label) in coords.iter().zip(labels.iter()) {
        writeln!(writer, "{x},{y},\"{label}\"").expect("row");
    }
    writer.flush().expect("flush");
    println!("[{label:<5}] wrote {out}");
}

fn main() {
    let epochs = env_usize("EPOCHS", 500);
    let perplexity = env_f64("PERPLEXITY", 30.0);
    let bands = env_usize("BANDS", 16).max(1);
    let limit = env_usize("LIMIT", 50_000);
    let seed = env_u64("SEED", 0x5EED_1234_ABCD_0001);
    let mode = env_string("MODE", "both");
    let prefix = std::env::var("OUT_PREFIX").unwrap_or_else(|_| "/tmp/strat".to_string());

    let load_start = Instant::now();
    let dataset = pollster::block_on(AnnotatedMs2Builder::<f32>::default().top_60_peaks().load())
        .expect("loading the harmonized MS2 dataset should succeed");
    let all = dataset.spectra().as_ref();
    let n_all = all.len();
    println!(
        "loaded {n_all} spectra in {}",
        fmt(load_start.elapsed().as_secs_f64())
    );

    // Primary NPC class per spectrum (first label before any `|`).
    let primary: Vec<&str> = all
        .iter()
        .map(|s| {
            s.metadata()
                .arbitrary_metadata_value("NPC_CLASSES")
                .unwrap_or("unknown")
                .split('|')
                .next()
                .unwrap_or("unknown")
        })
        .collect();

    // Proportional stratified sample by primary class, seeded.
    let target = limit.min(n_all);
    let mut strata: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, cls) in primary.iter().enumerate() {
        strata.entry(cls).or_default().push(i);
    }
    let mut state = seed | 1;
    let mut selected: Vec<usize> = Vec::with_capacity(target + strata.len());
    for indices in strata.values_mut() {
        let quota = ((indices.len() as f64) * (target as f64) / (n_all as f64)).round() as usize;
        if quota == 0 {
            continue;
        }
        // Partial Fisher-Yates: bring `quota` random elements to the front.
        let len = indices.len();
        for i in 0..quota.min(len) {
            let j = i + (next_u64(&mut state) as usize) % (len - i);
            indices.swap(i, j);
            selected.push(indices[i]);
        }
    }
    selected.sort_unstable();
    let n = selected.len();
    println!(
        "stratified subset: {n} spectra across {} primary classes (target {target})",
        strata.len()
    );
    assert!(n >= 4, "need at least 4 spectra");

    // Reconstruct the selected spectra as owned GenericSpectrum, tracking labels.
    let subset: Vec<GenericSpectrum<f32>> = selected
        .iter()
        .map(|&i| {
            let s = &all[i];
            let mut g = GenericSpectrum::<f32>::with_capacity(f64::from(s.precursor_mz()), s.len())
                .expect("subset spectrum should allocate");
            g.add_peaks(s.peaks()).expect("sorted peaks copy");
            g
        })
        .collect();
    let labels: Vec<&str> = selected.iter().map(|&i| primary[i]).collect();

    println!("config: perplexity {perplexity}, {epochs} epochs, mode {mode}, lsh bands {bands}\n");

    if mode == "exact" || mode == "both" {
        run(
            "exact",
            NeighborSearch::Exact,
            epochs,
            perplexity,
            &subset,
            &labels,
            &format!("{prefix}_exact.csv"),
        );
    }
    if mode == "lsh" || mode == "both" {
        run(
            "lsh",
            NeighborSearch::Lsh { bands },
            epochs,
            perplexity,
            &subset,
            &labels,
            &format!("{prefix}_lsh.csv"),
        );
    }
}
