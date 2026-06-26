//! Times the spectral t-SNE pipeline on the harmonized MS2 dataset using the
//! LSH MinHash sketch index for modified-cosine neighbor search, against the
//! exact path, to see how fast a full embedding can be produced.
//!
//! The dataset is downloaded once via mascot-rs and cached locally, then reused.
//! Both runs use the same modified-cosine scorer and the same bhtsne fit, so the
//! difference is the neighbor-search stage (`NeighborSearch::Lsh` versus
//! `NeighborSearch::Exact`). The per-phase timing (Cleaning, Indexing,
//! Searching, Fitting) comes from the progress callback.
//!
//! Run (downloads on first run, then reuses the local cache):
//!
//! ```text
//! cargo run --release --example spectral_tsne_lsh_timing --features bhtsne,minhash
//! MODE=both LIMIT=20000 EPOCHS=500 BANDS=16 cargo run --release \
//!   --example spectral_tsne_lsh_timing --features bhtsne,minhash
//! MODE=lsh LIMIT=0 cargo run --release --example spectral_tsne_lsh_timing --features bhtsne,minhash
//! ```

use std::time::Instant;

use mascot_rs::prelude::AnnotatedMs2Builder;
use mass_spectrometry::prelude::{
    ModifiedLinearCosine, NeighborSearch, SpectralTsne, SpectralTsnePhase, Spectrum,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn phase_index(phase: SpectralTsnePhase) -> usize {
    match phase {
        SpectralTsnePhase::Cleaning => 0,
        SpectralTsnePhase::Indexing => 1,
        SpectralTsnePhase::Searching => 2,
        SpectralTsnePhase::Fitting => 3,
    }
}

fn fmt(d: f64) -> String {
    if d >= 1.0 {
        format!("{d:.2} s")
    } else {
        format!("{:.1} ms", d * 1_000.0)
    }
}

/// Run one embedding, charging each inter-callback interval to the phase active
/// at its start, and print the per-phase breakdown.
fn run<S: Spectrum<Precision = f32>>(
    label: &str,
    tsne: &SpectralTsne,
    spectra: &[S],
    scorer: &ModifiedLinearCosine,
    epochs: usize,
) {
    let mut per_phase = [0.0f64; 4];
    let mut last_phase: Option<usize> = None;

    let start = Instant::now();
    let mut last_t = start;
    let coords = tsne
        .embed_with_progress(spectra, scorer, &mut |phase, _current, _total| {
            let now = Instant::now();
            if let Some(prev) = last_phase {
                per_phase[prev] += now.duration_since(last_t).as_secs_f64();
            }
            last_t = now;
            last_phase = Some(phase_index(phase));
        })
        .expect("embedding should succeed");
    if let Some(prev) = last_phase {
        per_phase[prev] += Instant::now().duration_since(last_t).as_secs_f64();
    }
    let total = start.elapsed().as_secs_f64();
    let neighbors = per_phase[1] + per_phase[2];

    println!(
        "[{label:<5}] {} spectra | clean {} | index {} | search {} | fit {} | neighbors {} | total {}",
        coords.len(),
        fmt(per_phase[0]),
        fmt(per_phase[1]),
        fmt(per_phase[2]),
        fmt(per_phase[3]),
        fmt(neighbors),
        fmt(total),
    );
    if epochs > 0 {
        println!(
            "[{label:<5}] fit per epoch {}",
            fmt(per_phase[3] / epochs as f64)
        );
    }
}

fn main() {
    let epochs = env_usize("EPOCHS", 500);
    let perplexity = env_f64("PERPLEXITY", 30.0);
    let limit = env_usize("LIMIT", 10_000);
    let bands = env_usize("BANDS", 16).max(1);
    let mode = env_string("MODE", "both");

    let load_start = Instant::now();
    let dataset = pollster::block_on(AnnotatedMs2Builder::<f32>::default().top_60_peaks().load())
        .expect("loading the harmonized MS2 dataset should succeed");
    let all = dataset.spectra().as_ref();
    let spectra = if limit == 0 || limit >= all.len() {
        all
    } else {
        &all[..limit]
    };
    let n = spectra.len();
    println!(
        "loaded {} spectra ({} skipped) in {}, using {n} (LIMIT={limit}), cached at {}",
        all.len(),
        dataset.skipped_records(),
        fmt(load_start.elapsed().as_secs_f64()),
        dataset.path().display(),
    );
    assert!(n >= 4, "need at least 4 spectra");
    println!(
        "config: perplexity {perplexity}, {epochs} epochs, mode {mode}, lsh bands {bands}, sketch MinHash<u32,128>\n"
    );

    let scorer = ModifiedLinearCosine::new(0.0, 0.25, 0.02).expect("valid scorer configuration");
    let base = SpectralTsne::new().perplexity(perplexity).epochs(epochs);

    if mode == "exact" || mode == "both" {
        let exact = base.neighbor_search(NeighborSearch::Exact);
        run("exact", &exact, spectra, &scorer, epochs);
    }
    if mode == "lsh" || mode == "both" {
        let lsh = base.neighbor_search(NeighborSearch::Lsh { bands });
        run("lsh", &lsh, spectra, &scorer, epochs);
    }
}
