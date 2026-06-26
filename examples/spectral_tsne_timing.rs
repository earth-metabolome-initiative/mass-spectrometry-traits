//! Times the spectral t-SNE pipeline on the harmonized MS2 dataset, splitting the FLASH index
//! build and search from the bhtsne fit, to see whether the index is the choke point.
//!
//! The dataset is downloaded once via mascot-rs and cached locally, then reused. The embedding runs
//! through `SpectralTsne` (bhtsne-backed), and the per-phase progress callback (Cleaning, Indexing,
//! Searching, Fitting) is timed to attribute wall clock to each stage.
//!
//! Run (downloads on first run, then reuses the local cache):
//!
//! Start small with the `LIMIT` env var (default 2000 spectra), then scale up once the choke point
//! is clear. `LIMIT=0` uses the full ~440k-spectrum set.
//!
//! ```text
//! cargo run --release --example spectral_tsne_timing --features bhtsne
//! LIMIT=10000 EPOCHS=500 PERPLEXITY=30 cargo run --release --example spectral_tsne_timing --features bhtsne
//! ```

use std::time::Instant;

use mascot_rs::prelude::AnnotatedMs2Builder;
use mass_spectrometry::prelude::{ModifiedLinearCosine, SpectralTsne, SpectralTsnePhase};

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

fn main() {
    let epochs = env_usize("EPOCHS", 1_000);
    let perplexity = env_f64("PERPLEXITY", 30.0);
    // Start small: cap the number of spectra so the index/fit cost stays in a few-second budget.
    // The full harmonized set is ~440k spectra, so an unbounded run takes many minutes. Scale this
    // up (LIMIT=10000, 50000, ...) once the small runs make the choke point clear. 0 means no cap.
    let limit = env_usize("LIMIT", 2_000);

    // 1. Load the harmonized MS2 dataset (top-60 variant), downloading and caching via mascot-rs.
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

    // 2. Scorer: modified cosine (mz_power, intensity_power, mz_tolerance).
    let scorer = ModifiedLinearCosine::new(0.0, 0.25, 0.02).expect("valid scorer configuration");

    // 3. Run the embedding, timing each phase from the progress callback. Phases are sequential, so
    //    each inter-callback interval is charged to the phase active at its start.
    let mut per_phase = [0.0f64; 4];
    let mut last_phase: Option<usize> = None;

    let tsne = SpectralTsne::new().perplexity(perplexity).epochs(epochs);

    let fit_start = Instant::now();
    let mut last_t = fit_start;
    let coords = tsne
        .embed_with_progress(spectra, &scorer, &mut |phase, _current, _total| {
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
    let total = fit_start.elapsed().as_secs_f64();
    assert_eq!(coords.len(), n);

    let index_time = per_phase[1] + per_phase[2];
    let fit_time = per_phase[3];

    println!();
    println!("spectral t-SNE timing: {n} spectra, perplexity {perplexity}, {epochs} epochs");
    println!("  cleaning   {:>10}", fmt(per_phase[0]));
    println!("  indexing   {:>10}", fmt(per_phase[1]));
    println!("  searching  {:>10}", fmt(per_phase[2]));
    println!("  fitting    {:>10}", fmt(per_phase[3]));
    println!("  -----------");
    println!("  index build + search  {:>10}", fmt(index_time));
    println!("  bhtsne fit            {:>10}", fmt(fit_time));
    println!("  measured total        {:>10}", fmt(total));
    println!();
    let (choke, ratio) = if index_time >= fit_time {
        ("index", index_time / fit_time.max(1e-9))
    } else {
        ("fit", fit_time / index_time.max(1e-9))
    };
    println!("choke point: {choke} ({ratio:.1}x the other)");
    if epochs > 0 {
        println!("fit per epoch: {}", fmt(fit_time / epochs as f64));
    }
}
