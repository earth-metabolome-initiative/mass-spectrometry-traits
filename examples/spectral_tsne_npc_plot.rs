//! Embeds the full harmonized MS2 dataset with the fast direct-cosine neighbor path and writes a
//! CSV of `x,y,npc_class` so the layout can be plotted colored by NPClassifier class.
//!
//! This is the "is direct cosine good enough" experiment: it uses `LinearCosine` (whose
//! `SpectralNeighbors` impl takes the direct `search_top_k` path, the cheap one), not the modified
//! neutral-loss metric that dominated the `spectral_tsne_timing` run. The NPC label is the
//! `NPC_CLASSES` MGF header field, read through `metadata().arbitrary_metadata_value`.
//!
//! ```text
//! cargo run --release --example spectral_tsne_npc_plot --features bhtsne
//! EPOCHS=500 PERPLEXITY=30 LIMIT=0 cargo run --release --example spectral_tsne_npc_plot --features bhtsne
//! ```
//!
//! Env knobs: `LIMIT` (spectra cap, 0 = all ~440k, default 0), `EPOCHS` (default 500),
//! `PERPLEXITY` (default 30), `MZ_TOLERANCE` (default 0.05), `OUT` (csv path, default
//! `examples/spectral_tsne_npc.csv`).

use std::io::Write;
use std::time::Instant;

use mascot_rs::prelude::AnnotatedMs2Builder;
use mass_spectrometry::prelude::{LinearCosine, SpectralTsne, SpectralTsnePhase};

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
        format!("{d:.1} s")
    } else {
        format!("{:.1} ms", d * 1_000.0)
    }
}

fn main() {
    let epochs = env_usize("EPOCHS", 500);
    let perplexity = env_f64("PERPLEXITY", 30.0);
    let mz_tolerance = env_f64("MZ_TOLERANCE", 0.05);
    let intensity_power = env_f64("INTENSITY_POWER", 1.0);
    let limit = env_usize("LIMIT", 0);
    let out = std::env::var("OUT").unwrap_or_else(|_| "examples/spectral_tsne_npc.csv".to_string());

    // 1. Load the harmonized MS2 dataset (top-60 variant), cached via mascot-rs.
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
        "loaded {} spectra in {}, using {n} (LIMIT={limit})",
        all.len(),
        fmt(load_start.elapsed().as_secs_f64()),
    );
    assert!(n >= 4, "need at least 4 spectra");

    // 2. NPC class label per spectrum (the MGF `NPC_CLASSES` header field). Missing -> "unknown".
    let labels: Vec<&str> = spectra
        .iter()
        .map(|s| {
            s.metadata()
                .arbitrary_metadata_value("NPC_CLASSES")
                .unwrap_or("unknown")
        })
        .collect();
    let labeled = labels.iter().filter(|l| **l != "unknown").count();
    println!("NPC_CLASSES present on {labeled}/{n} spectra");

    // 3. Fast direct-cosine neighbors + embedding. LinearCosine -> direct search_top_k path.
    let scorer =
        LinearCosine::new(0.0, intensity_power, mz_tolerance).expect("valid cosine config");
    let tsne = SpectralTsne::new()
        .perplexity(perplexity)
        .epochs(epochs)
        .mz_tolerance(mz_tolerance);

    let mut last_phase: Option<SpectralTsnePhase> = None;
    let mut phase_start = Instant::now();
    let fit_start = Instant::now();
    let coords = tsne
        .embed_with_progress(spectra, &scorer, &mut |phase, current, total| {
            if last_phase != Some(phase) {
                if let Some(prev) = last_phase {
                    eprintln!(
                        "  {prev:?} done in {}",
                        fmt(phase_start.elapsed().as_secs_f64())
                    );
                }
                eprintln!("phase {phase:?} ({total} steps)...");
                last_phase = Some(phase);
                phase_start = Instant::now();
            }
            // Coarse fit heartbeat every 10% so a long run shows life.
            if matches!(phase, SpectralTsnePhase::Fitting)
                && total >= 10
                && current % (total / 10).max(1) == 0
            {
                eprintln!(
                    "  fitting {current}/{total} ({})",
                    fmt(phase_start.elapsed().as_secs_f64())
                );
            }
        })
        .expect("embedding should succeed");
    if let Some(prev) = last_phase {
        eprintln!(
            "  {prev:?} done in {}",
            fmt(phase_start.elapsed().as_secs_f64())
        );
    }
    assert_eq!(coords.len(), n);
    println!(
        "embedded {n} spectra in {}",
        fmt(fit_start.elapsed().as_secs_f64())
    );

    // 4. Write x,y,npc_class.
    let write_start = Instant::now();
    let file = std::fs::File::create(&out).expect("should create output csv");
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "x,y,npc_class").expect("write header");
    for ([x, y], label) in coords.iter().zip(labels.iter()) {
        // Class names contain no commas in this dataset, but quote defensively.
        writeln!(writer, "{x},{y},\"{label}\"").expect("write row");
    }
    writer.flush().expect("flush csv");
    println!(
        "wrote {out} in {}",
        fmt(write_start.elapsed().as_secs_f64())
    );
}
