//! Embeds the harmonized MS2 dataset with the modified-cosine LSH neighbor path
//! (`NeighborSearch::Lsh`) and writes `x,y,npc_class` so the layout can be
//! plotted colored by NPClassifier class, to check that chemically related
//! spectra land together.
//!
//! This is the modified-cosine counterpart to `spectral_tsne_npc_plot` (which
//! uses direct `LinearCosine`). The NPC label is the `NPC_CLASSES` MGF header
//! field, read through `metadata().arbitrary_metadata_value`.
//!
//! ```text
//! cargo run --release --example spectral_tsne_lsh_npc_plot --features bhtsne,minhash
//! LIMIT=50000 EPOCHS=500 BANDS=16 cargo run --release \
//!   --example spectral_tsne_lsh_npc_plot --features bhtsne,minhash
//! ```
//!
//! Env knobs: `LIMIT` (spectra cap, 0 = all, default 50000), `EPOCHS` (default
//! 500), `PERPLEXITY` (default 30), `BANDS` (LSH bands per space, default 16),
//! `OUT` (csv path, default `examples/spectral_tsne_lsh_npc.csv`).

use std::io::Write;
use std::time::Instant;

use mascot_rs::prelude::AnnotatedMs2Builder;
use mass_spectrometry::prelude::{
    ModifiedLinearCosine, NeighborSearch, SpectralTsne, SpectralTsnePhase,
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
    let bands = env_usize("BANDS", 16).max(1);
    let limit = env_usize("LIMIT", 50_000);
    let out =
        std::env::var("OUT").unwrap_or_else(|_| "examples/spectral_tsne_lsh_npc.csv".to_string());

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

    let scorer = ModifiedLinearCosine::new(MZ_POWER, INTENSITY_POWER, MZ_TOLERANCE)
        .expect("valid scorer configuration");
    let tsne = SpectralTsne::new()
        .perplexity(perplexity)
        .epochs(epochs)
        .mz_tolerance(MZ_TOLERANCE)
        .neighbor_search(NeighborSearch::Lsh { bands });

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

    let file = std::fs::File::create(&out).expect("should create output csv");
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "x,y,npc_class").expect("write header");
    for ([x, y], label) in coords.iter().zip(labels.iter()) {
        writeln!(writer, "{x},{y},\"{label}\"").expect("write row");
    }
    writer.flush().expect("flush csv");
    println!("wrote {out}");
}
