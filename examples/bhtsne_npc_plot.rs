//! Companion to `fitsne_npc_plot`: runs the SAME cached direct-cosine k-NN graph through bhtsne
//! (CPU Barnes-Hut) instead of the GPU fitsne, so the two embedding engines can be compared on an
//! identical neighbor graph and identical metric.
//!
//! It loads the binary graph cached by `fitsne_npc_plot` (rows of `(index, arccos-distance)`),
//! converts each distance back to a cosine similarity (`cos(distance)`), and feeds it to
//! `SpectralTsne::embed_from_neighbors`, whose `LinearCosine` metric maps that similarity right back
//! to the same `arccos` distance. The result is bhtsne on the exact same graph fitsne saw. Writes
//! `x,y` in dataset order (join NPC labels by row index for plotting).
//!
//! ```text
//! GRAPH=/tmp/fitsne_knn_n439403_k90_ip0.25_mz0.05.bin EPOCHS=1000 \
//!   cargo run --release --example bhtsne_npc_plot --features bhtsne
//! ```
//!
//! Env knobs: `GRAPH` (cached graph path, required), `EPOCHS` (default 1000), `PERPLEXITY`
//! (default 30), `LR` (learning rate, default 200), `THETA` (Barnes-Hut theta, default 0.5),
//! `OUT` (csv path, default `examples/bhtsne_npc.csv`).

use std::io::Write;
use std::time::Instant;

use mass_spectrometry::prelude::{LinearCosine, SpectralTsne};

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

/// Reads the binary graph written by `fitsne_npc_plot::save_graph` and returns rows of
/// `(index, similarity)`, where `similarity = cos(stored arccos-distance)`.
fn load_graph_as_similarity(path: &str) -> Vec<Vec<(u32, f64)>> {
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
            let index = u32::from_le_bytes(take(4).try_into().unwrap());
            let distance = f32::from_le_bytes(take(4).try_into().unwrap()) as f64;
            row.push((index, distance.cos()));
        }
        graph.push(row);
    }
    graph
}

fn main() {
    let graph_path = std::env::var("GRAPH").expect("set GRAPH to the cached k-NN graph path");
    let epochs = env_usize("EPOCHS", 1000);
    let perplexity = env_f64("PERPLEXITY", 30.0);
    let learning_rate = env_f64("LR", 200.0);
    let theta = env_f64("THETA", 0.5);
    let out = std::env::var("OUT").unwrap_or_else(|_| "examples/bhtsne_npc.csv".to_string());

    let load_start = Instant::now();
    let neighbors = load_graph_as_similarity(&graph_path);
    let n = neighbors.len();
    println!(
        "loaded cached graph ({n} rows) from {graph_path} in {}",
        fmt(load_start.elapsed().as_secs_f64())
    );
    assert!(n >= 4, "need at least 4 spectra");

    // The metric only has to map cos-similarity back to the arccos distance the graph was built with.
    let scorer = LinearCosine::new(0.0, 0.25, 0.05).expect("valid cosine config");

    println!(
        "bhtsne (CPU Barnes-Hut): n={n}, perplexity={perplexity}, epochs={epochs}, theta={theta}"
    );
    let fit_start = Instant::now();
    let coords = SpectralTsne::new()
        .perplexity(perplexity)
        .epochs(epochs)
        .learning_rate(learning_rate)
        .theta(theta)
        .embed_from_neighbors(&neighbors, &scorer)
        .expect("bhtsne embedding should succeed");
    assert_eq!(coords.len(), n);
    println!(
        "embedded {n} spectra in {}",
        fmt(fit_start.elapsed().as_secs_f64())
    );

    let file = std::fs::File::create(&out).expect("should create output csv");
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "x,y").expect("write header");
    for [x, y] in &coords {
        writeln!(writer, "{x},{y}").expect("write row");
    }
    writer.flush().expect("flush csv");
    println!("wrote {out}");
}
