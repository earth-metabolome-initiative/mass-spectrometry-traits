//! How the MinHash sketch behaves when plain cosine is low but modified cosine
//! is high: the chemical-analog case (spectrum B is spectrum A with fragments
//! shifted by the precursor mass difference), which is the entire reason the
//! modified cosine exists.
//!
//! Each peak contributes two sketch keys: an absolute m/z bucket (direct
//! matches) and a precursor-relative neutral-loss bucket `precursor - mz`
//! (shifted matches). For an analog, the shifted fragments share the
//! neutral-loss key but not the m/z key, and any unshifted fragments share the
//! m/z key but not the neutral-loss key. So per matched peak only one of its
//! keys (out of roughly three in the union) is shared, which caps the set
//! Jaccard near 1/3 regardless of how high the modified cosine is. LSH banding
//! collides on Jaccard, not on modified cosine, so at a high rows-per-band
//! threshold strong analogs are rarely retrieved.
//!
//! This harness measures, averaged over random spectra:
//!   1. For several relationship types, the plain cosine, modified cosine, exact
//!      key-set Jaccard, MinHash-estimated Jaccard, and band-collision rate at
//!      the default 16 bands. This shows retrieval tracks Jaccard (same precursor
//!      plus same fragments), not modified cosine.
//!   2. For a pure analog, the band-collision rate as a function of
//!      rows-per-band, showing the threshold that would be needed to catch it.
//!   3. End-to-end: analog recovery rate of the LSH index versus the exact
//!      modified top-k, at the default banding and a looser banding.
//!
//! Run:
//!
//! ```text
//! cargo run --release --example minhash_analog_behavior --features minhash
//! DELTA=21.98 PAIRS=400 cargo run --release --example minhash_analog_behavior --features minhash
//! ```

use std::collections::HashSet;

use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineSketchIndex, GenericSpectrum, LinearCosine, LshSketcher, MinHash,
    ModifiedLinearCosine, RandomSpectrumConfig, ScalarSimilarity, Sketcher, SpectraIndexBuilder,
    Spectrum, SpectrumAlloc, SpectrumMut, TopKSearchState, sketch_spectrum, spectrum_sketch_keys,
};

type Spec = GenericSpectrum;
type Sketch = MinHash<u32, 128>;

const MZ_POWER: f64 = 0.0;
const INTENSITY_POWER: f64 = 0.25;
const MZ_TOLERANCE: f64 = 0.02;
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

/// Builds a spectrum from peaks, sorting by m/z and dropping any peak closer
/// than `3 * mz_tolerance` to the previous one so the result stays
/// well-separated (the scorers require it).
fn from_peaks(precursor_mz: f64, mut peaks: Vec<(f64, f64)>) -> Spec {
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut spectrum =
        Spec::with_capacity(precursor_mz, peaks.len()).expect("spectrum should allocate");
    let mut last = f64::NEG_INFINITY;
    for (mz, intensity) in peaks {
        if mz - last < 3.0 * MZ_TOLERANCE {
            continue;
        }
        spectrum
            .add_peak(mz, intensity)
            .expect("sorted, well-separated peaks");
        last = mz;
    }
    spectrum
}

/// A near-duplicate at the same precursor: small m/z and intensity jitter.
fn near_duplicate(base: &Spec, seed: u64) -> Spec {
    let mut state = nonzero_seed(seed);
    let peaks: Vec<(f64, f64)> = base
        .peaks()
        .map(|(mz, intensity)| {
            let mz_jitter = (next_unit_f64(&mut state) - 0.5) * 0.01;
            let intensity_scale = 0.9 + next_unit_f64(&mut state) * 0.2;
            (mz + mz_jitter, intensity * intensity_scale)
        })
        .collect();
    from_peaks(base.precursor_mz(), peaks)
}

/// A modified analog: precursor shifted by `delta`, and each fragment shifted by
/// `delta` with probability `shift_fraction` (it carries the modification),
/// otherwise kept at the same m/z. `shift_fraction = 1.0` is a pure analog
/// (cosine near zero, modified cosine near one).
fn analog(base: &Spec, delta: f64, shift_fraction: f64, seed: u64) -> Spec {
    let mut state = nonzero_seed(seed);
    let peaks: Vec<(f64, f64)> = base
        .peaks()
        .map(|(mz, intensity)| {
            if next_unit_f64(&mut state) < shift_fraction {
                (mz + delta, intensity)
            } else {
                (mz, intensity)
            }
        })
        .collect();
    from_peaks(base.precursor_mz() + delta, peaks)
}

fn score<Sim: ScalarSimilarity<Spec, Spec, Similarity = Result<(f64, usize), E>>, E>(
    scorer: &Sim,
    a: &Spec,
    b: &Spec,
) -> f64 {
    match scorer.similarity(a, b) {
        Ok((value, _)) => value,
        Err(_) => 0.0,
    }
}

fn key_jaccard(a: &Spec, b: &Spec) -> f64 {
    let ka: HashSet<u64> = spectrum_sketch_keys(a, MZ_TOLERANCE).collect();
    let kb: HashSet<u64> = spectrum_sketch_keys(b, MZ_TOLERANCE).collect();
    let intersection = ka.intersection(&kb).count();
    let union = ka.union(&kb).count();
    intersection as f64 / union.max(1) as f64
}

/// Whether the two sketches collide in at least one band at `bands` bands (the
/// LSH retrieval predicate), and the count of colliding bands.
fn band_collision(a: &Sketch, b: &Sketch, bands: usize) -> bool {
    let mut ba = Vec::new();
    let mut bb = Vec::new();
    a.band_hashes_into(bands, &mut ba);
    b.band_hashes_into(bands, &mut bb);
    ba.iter().zip(&bb).any(|(x, y)| x == y)
}

#[derive(Clone, Copy)]
enum Partner {
    NearDuplicate,
    Analog(f64),
    Distractor,
}

fn build_partner(kind: Partner, base: &Spec, delta: f64, seed: u64) -> Spec {
    match kind {
        Partner::NearDuplicate => near_duplicate(base, seed),
        Partner::Analog(fraction) => analog(base, delta, fraction, seed),
        Partner::Distractor => random_spectrum_from_seed(seed ^ 0x9E37_79B9_7F4A_7C15),
    }
}

fn main() {
    let pairs = env_usize("PAIRS", 400);
    let delta = env_f64("DELTA", 21.98);
    let library_compounds = env_usize("LIBRARY_COMPOUNDS", 400);
    let top_k = env_usize("TOP_K", 10);

    let cosine = LinearCosine::new(MZ_POWER, INTENSITY_POWER, MZ_TOLERANCE).expect("cosine config");
    let modified = ModifiedLinearCosine::new(MZ_POWER, INTENSITY_POWER, MZ_TOLERANCE)
        .expect("modified config");

    println!(
        "config: mz_power={MZ_POWER} intensity_power={INTENSITY_POWER} mz_tolerance={MZ_TOLERANCE} \
         sketch=MinHash<u32,128> bands={BANDS} delta={delta} pairs={pairs}"
    );

    // ---- 1. Relationship types: similarity versus sketch behavior ----
    println!(
        "\n--- relationship type: cosine vs modified vs sketch (averaged over {pairs} pairs) ---"
    );
    println!(
        "type                 cosine  modified  key_jaccard  minhash_jaccard  band_collision@{BANDS}"
    );
    let kinds: [(&str, Partner); 6] = [
        ("near-dup same-prec", Partner::NearDuplicate),
        ("analog f=0.25", Partner::Analog(0.25)),
        ("analog f=0.50", Partner::Analog(0.50)),
        ("analog f=0.75", Partner::Analog(0.75)),
        ("analog f=1.00", Partner::Analog(1.00)),
        ("distractor", Partner::Distractor),
    ];
    for (label, kind) in kinds {
        let mut cos_sum = 0.0;
        let mut mod_sum = 0.0;
        let mut kjac_sum = 0.0;
        let mut mjac_sum = 0.0;
        let mut collisions = 0usize;
        for p in 0..pairs {
            let base = random_spectrum_from_seed(0x1234_0000 + p as u64);
            let partner = build_partner(kind, &base, delta, 0x5678_0000 + p as u64);
            cos_sum += score(&cosine, &base, &partner);
            mod_sum += score(&modified, &base, &partner);
            kjac_sum += key_jaccard(&base, &partner);
            let sketch_a = sketch_spectrum::<_, Sketch>(&base, MZ_TOLERANCE);
            let sketch_b = sketch_spectrum::<_, Sketch>(&partner, MZ_TOLERANCE);
            mjac_sum += sketch_a.estimate_jaccard(&sketch_b);
            if band_collision(&sketch_a, &sketch_b, BANDS) {
                collisions += 1;
            }
        }
        let denom = pairs.max(1) as f64;
        println!(
            "{label:<20} {:>6.3}  {:>8.3}  {:>11.3}  {:>15.3}  {:>8.3}",
            cos_sum / denom,
            mod_sum / denom,
            kjac_sum / denom,
            mjac_sum / denom,
            collisions as f64 / denom,
        );
    }

    // ---- 2. Pure analog: band-collision rate versus rows per band ----
    println!("\n--- pure analog (f=1.00): band-collision rate vs rows per band ---");
    println!("bands  rows/band  analog_collision  near_dup_collision");
    for &bands in &[128usize, 64, 32, 16, 8, 4] {
        let rows = 128 / bands;
        let mut analog_hits = 0usize;
        let mut near_dup_hits = 0usize;
        for p in 0..pairs {
            let base = random_spectrum_from_seed(0x1234_0000 + p as u64);
            let an = analog(&base, delta, 1.0, 0x5678_0000 + p as u64);
            let nd = near_duplicate(&base, 0x5678_0000 + p as u64);
            let sketch_base = sketch_spectrum::<_, Sketch>(&base, MZ_TOLERANCE);
            if band_collision(
                &sketch_base,
                &sketch_spectrum::<_, Sketch>(&an, MZ_TOLERANCE),
                bands,
            ) {
                analog_hits += 1;
            }
            if band_collision(
                &sketch_base,
                &sketch_spectrum::<_, Sketch>(&nd, MZ_TOLERANCE),
                bands,
            ) {
                near_dup_hits += 1;
            }
        }
        let denom = pairs.max(1) as f64;
        println!(
            "{bands:<5}  {rows:<9}  {:>16.3}  {:>18.3}",
            analog_hits as f64 / denom,
            near_dup_hits as f64 / denom,
        );
    }

    // ---- 3. End-to-end: analog recovery in a library ----
    // Library layout: compound j contributes its base at index 2j and its pure
    // analog at index 2j+1. Query each base; its analog is the planted neighbor.
    let mut library: Vec<Spec> = Vec::with_capacity(library_compounds * 2);
    for j in 0..library_compounds {
        let base = random_spectrum_from_seed(0xABCD_0000 + j as u64);
        let an = analog(&base, delta, 1.0, 0xEF01_0000 + j as u64);
        library.push(base);
        library.push(an);
    }

    let exact = FlashCosineIndex::<f64>::builder()
        .mz_power(MZ_POWER)
        .intensity_power(INTENSITY_POWER)
        .mz_tolerance(MZ_TOLERANCE)
        .build(&library)
        .expect("exact index should build");
    let mut exact_state = exact.new_search_state();
    let mut exact_recovered = 0usize;
    for j in 0..library_compounds {
        let results = exact
            .search_modified_top_k_with_state(&library[2 * j], top_k + 1, &mut exact_state)
            .expect("exact search should succeed");
        let analog_id = (2 * j + 1) as u32;
        if results.iter().any(|r| r.spectrum_id == analog_id) {
            exact_recovered += 1;
        }
    }

    println!(
        "\n--- end-to-end analog recovery (library {} spectra, top_k={top_k}) ---",
        library.len()
    );
    println!(
        "exact modified top-k:       analog recovered for {:.3} of queries",
        exact_recovered as f64 / library_compounds.max(1) as f64
    );

    for &bands in &[BANDS, 64usize] {
        let lsh = FlashCosineSketchIndex::<f64, Sketch>::build_with_bands(
            &library,
            MZ_POWER,
            INTENSITY_POWER,
            MZ_TOLERANCE,
            bands,
        )
        .expect("LSH index should build");
        let mut state = lsh.new_search_state();
        let mut top_k_state = TopKSearchState::new();
        let mut recovered = 0usize;
        for j in 0..library_compounds {
            let analog_id = (2 * j + 1) as u32;
            let mut found = false;
            lsh.for_each_modified_top_k_with_state(
                &library[2 * j],
                top_k + 1,
                &mut state,
                &mut top_k_state,
                |result| {
                    if result.spectrum_id == analog_id {
                        found = true;
                    }
                },
            )
            .expect("LSH search should succeed");
            if found {
                recovered += 1;
            }
        }
        let rows = 128 / bands;
        println!(
            "LSH {bands} bands ({rows} rows/band): analog recovered for {:.3} of queries",
            recovered as f64 / library_compounds.max(1) as f64
        );
    }
}
