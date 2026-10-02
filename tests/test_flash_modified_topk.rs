//! Recall and exactness of the two-stage approximate modified top-k search.
//!
//! The dense `search_modified_top_k` is the ground truth. The two-stage
//! `search_modified_top_k_approx` must (1) reproduce it exactly when every query
//! peak seeds candidate generation (`max_query_peaks = usize::MAX`), and (2)
//! recover a high fraction of the true neighbors at a small peak budget, which
//! is the property that makes it usable for t-SNE neighbor search.

use mass_spectrometry::prelude::{
    FlashCosineIndex, GenericSpectrum, RandomSpectrumConfig, SpectraIndexBuilder, Spectrum,
    SpectrumAlloc, SpectrumMut,
};

const MZ_POWER: f64 = 0.0;
const INTENSITY_POWER: f64 = 0.25;
const MZ_TOLERANCE: f64 = 0.02;

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn next_unit(state: &mut u64) -> f64 {
    ((next_u64(state) >> 11) as f64) / ((1u64 << 53) as f64)
}

fn random_base(seed: u64) -> GenericSpectrum {
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let n_peaks = 40 + (next_u64(&mut state) % 41) as usize;
    let precursor_mz = 650.0 + next_unit(&mut state) * 550.0;
    let config = RandomSpectrumConfig {
        precursor_mz,
        n_peaks,
        mz_min: 50.0,
        mz_max: 600.0,
        min_peak_gap: 0.25,
        intensity_min: 1.0,
        intensity_max: 1_000.0,
    };
    GenericSpectrum::random(config, seed).expect("random base spectrum should build")
}

/// A near-duplicate of `template`: same precursor, small jittered peaks.
fn perturb(template: &GenericSpectrum, seed: u64) -> GenericSpectrum {
    let mut state = if seed == 0 {
        0xD1B5_4A32_D192_ED03
    } else {
        seed
    };
    let mut spectrum = GenericSpectrum::with_capacity(template.precursor_mz(), template.len())
        .expect("perturbed spectrum should allocate");
    for (mz, intensity) in template.peaks() {
        let mz_jitter = (next_unit(&mut state) - 0.5) * 0.008;
        let intensity_scale = 0.9 + next_unit(&mut state) * 0.2;
        spectrum
            .add_peak(mz + mz_jitter, intensity * intensity_scale)
            .expect("small perturbations should preserve well-separated peaks");
    }
    spectrum
}

/// A modified analog of `template`: precursor and every fragment shifted by the
/// same `delta`. Plain cosine is near zero (no shared m/z) while modified cosine
/// is near one (every fragment matches under the precursor shift).
#[cfg(feature = "minhash")]
fn analog(template: &GenericSpectrum, delta: f64) -> GenericSpectrum {
    let mut spectrum =
        GenericSpectrum::with_capacity(template.precursor_mz() + delta, template.len())
            .expect("analog spectrum should allocate");
    for (mz, intensity) in template.peaks() {
        spectrum
            .add_peak(mz + delta, intensity)
            .expect("a uniform shift preserves sorted, well-separated peaks");
    }
    spectrum
}

fn clustered_library(clusters: usize, cluster_size: usize, seed: u64) -> Vec<GenericSpectrum> {
    let mut spectra = Vec::with_capacity(clusters * cluster_size);
    for cluster_index in 0..clusters {
        let base_seed =
            seed.wrapping_add((cluster_index as u64).wrapping_mul(0xA076_1D64_78BD_642F));
        let base = random_base(base_seed);
        for variant in 0..cluster_size {
            let variant_seed = base_seed ^ (variant as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB);
            spectra.push(perturb(&base, variant_seed));
        }
    }
    spectra
}

fn build_index(library: &[GenericSpectrum]) -> FlashCosineIndex<f64> {
    FlashCosineIndex::<f64>::builder()
        .mz_power(MZ_POWER)
        .intensity_power(INTENSITY_POWER)
        .mz_tolerance(MZ_TOLERANCE)
        .build(library)
        .expect("index build should succeed")
}

fn ids(results: &[mass_spectrometry::prelude::FlashSearchResult]) -> Vec<u32> {
    results.iter().map(|result| result.spectrum_id).collect()
}

/// Exact when every query peak seeds candidate generation: the approximate path
/// at `max_query_peaks = usize::MAX` equals the dense path result for result, in
/// the same ranked order.
#[test]
fn approx_with_all_peaks_equals_dense() {
    let library = clustered_library(30, 20, 0x51A7_2026_0601_0001);
    let index = build_index(&library);
    let k = 25;

    for query_id in (0..library.len()).step_by(13) {
        let dense = index
            .search_modified_top_k(&library[query_id], k)
            .expect("dense modified top-k should succeed");
        let approx = index
            .search_modified_top_k_approx(&library[query_id], k, usize::MAX, usize::MAX)
            .expect("approximate modified top-k should succeed");
        assert_eq!(dense, approx, "query {query_id}");
    }
}

/// At a small peak budget the two-stage search still recovers the large majority
/// of true neighbors. The intra-cluster near-duplicates dominate the top-k and
/// share the heaviest peaks, so a few seed peaks suffice for high recall.
#[test]
fn approx_small_budget_has_high_recall() {
    let library = clustered_library(40, 24, 0x51A7_2026_0601_0002);
    let index = build_index(&library);
    let k = 20;
    let max_query_peaks = 12;

    let mut total_truth = 0usize;
    let mut total_recovered = 0usize;
    for query_id in (0..library.len()).step_by(7) {
        let truth = ids(&index
            .search_modified_top_k(&library[query_id], k)
            .expect("dense modified top-k should succeed"));
        let approx = ids(&index
            .search_modified_top_k_approx(&library[query_id], k, max_query_peaks, usize::MAX)
            .expect("approximate modified top-k should succeed"));

        total_truth += truth.len();
        total_recovered += truth.iter().filter(|id| approx.contains(id)).count();
    }

    let recall = total_recovered as f64 / total_truth as f64;
    assert!(
        recall >= 0.9,
        "expected modified top-k recall >= 0.9 at max_query_peaks={max_query_peaks}, got {recall:.4}"
    );
}

/// The LSH index retrieves candidates sublinearly and exactly re-ranks them, so
/// it must recover the large majority of the exact modified top-k neighbors,
/// which are high-similarity and therefore collide in the MinHash bands.
#[cfg(feature = "minhash")]
#[test]
fn lsh_index_recovers_high_similarity_neighbors() {
    use mass_spectrometry::prelude::{FlashCosineSketchIndex, MinHash};

    let library = clustered_library(40, 24, 0x51A7_2026_0601_0004);
    let exact = build_index(&library);
    let lsh = FlashCosineSketchIndex::<f64, MinHash<u64, 128>>::build(
        &library,
        MZ_POWER,
        INTENSITY_POWER,
        MZ_TOLERANCE,
    )
    .expect("LSH index build should succeed");
    let k = 20;

    let mut total_truth = 0usize;
    let mut recovered = 0usize;
    for query_id in (0..library.len()).step_by(7) {
        let truth = ids(&exact
            .search_modified_top_k(&library[query_id], k)
            .expect("exact modified top-k should succeed"));
        let got = ids(&lsh
            .search_modified_top_k(&library[query_id], k)
            .expect("LSH modified top-k should succeed"));
        total_truth += truth.len();
        recovered += truth.iter().filter(|id| got.contains(id)).count();
    }

    let recall = recovered as f64 / total_truth as f64;
    assert!(recall >= 0.8, "expected LSH recall >= 0.8, got {recall:.4}");
}

/// Modified analogs are the reason modified cosine exists: low plain cosine, high
/// modified cosine. Each library compound contributes a base spectrum and its
/// analog (precursor and all fragments shifted), and the LSH index must retrieve
/// the analog as a neighbor of the base, just as the exact path does.
#[cfg(feature = "minhash")]
#[test]
fn lsh_index_recovers_modified_analogs() {
    use mass_spectrometry::prelude::{FlashCosineSketchIndex, MinHash};

    let compounds = 60;
    let delta = 21.98;
    let mut library: Vec<GenericSpectrum> = Vec::with_capacity(compounds * 2);
    for j in 0..compounds {
        let base = random_base(0x4D6F_6431_0000_0000 ^ j as u64);
        let shifted = analog(&base, delta);
        library.push(base);
        library.push(shifted);
    }

    let exact = build_index(&library);
    let lsh = FlashCosineSketchIndex::<f64, MinHash<u64, 128>>::build(
        &library,
        MZ_POWER,
        INTENSITY_POWER,
        MZ_TOLERANCE,
    )
    .expect("LSH index build should succeed");
    let k = 10;

    let mut exact_recovered = 0usize;
    let mut lsh_recovered = 0usize;
    for j in 0..compounds {
        let query = &library[2 * j];
        let analog_id = (2 * j + 1) as u32;
        if ids(&exact
            .search_modified_top_k(query, k)
            .expect("exact modified top-k should succeed"))
        .contains(&analog_id)
        {
            exact_recovered += 1;
        }
        if ids(&lsh
            .search_modified_top_k(query, k)
            .expect("LSH modified top-k should succeed"))
        .contains(&analog_id)
        {
            lsh_recovered += 1;
        }
    }

    assert_eq!(
        exact_recovered, compounds,
        "exact modified top-k must find every analog"
    );
    let recall = lsh_recovered as f64 / compounds as f64;
    assert!(
        recall >= 0.9,
        "expected LSH analog recall >= 0.9, got {recall:.4}"
    );
}
