//! Regression tests for ModifiedLinearCosine vs ModifiedHungarianCosine divergences.
//!
//! These 4 spectrum pairs were identified by the mass-spectrometry-benchmarks suite
//! as producing different scores/match counts between the two algorithms. Both use
//! SiriusMergeClosePeaks preprocessing, so on well-separated spectra they should
//! produce identical results. Divergences indicate bugs.
//!
//! Spectra are from GNPS (ccmslib IDs), stored in `tests/fixtures/benchmark_spectra/`.

use mass_spectrometry::prelude::{
    GenericSpectrum, ModifiedHungarianCosine, ModifiedLinearCosine, ScalarSimilarity,
    SiriusMergeClosePeaks, SpectralProcessor, SpectrumMut,
};

const TOLERANCE: f32 = 0.01;
const MZ_POWER: f32 = 0.0;
const INTENSITY_POWER: f32 = 1.0;

/// Load a spectrum from `tests/fixtures/benchmark_spectra/<id>.txt`.
///
/// File format: first line is precursor_mz, subsequent lines are `mz intensity`.
fn load_spectrum(id: u32) -> GenericSpectrum<f32, f32> {
    let path = format!("tests/fixtures/benchmark_spectra/{id}.txt");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    let mut lines = content.lines();
    let precursor_mz: f32 = lines
        .next()
        .expect("empty fixture file")
        .trim()
        .parse()
        .expect("bad precursor_mz");
    let peaks: Vec<(f32, f32)> = lines
        .filter(|l| !l.is_empty())
        .map(|line| {
            let mut parts = line.split_whitespace();
            let mz: f32 = parts.next().expect("missing mz").parse().expect("bad mz");
            let intensity: f32 = parts
                .next()
                .expect("missing intensity")
                .parse()
                .expect("bad intensity");
            (mz, intensity)
        })
        .collect();

    let mut spectrum = GenericSpectrum::try_with_capacity(precursor_mz, peaks.len())
        .expect("valid precursor_mz");
    for (mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .unwrap_or_else(|e| panic!("bad peak mz={mz} int={intensity}: {e}"));
    }
    spectrum
}

/// Assert that ModifiedLinearCosine and ModifiedHungarianCosine produce
/// identical scores and match counts for a given pair, in both directions.
fn assert_linear_matches_hungarian(left_id: u32, right_id: u32) {
    let merger = SiriusMergeClosePeaks::new(TOLERANCE).expect("valid tolerance");
    let left = merger.process(&load_spectrum(left_id));
    let right = merger.process(&load_spectrum(right_id));

    let hungarian = ModifiedHungarianCosine::new(MZ_POWER, INTENSITY_POWER, TOLERANCE)
        .expect("valid config");
    let linear =
        ModifiedLinearCosine::new(MZ_POWER, INTENSITY_POWER, TOLERANCE).expect("valid config");

    // Forward direction
    let (h_score, h_matches) = hungarian
        .similarity(&left, &right)
        .unwrap_or_else(|e| panic!("hungarian forward failed: {e}"));
    let (l_score, l_matches) = linear
        .similarity(&left, &right)
        .unwrap_or_else(|e| panic!("linear forward failed: {e}"));

    assert_eq!(
        h_matches, l_matches,
        "pair ({left_id}, {right_id}) forward: match count differs: hungarian={h_matches}, linear={l_matches}"
    );
    assert!(
        (h_score - l_score).abs() < 1e-6,
        "pair ({left_id}, {right_id}) forward: score differs: hungarian={h_score}, linear={l_score}, diff={}",
        (h_score - l_score).abs()
    );

    // Reverse direction
    let (h_score_rev, h_matches_rev) = hungarian
        .similarity(&right, &left)
        .unwrap_or_else(|e| panic!("hungarian reverse failed: {e}"));
    let (l_score_rev, l_matches_rev) = linear
        .similarity(&right, &left)
        .unwrap_or_else(|e| panic!("linear reverse failed: {e}"));

    assert_eq!(
        h_matches_rev, l_matches_rev,
        "pair ({left_id}, {right_id}) reverse: match count differs: hungarian={h_matches_rev}, linear={l_matches_rev}"
    );
    assert!(
        (h_score_rev - l_score_rev).abs() < 1e-6,
        "pair ({left_id}, {right_id}) reverse: score differs: hungarian={h_score_rev}, linear={l_score_rev}, diff={}",
        (h_score_rev - l_score_rev).abs()
    );

    // Bidirectionality: forward == reverse
    assert_eq!(
        h_matches, h_matches_rev,
        "pair ({left_id}, {right_id}) hungarian not bidirectional: forward={h_matches}, reverse={h_matches_rev}"
    );
    assert!(
        (h_score - h_score_rev).abs() < 1e-6,
        "pair ({left_id}, {right_id}) hungarian not bidirectional: forward={h_score}, reverse={h_score_rev}"
    );
}

/// Pair (406, 425): 500 vs 100 peaks, score diff ~1.21e-3, match diff 1
#[test]
fn regression_pair_406_425() {
    assert_linear_matches_hungarian(406, 425);
}

/// Pair (207, 409): 500 vs 326 peaks, score diff ~4.56e-4, match diff 1
#[test]
fn regression_pair_207_409() {
    assert_linear_matches_hungarian(207, 409);
}

/// Pair (390, 397): 461 vs 455 peaks, score diff ~7.90e-6, match diff 1
#[test]
fn regression_pair_390_397() {
    assert_linear_matches_hungarian(390, 397);
}

/// Pair (376, 387): 343 vs 500 peaks, score diff ~3.09e-7, match diff 1
#[test]
fn regression_pair_376_387() {
    assert_linear_matches_hungarian(376, 387);
}
