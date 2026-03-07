//! Tests for the SiriusMergeClosePeaks spectral processor.

use mass_spectrometry::prelude::{
    AspirinSpectrum, GenericSpectrum, HungarianCosine, LinearCosine, ScalarSimilarity,
    SiriusMergeClosePeaks, SpectralProcessor, Spectrum, SpectrumMut,
};

const TOLERANCE: f64 = 0.1;

fn merger() -> SiriusMergeClosePeaks {
    SiriusMergeClosePeaks::new(TOLERANCE).expect("valid tolerance")
}

fn make_spectrum(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut s =
        GenericSpectrum::try_with_capacity(precursor, peaks.len()).expect("valid precursor");
    for &(mz, intensity) in peaks {
        s.add_peak(mz, intensity).expect("valid peak");
    }
    s
}

// ---------- empty and single peak ----------

#[test]
fn empty_spectrum() {
    let s = make_spectrum(100.0, &[]);
    let result = merger().process(&s);
    assert_eq!(result.len(), 0);
}

#[test]
fn single_peak_unchanged() {
    let s = make_spectrum(100.0, &[(50.0, 1000.0)]);
    let result = merger().process(&s);
    assert_eq!(result.len(), 1);
    let (mz, intensity) = result.peak_nth(0);
    assert!((mz - 50.0).abs() < 1e-6);
    assert!((intensity - 1000.0).abs() < 1e-6);
}

// ---------- identity: well-separated spectrum unchanged ----------

#[test]
fn well_separated_spectrum_unchanged() {
    let peaks = &[(100.0, 500.0), (101.0, 300.0), (102.0, 700.0)];
    let s = make_spectrum(200.0, peaks);
    let result = merger().process(&s);
    assert_eq!(result.len(), 3);
    for (i, &(expected_mz, expected_int)) in peaks.iter().enumerate() {
        let (mz, intensity) = result.peak_nth(i);
        assert!(
            (mz - expected_mz).abs() < 1e-6,
            "peak {i} mz: expected {expected_mz}, got {mz}"
        );
        assert!(
            (intensity - expected_int).abs() < 1e-6,
            "peak {i} intensity: expected {expected_int}, got {intensity}"
        );
    }
}

// ---------- merge pair ----------

#[test]
fn merge_pair_keeps_higher_intensity_mz() {
    // Two peaks within 2*0.1 = 0.2 of each other.
    // Peak at 100.15 has higher intensity → its mz survives.
    let s = make_spectrum(200.0, &[(100.0, 300.0), (100.15, 500.0)]);
    let result = merger().process(&s);
    assert_eq!(result.len(), 1);
    let (mz, intensity) = result.peak_nth(0);
    assert!((mz - 100.15).abs() < 1e-6, "expected mz=100.15, got {mz}");
    assert!(
        (intensity - 800.0).abs() < 1e-6,
        "expected intensity=800.0, got {intensity}"
    );
}

// ---------- cluster of three ----------

#[test]
fn cluster_of_three_merges_to_one() {
    // Three peaks all within 2*0.1=0.2 of the dominant peak.
    let s = make_spectrum(200.0, &[(100.0, 200.0), (100.1, 1000.0), (100.19, 300.0)]);
    let result = merger().process(&s);
    assert_eq!(result.len(), 1);
    let (mz, intensity) = result.peak_nth(0);
    assert!(
        (mz - 100.1).abs() < 1e-6,
        "expected mz=100.1 (highest intensity), got {mz}"
    );
    assert!(
        (intensity - 1500.0).abs() < 1e-6,
        "expected intensity=1500.0, got {intensity}"
    );
}

// ---------- preserves precursor_mz ----------

#[test]
fn preserves_precursor_mz() {
    let precursor = 123.456;
    let s = make_spectrum(precursor, &[(50.0, 100.0), (50.1, 200.0)]);
    let result = merger().process(&s);
    assert!(
        (result.precursor_mz() - precursor).abs() < 1e-6,
        "precursor_mz should be preserved"
    );
}

// ---------- aspirin reference: has close peaks that need merging ----------

#[test]
fn aspirin_after_merge_is_well_separated() {
    let aspirin: GenericSpectrum =
        GenericSpectrum::aspirin().expect("reference spectrum should build");

    // Aspirin has peaks 105.0333 and 105.0445 that are only 0.0112 apart,
    // which is less than 2*0.1=0.2.
    let result = merger().process(&aspirin);

    // Verify the strict well-separated invariant: consecutive peaks > 2*tolerance apart.
    let min_gap = 2.0 * TOLERANCE;
    let mz_values: Vec<f64> = result.mz().collect();
    for w in mz_values.windows(2) {
        assert!(
            w[1] - w[0] > min_gap,
            "peaks {:.4} and {:.4} are only {:.4} apart, need > {min_gap:.4}",
            w[0],
            w[1],
            w[1] - w[0],
        );
    }

    // Output should have fewer peaks than input since some were merged.
    assert!(
        result.len() < aspirin.len(),
        "expected fewer peaks after merging, got {} vs {}",
        result.len(),
        aspirin.len()
    );
}

// ---------- LinearCosine after merge: no panic, matches HungarianCosine ----------

#[test]
fn linear_cosine_after_merge_matches_hungarian() {
    let aspirin: GenericSpectrum =
        GenericSpectrum::aspirin().expect("reference spectrum should build");
    let m = merger();
    let merged = m.process(&aspirin);

    let linear = LinearCosine::new(1.0, 1.0, TOLERANCE).expect("valid config");
    let hungarian = HungarianCosine::new(1.0, 1.0, TOLERANCE).expect("valid config");

    // Use a well-separated reference as the other spectrum.
    let cocaine: GenericSpectrum = make_spectrum(
        304.154,
        &[
            (82.065, 100.0),
            (150.091, 200.0),
            (182.117, 1000.0),
            (272.128, 50.0),
            (304.154, 500.0),
        ],
    );

    let (lin_score, lin_matches) = linear
        .similarity(&merged, &cocaine)
        .expect("LinearCosine should succeed on merged spectrum");
    let (hun_score, hun_matches) = hungarian
        .similarity(&merged, &cocaine)
        .expect("HungarianCosine should succeed");

    assert!(
        (lin_score - hun_score).abs() < 1e-6,
        "LinearCosine={lin_score} vs HungarianCosine={hun_score}"
    );
    assert_eq!(lin_matches, hun_matches);
}

// ---------- self-similarity preserved after merge ----------

#[test]
fn self_similarity_after_merge() {
    let aspirin: GenericSpectrum =
        GenericSpectrum::aspirin().expect("reference spectrum should build");
    let merged = merger().process(&aspirin);

    let linear = LinearCosine::new(1.0, 1.0, TOLERANCE).expect("valid config");
    let (sim, matches) = linear
        .similarity(&merged, &merged)
        .expect("self-similarity should succeed");

    assert!(
        (1.0 - sim).abs() < 1e-6,
        "self-similarity should be ~1.0, got {sim}"
    );
    assert_eq!(
        matches,
        merged.len(),
        "all peaks should match in self-similarity"
    );
}

#[test]
fn overflowing_cluster_is_clamped_to_max_finite_intensity() {
    let s = make_spectrum(200.0, &[(100.0, f64::MAX), (100.05, f64::MAX)]);
    let result = merger().process(&s);
    assert_eq!(result.len(), 1);
    let (_, intensity) = result.peak_nth(0);
    assert!(intensity.is_finite(), "merged intensity must be finite");
    assert_eq!(intensity, f64::MAX);
}

#[test]
fn overflow_clamping_preserves_sorted_valid_output() {
    let s = make_spectrum(
        200.0,
        &[(100.0, f64::MAX), (100.05, f64::MAX), (101.0, 1.0)],
    );
    let result = merger().process(&s);

    assert_eq!(result.len(), 2);
    let (mz0, i0) = result.peak_nth(0);
    let (mz1, i1) = result.peak_nth(1);

    assert!(mz0 < mz1, "merged peaks must stay sorted");
    assert!(
        i0.is_finite() && i1.is_finite(),
        "intensities must remain finite"
    );
    assert_eq!(i0, f64::MAX, "overflowed merged cluster must be clamped");
}
