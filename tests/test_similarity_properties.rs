//! Property tests for core similarity invariants.

use mass_spectrometry::prelude::{
    EntropySimilarity, GenericSpectrum, HungarianCosine, LinearCosine, ModifiedHungarianCosine,
    ScalarSimilarity, Spectrum, SpectrumAlloc, SpectrumMut,
};
use proptest::prelude::*;

const LINEAR_MZ_TOLERANCE: f32 = 0.1;
const STRICT_LINEAR_MIN_GAP: f32 = (2.0 * LINEAR_MZ_TOLERANCE) + 1e-4;

fn build_spectrum(precursor_mz: f32, peaks: Vec<(f32, f32)>) -> GenericSpectrum<f32, f32> {
    let mut peaks = peaks;
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz.max(0.001), peaks.len())
        .expect("valid spectrum allocation");
    let mut last_mz: Option<f32> = None;

    for (mz_raw, intensity_raw) in peaks {
        let mut mz = mz_raw.max(0.001);
        if let Some(prev) = last_mz
            && mz <= prev
        {
            mz = prev + 1e-4;
        }
        let intensity = intensity_raw.max(1e-6);
        spectrum
            .add_peak(mz, intensity)
            .expect("generated peaks must be sorted by m/z");
        last_mz = Some(mz);
    }

    spectrum
}

/// Build a spectrum where consecutive peaks are at least `min_gap` apart.
///
/// For [`LinearCosine`] tests we pass a strict gap (`2*tolerance + epsilon`)
/// to satisfy the strict well-separated precondition.
fn build_well_separated_spectrum(
    precursor_mz: f32,
    peaks: Vec<(f32, f32)>,
    min_gap: f32,
) -> GenericSpectrum<f32, f32> {
    let mut peaks = peaks;
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz.max(0.001), peaks.len())
        .expect("valid spectrum allocation");
    let mut last_mz: Option<f32> = None;

    for (mz_raw, intensity_raw) in peaks {
        let mut mz = mz_raw.max(0.001);
        if let Some(prev) = last_mz {
            let required = prev + min_gap;
            if mz < required {
                mz = required;
            }
        }
        let intensity = intensity_raw.max(1e-6);
        spectrum
            .add_peak(mz, intensity)
            .expect("generated peaks must be sorted by m/z");
        last_mz = Some(mz);
    }

    spectrum
}

fn scorer_exact() -> HungarianCosine<f32, f32> {
    HungarianCosine::new(1.0, 1.0, LINEAR_MZ_TOLERANCE).expect("valid scorer config")
}

fn scorer_linear() -> LinearCosine<f32, f32> {
    LinearCosine::new(1.0, 1.0, LINEAR_MZ_TOLERANCE).expect("valid scorer config")
}

fn scorer_modified() -> ModifiedHungarianCosine<f32, f32> {
    ModifiedHungarianCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn scorer_entropy() -> EntropySimilarity<f32> {
    EntropySimilarity::unweighted(0.1).expect("valid scorer config")
}

proptest! {
    #[test]
    fn exact_similarity_is_bounded_and_symmetric(
        p1 in 10.0_f32..1500.0_f32,
        p2 in 10.0_f32..1500.0_f32,
        peaks1 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
        peaks2 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
    ) {
        let left = build_spectrum(p1, peaks1);
        let right = build_spectrum(p2, peaks2);
        let scorer = scorer_exact();

        let (ab_score, ab_matches) = scorer
            .similarity(&left, &right)
            .expect("similarity computation should succeed");
        let (ba_score, ba_matches) = scorer
            .similarity(&right, &left)
            .expect("similarity computation should succeed");

        prop_assert!((0.0..=1.0).contains(&ab_score));
        prop_assert!(ab_matches <= left.len().min(right.len()));
        prop_assert!((ab_score - ba_score).abs() < 1e-4);
        prop_assert_eq!(ab_matches, ba_matches);
    }

    #[test]
    fn modified_similarity_is_bounded_and_symmetric(
        p1 in 10.0_f32..1500.0_f32,
        p2 in 10.0_f32..1500.0_f32,
        peaks1 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
        peaks2 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
    ) {
        let left = build_spectrum(p1, peaks1);
        let right = build_spectrum(p2, peaks2);
        let scorer = scorer_modified();

        let (ab_score, ab_matches) = scorer
            .similarity(&left, &right)
            .expect("similarity computation should succeed");
        let (ba_score, ba_matches) = scorer
            .similarity(&right, &left)
            .expect("similarity computation should succeed");

        prop_assert!((0.0..=1.0).contains(&ab_score));
        prop_assert!(ab_matches <= left.len().min(right.len()));
        prop_assert!((ab_score - ba_score).abs() < 1e-4);
        prop_assert_eq!(ab_matches, ba_matches);
    }

    #[test]
    fn entropy_similarity_is_bounded_and_symmetric(
        p1 in 10.0_f32..1500.0_f32,
        p2 in 10.0_f32..1500.0_f32,
        peaks1 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
        peaks2 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
    ) {
        let left = build_spectrum(p1, peaks1);
        let right = build_spectrum(p2, peaks2);
        let scorer = scorer_entropy();

        let (ab_score, ab_matches) = scorer
            .similarity(&left, &right)
            .expect("similarity computation should succeed");
        let (ba_score, ba_matches) = scorer
            .similarity(&right, &left)
            .expect("similarity computation should succeed");

        prop_assert!((0.0..=1.0).contains(&ab_score));
        prop_assert!(ab_matches <= left.len().min(right.len()));
        prop_assert!((ab_score - ba_score).abs() < 1e-4);
        prop_assert_eq!(ab_matches, ba_matches);
    }

    #[test]
    fn self_similarity_remains_maximal(
        precursor in 10.0_f32..1500.0_f32,
        peaks in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
    ) {
        let spectrum = build_spectrum(precursor, peaks);
        let exact = scorer_exact();
        let modified = scorer_modified();
        let entropy = scorer_entropy();

        let (exact_score, exact_matches) = exact
            .similarity(&spectrum, &spectrum)
            .expect("similarity computation should succeed");
        let (modified_score, modified_matches) = modified
            .similarity(&spectrum, &spectrum)
            .expect("similarity computation should succeed");
        let (entropy_score, entropy_matches) = entropy
            .similarity(&spectrum, &spectrum)
            .expect("similarity computation should succeed");

        prop_assert!((1.0 - exact_score).abs() < 1e-4);
        prop_assert!((1.0 - modified_score).abs() < 1e-4);
        prop_assert!((1.0 - entropy_score).abs() < 1e-4);
        prop_assert_eq!(exact_matches, spectrum.len());
        prop_assert_eq!(modified_matches, spectrum.len());
        prop_assert_eq!(entropy_matches, spectrum.len());
    }

    /// Regression test for the Crouse rectangular LAPJV epsilon-tolerance fix.
    ///
    /// With `mz_power=0` and `tolerance=2.0`, the affine cost matrix becomes
    /// near-degenerate: m/z position contributes nothing to the cost, and all
    /// peaks within the wide tolerance window have costs that differ only by
    /// intensity ratios (all clustering near 1 + f64::EPSILON).  This creates
    /// the pathological case identified by Bijsterbosch & Volgenant (2010)
    /// where the augmentation-only rectangular LAPJV accumulates enough
    /// floating-point rounding in dual variables that exact-equality frontier
    /// comparisons miss true minimum-distance ties, yielding suboptimal
    /// augmenting paths.
    ///
    /// Before the fix, self-similarity for spectra with ~200+ peaks scored as
    /// low as 0.999938 instead of 1.0 (87 failures observed in the matchms
    /// validation suite, worst case: stypoltrione with 218 peaks).
    ///
    /// See `geometric-traits/.../crouse/inner.rs` module docs for the full
    /// analysis and literature references (Crouse 2016, Bijsterbosch &
    /// Volgenant 2010, Jonker & Volgenant 1987).
    #[test]
    fn self_similarity_wide_tolerance_mz_power_zero(
        precursor in 10.0_f32..1500.0_f32,
        peaks in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..256),
    ) {
        let spectrum = build_spectrum(precursor, peaks);
        let scorer = HungarianCosine::new(0.0, 1.0, 2.0).expect("valid scorer config");

        let (score, matches) = scorer
            .similarity(&spectrum, &spectrum)
            .expect("similarity computation should succeed");

        prop_assert!(
            (1.0 - score).abs() < 1e-6,
            "self-similarity score {} deviates from 1.0 (spectrum has {} peaks)",
            score,
            spectrum.len()
        );
        prop_assert_eq!(matches, spectrum.len());
    }

    #[test]
    fn linear_cosine_is_bounded_and_symmetric(
        p1 in 10.0_f32..1500.0_f32,
        p2 in 10.0_f32..1500.0_f32,
        peaks1 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
        peaks2 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
    ) {
        let left = build_well_separated_spectrum(p1, peaks1, STRICT_LINEAR_MIN_GAP);
        let right = build_well_separated_spectrum(p2, peaks2, STRICT_LINEAR_MIN_GAP);
        let scorer = scorer_linear();

        let (ab_score, ab_matches) = scorer
            .similarity(&left, &right)
            .expect("similarity computation should succeed");
        let (ba_score, ba_matches) = scorer
            .similarity(&right, &left)
            .expect("similarity computation should succeed");

        prop_assert!((0.0..=1.0).contains(&ab_score));
        prop_assert!(ab_matches <= left.len().min(right.len()));
        prop_assert!((ab_score - ba_score).abs() < 1e-4);
        prop_assert_eq!(ab_matches, ba_matches);
    }

    #[test]
    fn linear_cosine_matches_hungarian_on_well_separated(
        p1 in 10.0_f32..1500.0_f32,
        p2 in 10.0_f32..1500.0_f32,
        peaks1 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
        peaks2 in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
    ) {
        let left = build_well_separated_spectrum(p1, peaks1, STRICT_LINEAR_MIN_GAP);
        let right = build_well_separated_spectrum(p2, peaks2, STRICT_LINEAR_MIN_GAP);
        let linear = scorer_linear();
        let hungarian = scorer_exact();

        let (linear_score, linear_matches) = linear
            .similarity(&left, &right)
            .expect("LinearCosine similarity should succeed");
        let (hungarian_score, hungarian_matches) = hungarian
            .similarity(&left, &right)
            .expect("HungarianCosine similarity should succeed");

        prop_assert!(
            (linear_score - hungarian_score).abs() < 1e-4,
            "LinearCosine={} vs HungarianCosine={} (left={} peaks, right={} peaks)",
            linear_score, hungarian_score, left.len(), right.len()
        );
        prop_assert_eq!(linear_matches, hungarian_matches);
    }

    #[test]
    fn linear_cosine_self_similarity(
        precursor in 10.0_f32..1500.0_f32,
        peaks in prop::collection::vec((1.0_f32..1200.0_f32, 1e-4_f32..5000.0_f32), 1..96),
    ) {
        let spectrum = build_well_separated_spectrum(precursor, peaks, STRICT_LINEAR_MIN_GAP);
        let scorer = scorer_linear();

        let (score, matches) = scorer
            .similarity(&spectrum, &spectrum)
            .expect("similarity computation should succeed");

        prop_assert!((1.0 - score).abs() < 1e-4);
        prop_assert_eq!(matches, spectrum.len());
    }
}
