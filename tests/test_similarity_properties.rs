//! Property tests for core similarity invariants.

use mass_spectrometry::prelude::{
    EntropySimilarity, ExactCosine, GenericSpectrum, ModifiedCosine, ScalarSimilarity, Spectrum,
    SpectrumAlloc, SpectrumMut,
};
use proptest::prelude::*;

fn build_spectrum(precursor_mz: f32, peaks: Vec<(f32, f32)>) -> GenericSpectrum<f32, f32> {
    let mut peaks = peaks;
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz.max(0.001), peaks.len());
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

fn scorer_exact() -> ExactCosine<f32, f32> {
    ExactCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
}

fn scorer_modified() -> ModifiedCosine<f32, f32> {
    ModifiedCosine::new(1.0, 1.0, 0.1).expect("valid scorer config")
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
}
