//! Tests for the spectral t-SNE wrapper (`feature = "bhtsne"`).

#![cfg(feature = "bhtsne")]

use core::f64::consts::FRAC_PI_2;

use mass_spectrometry::prelude::*;

fn reference_library() -> Vec<GenericSpectrum> {
    vec![
        GenericSpectrum::cocaine().unwrap(),
        GenericSpectrum::glucose().unwrap(),
        GenericSpectrum::aspirin().unwrap(),
        GenericSpectrum::phenylalanine().unwrap(),
        GenericSpectrum::salicin().unwrap(),
    ]
}

#[test]
fn scorers_define_their_metric_correct_distance() {
    // Cosine similarities are an angle, so the geodesic arccos distance:
    // identical -> 0, orthogonal -> pi/2.
    let cosine = LinearCosine::new(1.0, 1.0, 0.1).unwrap();
    assert_eq!(cosine.distance(1.0), 0.0);
    assert!((cosine.distance(0.0) - FRAC_PI_2).abs() < 1e-12);
    let modified_cosine = ModifiedLinearCosine::new(1.0, 1.0, 0.1).unwrap();
    assert_eq!(modified_cosine.distance(1.0), 0.0);
    assert!((modified_cosine.distance(0.0) - FRAC_PI_2).abs() < 1e-12);

    // Entropy similarities are 1 - JSD, so the Jensen-Shannon sqrt(1 - sim)
    // distance: identical -> 0, orthogonal -> 1.
    let entropy = LinearEntropy::new(0.0, 1.0, 0.1, true).unwrap();
    assert_eq!(entropy.distance(1.0), 0.0);
    assert_eq!(entropy.distance(0.0), 1.0);
    let modified_entropy = ModifiedLinearEntropy::new(0.0, 1.0, 0.1, true).unwrap();
    assert_eq!(modified_entropy.distance(1.0), 0.0);
    assert_eq!(modified_entropy.distance(0.0), 1.0);
}

#[test]
fn embeds_a_cosine_library_into_finite_2d_points() {
    let library = reference_library();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).expect("valid cosine config");

    // No explicit distance: the cosine scorer selects arccos automatically.
    let embedding = SpectralTsne::new()
        .perplexity(1.0)
        .epochs(250)
        .mz_tolerance(0.1)
        .embed(&library, &scorer)
        .expect("cosine embedding should succeed");

    assert_eq!(embedding.len(), library.len());
    assert!(
        embedding
            .iter()
            .all(|[x, y]| x.is_finite() && y.is_finite()),
        "all embedded coordinates must be finite: {embedding:?}"
    );
}

#[test]
fn embeds_an_entropy_library_into_finite_2d_points() {
    let library = reference_library();
    let scorer = LinearEntropy::new(0.0, 1.0, 0.1, true).expect("valid entropy config");

    let embedding = SpectralTsne::new()
        .perplexity(1.0)
        .epochs(250)
        .mz_tolerance(0.1)
        .embed(&library, &scorer)
        .expect("entropy embedding should succeed");

    assert_eq!(embedding.len(), library.len());
    assert!(
        embedding
            .iter()
            .all(|[x, y]| x.is_finite() && y.is_finite())
    );
}

#[test]
fn top_k_neighbors_excludes_self_and_ranks_by_similarity() {
    let merger = SiriusMergeClosePeaks::new(0.1).unwrap();
    let cleaned: Vec<GenericSpectrum> = reference_library()
        .iter()
        .map(|s| merger.process(s))
        .collect();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();

    let k = 2;
    let rows = scorer
        .top_k_neighbors(&cleaned, k)
        .expect("neighbor search should succeed");

    assert_eq!(rows.len(), cleaned.len());
    for (i, row) in rows.iter().enumerate() {
        assert!(row.len() <= k, "row {i} has more than k neighbors: {row:?}");
        assert!(
            row.iter().all(|&(id, _)| (id as usize) != i),
            "row {i} must not contain itself: {row:?}"
        );
        assert!(
            row.iter().all(|&(id, _)| (id as usize) < cleaned.len()),
            "row {i} has an out-of-range index: {row:?}"
        );
        assert!(
            row.windows(2).all(|w| w[0].1 >= w[1].1),
            "row {i} must be in descending similarity: {row:?}"
        );
    }
}

#[test]
fn embeds_a_larger_library_with_k_above_one() {
    // 15 spectra with perplexity 3 gives k = 9 neighbors per point, exercising
    // the real index neighborhoods (not just the small-n edge case).
    let library: Vec<GenericSpectrum> = (0..3).flat_map(|_| reference_library()).collect();
    let scorer = LinearEntropy::new(0.0, 1.0, 0.1, true).unwrap();

    let embedding = SpectralTsne::new()
        .perplexity(3.0)
        .epochs(250)
        .mz_tolerance(0.1)
        .embed(&library, &scorer)
        .expect("larger entropy embedding should succeed");

    assert_eq!(embedding.len(), library.len());
    assert!(
        embedding
            .iter()
            .all(|[x, y]| x.is_finite() && y.is_finite())
    );
}

#[test]
fn perplexity_is_clamped_to_the_data_size() {
    // A perplexity far larger than (n - 1) / 3 would otherwise panic inside
    // bhtsne; the wrapper clamps it instead.
    let library = reference_library();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();

    let embedding = SpectralTsne::new()
        .perplexity(1000.0)
        .epochs(250)
        .mz_tolerance(0.1)
        .embed(&library, &scorer)
        .expect("oversized perplexity should be clamped, not panic");

    assert_eq!(embedding.len(), library.len());
}

#[test]
fn rejects_too_few_spectra() {
    let library: Vec<GenericSpectrum> = vec![
        GenericSpectrum::cocaine().unwrap(),
        GenericSpectrum::glucose().unwrap(),
        GenericSpectrum::aspirin().unwrap(),
    ];
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();

    let error = SpectralTsne::new()
        .mz_tolerance(0.1)
        .embed(&library, &scorer)
        .expect_err("three spectra is too few");
    assert_eq!(
        error,
        SpectralTsneError::TooFewSpectra {
            found: 3,
            needed: 4
        }
    );
}

#[test]
fn rejects_non_positive_perplexity_and_theta() {
    let library = reference_library();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();

    let perplexity_error = SpectralTsne::new()
        .perplexity(0.0)
        .mz_tolerance(0.1)
        .embed(&library, &scorer)
        .expect_err("zero perplexity is invalid");
    assert_eq!(perplexity_error, SpectralTsneError::InvalidPerplexity(0.0));

    let theta_error = SpectralTsne::new()
        .perplexity(1.0)
        .theta(0.0)
        .mz_tolerance(0.1)
        .embed(&library, &scorer)
        .expect_err("zero theta is invalid");
    assert_eq!(theta_error, SpectralTsneError::InvalidTheta(0.0));
}
