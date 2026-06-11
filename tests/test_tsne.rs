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
fn embedding_is_deterministic_and_seed_changes_it() {
    let library = reference_library();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();
    let run = |tsne: SpectralTsne| tsne.embed(&library, &scorer).unwrap();

    // Same configuration (default seed) is bit-for-bit reproducible.
    let a = run(SpectralTsne::new()
        .perplexity(1.0)
        .epochs(250)
        .mz_tolerance(0.1));
    let b = run(SpectralTsne::new()
        .perplexity(1.0)
        .epochs(250)
        .mz_tolerance(0.1));
    assert_eq!(a, b, "the default seed must give a reproducible embedding");

    // A different seed changes the layout.
    let c = run(SpectralTsne::new()
        .perplexity(1.0)
        .epochs(250)
        .mz_tolerance(0.1)
        .seed(42));
    assert_ne!(a, c, "a different seed should change the embedding");
}

#[test]
fn progress_reports_each_phase_in_order() {
    let library: Vec<GenericSpectrum> = (0..2).flat_map(|_| reference_library()).collect();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();

    let mut events: Vec<(SpectralTsnePhase, usize, usize)> = Vec::new();
    let embedding = SpectralTsne::new()
        .perplexity(2.0)
        .epochs(250)
        .mz_tolerance(0.1)
        .embed_with_progress(&library, &scorer, &mut |phase, done, total| {
            events.push((phase, done, total));
        })
        .expect("embedding with progress should succeed");

    assert_eq!(embedding.len(), library.len());
    // Every phase is reported at least once.
    for phase in [
        SpectralTsnePhase::Cleaning,
        SpectralTsnePhase::Indexing,
        SpectralTsnePhase::Searching,
        SpectralTsnePhase::Fitting,
    ] {
        assert!(
            events.iter().any(|&(p, _, _)| p == phase),
            "phase {phase:?} was never reported: {events:?}"
        );
    }
    // Cleaning and Searching reach their totals (one tick per spectrum).
    assert!(events.contains(&(SpectralTsnePhase::Cleaning, library.len(), library.len())));
    assert!(events.contains(&(SpectralTsnePhase::Searching, library.len(), library.len())));

    // Fitting ticks once per epoch and reaches the final epoch.
    let fitting: Vec<_> = events
        .iter()
        .filter(|&&(p, _, _)| p == SpectralTsnePhase::Fitting)
        .collect();
    assert!(
        fitting.len() > 2,
        "Fitting should tick per epoch, got {fitting:?}"
    );
    assert!(events.contains(&(SpectralTsnePhase::Fitting, 250, 250)));
}

#[test]
fn embed_with_frames_streams_one_layout_per_epoch() {
    let library = reference_library();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();
    let epochs = 250;

    let mut frames: Vec<(usize, Vec<f64>)> = Vec::new();
    let embedding = SpectralTsne::new()
        .perplexity(1.0)
        .epochs(epochs)
        .mz_tolerance(0.1)
        .embed_with_frames(
            &library,
            &scorer,
            &mut |_, _, _| {},
            &mut |epoch, layout| frames.push((epoch, layout.to_vec())),
        )
        .expect("embedding with frames should succeed");

    // One frame per epoch, each a full flat 2D layout of finite coordinates.
    assert_eq!(frames.len(), epochs);
    assert_eq!(frames.last().unwrap().0, epochs - 1);
    for (epoch, layout) in &frames {
        assert_eq!(
            layout.len(),
            library.len() * 2,
            "frame {epoch} has the wrong length"
        );
        assert!(layout.iter().all(|v| v.is_finite()));
    }

    // The final frame is the returned layout, flattened (same coordinate space).
    let final_flat: Vec<f64> = embedding.iter().flatten().copied().collect();
    assert_eq!(frames.last().unwrap().1, final_flat);
}

fn spectrum(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut s = GenericSpectrum::with_capacity(precursor, peaks.len()).unwrap();
    for &(mz, intensity) in peaks {
        s.add_peak(mz, intensity).unwrap();
    }
    s
}

#[test]
fn sparse_neighbors_and_similarity_floor_stay_finite() {
    // s0 and s1 overlap; s2, s3, s4 are disjoint isolates with zero matches, so
    // their rows are fully padded (exercising the no-neighbor anchor fallback)
    // and s0/s1 are padded past their single real neighbor. The large finite
    // neutral distance must not produce NaN/Inf coordinates.
    let library = vec![
        spectrum(350.0, &[(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]),
        spectrum(350.0, &[(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]),
        spectrum(1250.0, &[(1000.0, 1.0), (1100.0, 2.0), (1200.0, 3.0)]),
        spectrum(2250.0, &[(2000.0, 1.0), (2100.0, 2.0), (2200.0, 3.0)]),
        spectrum(3250.0, &[(3000.0, 1.0), (3100.0, 2.0), (3200.0, 3.0)]),
    ];
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();

    // Default (padding neutralization only) and an aggressive similarity floor
    // (neutralizing weak real neighbors too) must both stay finite.
    for floor in [0.0, 0.9] {
        let embedding = SpectralTsne::new()
            .perplexity(1.0)
            .epochs(250)
            .mz_tolerance(0.1)
            .min_neighbor_similarity(floor)
            .embed(&library, &scorer)
            .unwrap_or_else(|e| panic!("embed with floor {floor} should succeed: {e:?}"));

        assert_eq!(embedding.len(), library.len());
        assert!(
            embedding
                .iter()
                .all(|[x, y]| x.is_finite() && y.is_finite()),
            "floor {floor} produced non-finite coordinates: {embedding:?}"
        );
    }
}

#[test]
fn embed_from_neighbors_matches_the_full_embed() {
    let library = reference_library();
    let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();
    let tsne = SpectralTsne::new()
        .perplexity(1.0)
        .epochs(250)
        .mz_tolerance(0.1);

    // Reproduce the neighbors embed() would compute internally, then feed them
    // back: same seed and parameters must give the same embedding.
    let merger = SiriusMergeClosePeaks::new(0.1).unwrap();
    let cleaned: Vec<GenericSpectrum> = library.iter().map(|s| merger.process(s)).collect();
    let k = (3.0_f64 * 1.0_f64.min((cleaned.len() as f64 - 1.0) / 3.0)) as usize;
    let neighbors = scorer.top_k_neighbors(&cleaned, k.max(1)).unwrap();

    let from_neighbors = tsne.embed_from_neighbors(&neighbors, &scorer).unwrap();
    let full = tsne.embed(&library, &scorer).unwrap();
    assert_eq!(from_neighbors, full);
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
