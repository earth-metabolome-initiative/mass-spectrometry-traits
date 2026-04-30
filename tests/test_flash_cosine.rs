//! Tests for the FlashCosineIndex.
//!
//! Verifies exact equivalence with LinearCosine on well-separated spectra,
//! self-similarity, symmetry, empty/edge cases, and modified search.

use mass_spectrometry::prelude::{
    CocaineSpectrum, FlashCosineIndex, FlashCosineIndexError, FlashCosineThresholdIndex,
    FlashSearchResult, GenericSpectrum, GlucoseSpectrum, HydroxyCholesterolSpectrum, LinearCosine,
    PhenylalanineSpectrum, SalicinSpectrum, ScalarSimilarity, SimilarityComputationError,
    SimilarityConfigError, Spectrum, SpectrumAlloc, SpectrumMut,
};

fn make_spectrum_f64(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor, peaks.len()).expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum.add_peak(mz, intensity).expect("valid sorted peak");
    }
    spectrum
}

fn reference_spectra() -> Vec<(&'static str, GenericSpectrum)> {
    vec![
        ("cocaine", GenericSpectrum::cocaine().unwrap()),
        ("glucose", GenericSpectrum::glucose().unwrap()),
        (
            "hydroxy_cholesterol",
            GenericSpectrum::hydroxy_cholesterol().unwrap(),
        ),
        ("salicin", GenericSpectrum::salicin().unwrap()),
        ("phenylalanine", GenericSpectrum::phenylalanine().unwrap()),
    ]
}

#[derive(Clone)]
struct RawSpectrum {
    precursor_mz: f64,
    peaks: Vec<(f64, f64)>,
}

impl Spectrum for RawSpectrum {
    type SortedIntensitiesIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, (f64, f64)>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.peaks.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.peaks.iter().map(|peak| peak.1)
    }

    fn intensity_nth(&self, n: usize) -> f64 {
        self.peaks[n].1
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.peaks.iter().map(|peak| peak.0)
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.peaks[index..].iter().map(|peak| peak.0)
    }

    fn mz_nth(&self, n: usize) -> f64 {
        self.peaks[n].0
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.peaks.iter().copied()
    }

    fn peak_nth(&self, n: usize) -> (f64, f64) {
        self.peaks[n]
    }

    fn precursor_mz(&self) -> f64 {
        self.precursor_mz
    }
}

fn sorted_results(mut results: Vec<FlashSearchResult>) -> Vec<FlashSearchResult> {
    results.sort_by_key(|result| result.spectrum_id);
    results
}

fn assert_results_close(
    actual: Vec<FlashSearchResult>,
    expected: Vec<FlashSearchResult>,
    label: &str,
) {
    let actual = sorted_results(actual);
    let expected = sorted_results(expected);
    assert_eq!(actual.len(), expected.len(), "{label}: result count");
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(actual.spectrum_id, expected.spectrum_id, "{label}: id");
        assert_eq!(actual.n_matches, expected.n_matches, "{label}: matches");
        assert!(
            (actual.score - expected.score).abs() <= 1.0e-12,
            "{label}: score {} != {}",
            actual.score,
            expected.score
        );
    }
}

// ---------- self-similarity: flash score ~1.0 for each spectrum ----------

#[test]
fn self_similarity_all_reference() {
    let spectra = reference_spectra();

    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");

    for (i, (name, spectrum)) in spectra.iter().enumerate() {
        let results = index.search(spectrum).expect("search should succeed");
        let self_result = results
            .iter()
            .find(|r| r.spectrum_id == i as u32)
            .unwrap_or_else(|| panic!("{name}: self-match not found in results"));
        assert!(
            (1.0 - self_result.score).abs() < 1e-10,
            "{name}: self-similarity expected ~1.0, got {}",
            self_result.score
        );
        assert_eq!(
            self_result.n_matches,
            spectrum.len(),
            "{name}: expected {} matches, got {}",
            spectrum.len(),
            self_result.n_matches
        );
    }
}

// ---------- exact equivalence with LinearCosine ----------

#[test]
fn equivalence_with_linear_cosine() {
    let spectra = reference_spectra();
    let linear = LinearCosine::new(1.0_f64, 1.0_f64, 0.1_f64).expect("valid scorer config");

    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");

    // Test every pair.
    for (qname, query) in spectra.iter() {
        let results = index.search(query).expect("search should succeed");

        for (li, (lname, library)) in spectra.iter().enumerate() {
            let (linear_score, linear_matches): (f64, usize) = linear
                .similarity(query, library)
                .expect("LinearCosine should succeed");

            let flash_result = results.iter().find(|r| r.spectrum_id == li as u32);

            if linear_matches == 0 {
                // Flash should not return this spectrum (or return score 0).
                if let Some(r) = flash_result {
                    assert!(
                        r.score.abs() < 1e-12,
                        "{qname} vs {lname}: LinearCosine has 0 matches but Flash score = {}",
                        r.score
                    );
                }
            } else {
                let r = flash_result.unwrap_or_else(|| {
                    panic!("{qname} vs {lname}: LinearCosine matches={linear_matches} score={linear_score} but Flash returned no result")
                });
                assert!(
                    (r.score - linear_score).abs() < 1e-10,
                    "{qname} vs {lname}: Flash={} vs Linear={} (diff={})",
                    r.score,
                    linear_score,
                    (r.score - linear_score).abs()
                );
                assert_eq!(
                    r.n_matches, linear_matches,
                    "{qname} vs {lname}: Flash matches={} vs Linear matches={}",
                    r.n_matches, linear_matches
                );
            }
        }
    }
}

// ---------- different mz_power / intensity_power ----------

#[test]
fn equivalence_mz_power_0() {
    let spectra = reference_spectra();
    let linear = LinearCosine::new(0.0_f64, 1.0_f64, 0.1_f64).expect("valid config");
    let index = FlashCosineIndex::new(0.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");

    let query = &spectra[0].1;
    let results = index.search(query).expect("search should succeed");

    for (li, (_, library)) in spectra.iter().enumerate() {
        let (linear_score, _): (f64, usize) = linear
            .similarity(query, library)
            .expect("LinearCosine should succeed");
        if let Some(r) = results.iter().find(|r| r.spectrum_id == li as u32) {
            assert!(
                (r.score - linear_score).abs() < 1e-10,
                "mz_power=0: Flash={} vs Linear={}",
                r.score,
                linear_score
            );
        }
    }
}

// ---------- empty library / empty query ----------

#[test]
fn empty_library() {
    let empty: Vec<&GenericSpectrum> = Vec::new();
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, empty)
        .expect("empty index build should succeed");
    assert_eq!(index.n_spectra(), 0);

    let query: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
    let results = index.search(&query).expect("search should succeed");
    assert!(results.is_empty());
}

#[test]
fn empty_query() {
    let spectra = reference_spectra();
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");

    let empty = make_spectrum_f64(100.0, &[]);
    let results = index.search(&empty).expect("search should succeed");
    assert!(results.is_empty());
}

// ---------- zero-intensity spectrum ----------

#[test]
fn zero_intensity_spectrum() {
    // Zero-intensity peaks are now rejected; use an empty spectrum instead.
    let empty = make_spectrum_f64(200.0, &[]);
    let normal = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let library = [empty, normal];

    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, library.iter())
        .expect("index build should succeed");

    let query = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let results = index.search(&query).expect("search should succeed");

    // The empty spectrum should yield score 0 or not appear.
    for r in &results {
        if r.spectrum_id == 0 {
            assert!(r.score.abs() < 1e-12, "empty spectrum should score 0");
        }
    }
    // The normal spectrum should have self-similarity ~1.0.
    let normal_result = results.iter().find(|r| r.spectrum_id == 1);
    assert!(normal_result.is_some());
    assert!((1.0 - normal_result.unwrap().score).abs() < 1e-10);
}

// ---------- well-separated precondition enforcement ----------

#[test]
fn rejects_non_well_separated_library() {
    // Gap = 0.15, tolerance = 0.1, min_gap = 0.2 → gap < min_gap.
    let bad = make_spectrum_f64(200.0, &[(100.0, 10.0), (100.15, 8.0)]);
    let result = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, [&bad]);
    assert!(result.is_err());
}

#[test]
fn rejects_non_well_separated_query() {
    let good = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 8.0)]);
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, [&good])
        .expect("index build should succeed");

    let bad = make_spectrum_f64(200.0, &[(100.0, 10.0), (100.15, 8.0)]);
    let result = index.search(&bad);
    assert!(result.is_err());
}

// ---------- single-spectrum library ----------

#[test]
fn single_spectrum_library() {
    let cocaine: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, [&cocaine])
        .expect("index build should succeed");

    let results = index.search(&cocaine).expect("search should succeed");
    assert_eq!(results.len(), 1);
    assert!((1.0 - results[0].score).abs() < 1e-10);
}

#[test]
fn accessors_and_search_with_state_match_stateless_results() {
    let spectra = reference_spectra();
    let index = FlashCosineIndex::new(0.5_f64, 2.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");

    assert_eq!(index.mz_power(), 0.5);
    assert_eq!(index.intensity_power(), 2.0);
    assert_eq!(index.tolerance(), 0.1);
    assert_eq!(index.n_spectra(), spectra.len() as u32);

    let mut state = index.new_search_state();
    let query_a = &spectra[0].1;
    let query_b = &spectra[1].1;
    let empty = make_spectrum_f64(100.0, &[]);

    let stateless_a = sorted_results(index.search(query_a).expect("search should succeed"));
    let stateful_a = sorted_results(
        index
            .search_with_state(query_a, &mut state)
            .expect("stateful search should succeed"),
    );
    assert_eq!(stateful_a, stateless_a);

    let stateless_b = sorted_results(index.search(query_b).expect("search should succeed"));
    let stateful_b = sorted_results(
        index
            .search_with_state(query_b, &mut state)
            .expect("stateful search should succeed"),
    );
    assert_eq!(stateful_b, stateless_b);

    let empty_results = index
        .search_with_state(&empty, &mut state)
        .expect("empty stateful search should succeed");
    assert!(empty_results.is_empty());

    let repeated_a = sorted_results(
        index
            .search_with_state(query_a, &mut state)
            .expect("reused stateful search should succeed"),
    );
    assert_eq!(repeated_a, stateless_a);
}

#[test]
fn thresholded_search_matches_filtered_direct_search() {
    let spectra = reference_spectra();
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");

    for threshold in [0.0_f64, 0.5_f64, 0.9_f64] {
        let mut state = index.new_search_state();
        for (_, query) in &spectra {
            let expected: Vec<FlashSearchResult> = index
                .search(query)
                .expect("direct search should succeed")
                .into_iter()
                .filter(|result| result.score >= threshold)
                .collect();
            let thresholded = index
                .search_threshold_with_state(query, threshold, &mut state)
                .expect("thresholded search should succeed");
            assert_results_close(thresholded, expected, &format!("threshold={threshold}"));
        }
    }
}

#[test]
fn thresholded_emitter_reuses_state_and_validates_threshold() {
    let spectra = reference_spectra();
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");
    let mut state = index.new_search_state();

    let query_a = &spectra[0].1;
    let query_b = &spectra[1].1;

    let mut emitted_a = Vec::new();
    index
        .for_each_threshold_with_state(query_a, 0.9, &mut state, |result| {
            emitted_a.push(result);
        })
        .expect("thresholded emitter should succeed");
    let expected_a: Vec<FlashSearchResult> = index
        .search(query_a)
        .expect("direct search should succeed")
        .into_iter()
        .filter(|result| result.score >= 0.9)
        .collect();
    assert_results_close(emitted_a, expected_a, "emitted_a");

    let mut emitted_b = Vec::new();
    index
        .for_each_threshold_with_state(query_b, 1.1, &mut state, |result| {
            emitted_b.push(result);
        })
        .expect("threshold above one should still validate query and emit no results");
    assert!(emitted_b.is_empty());

    let error = index
        .search_threshold(query_a, f64::NAN)
        .expect_err("non-finite threshold should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("score_threshold")
    );
}

#[test]
fn threshold_index_matches_filtered_direct_search() {
    let spectra = reference_spectra();
    let direct_index =
        FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
            .expect("direct index build should succeed");

    for threshold in [0.5_f64, 0.7_f64, 0.9_f64] {
        let threshold_index = FlashCosineThresholdIndex::new(
            1.0_f64,
            1.0_f64,
            0.1_f64,
            threshold,
            spectra.iter().map(|(_, s)| s),
        )
        .expect("threshold index build should succeed");
        assert_eq!(threshold_index.score_threshold(), threshold);
        assert_eq!(threshold_index.n_spectra(), spectra.len() as u32);

        let mut external_state = threshold_index.new_search_state();
        let mut indexed_state = threshold_index.new_search_state();
        for (query_id, (_, query)) in spectra.iter().enumerate() {
            let expected: Vec<FlashSearchResult> = direct_index
                .search(query)
                .expect("direct search should succeed")
                .into_iter()
                .filter(|result| result.score >= threshold)
                .collect();

            let external = threshold_index
                .search_with_state(query, &mut external_state)
                .expect("threshold search should succeed");
            assert_results_close(
                external,
                expected.clone(),
                &format!("external threshold={threshold}"),
            );

            let mut indexed = Vec::new();
            threshold_index
                .for_each_indexed_with_state(query_id as u32, &mut indexed_state, |result| {
                    indexed.push(result);
                })
                .expect("indexed threshold search should succeed");
            assert_results_close(indexed, expected, &format!("indexed threshold={threshold}"));
        }
    }
}

#[test]
fn threshold_index_validates_threshold_and_query_id() {
    let spectra = reference_spectra();
    let nan_threshold = FlashCosineThresholdIndex::new(
        1.0_f64,
        1.0_f64,
        0.1_f64,
        f64::NAN,
        spectra.iter().map(|(_, s)| s),
    );
    assert!(matches!(
        nan_threshold,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("score_threshold")
        ))
    ));

    let threshold_index = FlashCosineThresholdIndex::new(
        1.0_f64,
        1.0_f64,
        0.1_f64,
        1.1_f64,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("threshold above one should build");
    assert_eq!(threshold_index.n_prefix_peaks(), 0);

    let mut state = threshold_index.new_search_state();
    let mut emitted = Vec::new();
    threshold_index
        .for_each_indexed_with_state(0, &mut state, |result| emitted.push(result))
        .expect("threshold above one should emit no indexed results");
    assert!(emitted.is_empty());

    let out_of_bounds = threshold_index
        .for_each_indexed_with_state(spectra.len() as u32, &mut state, |_| {})
        .expect_err("out-of-bounds query id should fail");
    assert_eq!(out_of_bounds, SimilarityComputationError::IndexOverflow);
}

#[test]
fn modified_search_with_state_reuses_buffers_without_leaking_matches() {
    let library = [
        make_spectrum_f64(300.0, &[(100.0, 10.0), (200.0, 5.0)]),
        make_spectrum_f64(500.0, &[(50.0, 3.0)]),
    ];
    let query = make_spectrum_f64(310.0, &[(100.0, 10.0), (210.0, 5.0)]);
    let nonmatching_query = make_spectrum_f64(700.0, &[(400.0, 9.0)]);

    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, library.iter())
        .expect("index build should succeed");
    let mut state = index.new_search_state();

    let stateless = sorted_results(
        index
            .search_modified(&query)
            .expect("modified search should succeed"),
    );
    let stateful = sorted_results(
        index
            .search_modified_with_state(&query, &mut state)
            .expect("stateful modified search should succeed"),
    );
    assert_eq!(stateful, stateless);

    let no_match = index
        .search_modified_with_state(&nonmatching_query, &mut state)
        .expect("nonmatching modified search should succeed");
    assert!(no_match.is_empty());

    let repeated = sorted_results(
        index
            .search_modified_with_state(&query, &mut state)
            .expect("reused stateful modified search should succeed"),
    );
    assert_eq!(repeated, stateless);
}

#[test]
fn constructor_and_query_validation_errors_are_exposed() {
    let spectra = reference_spectra();

    let nan_power = FlashCosineIndex::new(f64::NAN, 1.0, 0.1, spectra.iter().map(|(_, s)| s));
    assert!(matches!(
        nan_power,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("mz_power")
        ))
    ));

    let inf_intensity =
        FlashCosineIndex::new(1.0, f64::INFINITY, 0.1, spectra.iter().map(|(_, s)| s));
    assert!(matches!(
        inf_intensity,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("intensity_power")
        ))
    ));

    let nan_tolerance = FlashCosineIndex::new(1.0, 1.0, f64::NAN, spectra.iter().map(|(_, s)| s));
    assert!(matches!(
        nan_tolerance,
        Err(FlashCosineIndexError::Config(
            SimilarityConfigError::NonFiniteParameter("mz_tolerance")
        ))
    ));

    let bad_library = RawSpectrum {
        precursor_mz: f64::NAN,
        peaks: vec![(100.0, 1.0)],
    };
    let build_error = FlashCosineIndex::new(1.0, 1.0, 0.1, [&bad_library]);
    assert!(matches!(
        build_error,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("precursor_mz")
        ))
    ));

    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");
    let bad_query = RawSpectrum {
        precursor_mz: f64::NAN,
        peaks: vec![(100.0, 1.0)],
    };
    assert!(matches!(
        index.search_modified(&bad_query),
        Err(SimilarityComputationError::NonFiniteValue(
            "query_precursor_mz"
        ))
    ));

    let mut state = index.new_search_state();
    assert!(matches!(
        index.search_modified_with_state(&bad_query, &mut state),
        Err(SimilarityComputationError::NonFiniteValue(
            "query_precursor_mz"
        ))
    ));
}

// ---------- modified search ----------

#[test]
fn modified_search_includes_shifted_matches() {
    // Library spectrum: precursor = 300, peaks at 100 and 200.
    // Query spectrum: precursor = 310, peaks at 100 and 210.
    //
    // Direct match: query 100 matches lib 100.
    // Shifted match: query NL = 310-210 = 100, lib NL = 300-200 = 100 → match.
    let lib = make_spectrum_f64(300.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let query = make_spectrum_f64(310.0, &[(100.0, 10.0), (210.0, 5.0)]);

    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, [&lib])
        .expect("index build should succeed");

    let direct_results = index.search(&query).expect("direct search should succeed");
    let modified_results = index
        .search_modified(&query)
        .expect("modified search should succeed");

    // Direct search should find 1 match (mz=100).
    let direct = direct_results.iter().find(|r| r.spectrum_id == 0);
    assert!(direct.is_some());
    assert_eq!(direct.unwrap().n_matches, 1);

    // Modified search should find 2 matches (direct + shifted).
    let modified = modified_results.iter().find(|r| r.spectrum_id == 0);
    assert!(modified.is_some());
    assert_eq!(modified.unwrap().n_matches, 2);
    assert!(modified.unwrap().score > direct.unwrap().score);
}

#[test]
fn modified_search_anti_double_counting() {
    // Library: precursor=200, peak at 100 (NL=100).
    // Query: precursor=200, peak at 100 (NL=100).
    //
    // Both direct and shifted match the same library peak.
    // Anti-double-counting should prevent counting it twice.
    let lib = make_spectrum_f64(200.0, &[(100.0, 10.0)]);
    let query = make_spectrum_f64(200.0, &[(100.0, 10.0)]);

    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, [&lib])
        .expect("index build should succeed");

    let direct = index.search(&query).expect("search should succeed");
    let modified = index
        .search_modified(&query)
        .expect("modified search should succeed");

    // Both should yield exactly 1 match due to anti-double-counting.
    assert_eq!(direct[0].n_matches, 1);
    assert_eq!(modified[0].n_matches, 1);
    assert!((direct[0].score - modified[0].score).abs() < 1e-12);
}
