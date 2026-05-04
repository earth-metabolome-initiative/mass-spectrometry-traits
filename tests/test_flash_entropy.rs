//! Tests for the FlashEntropyIndex.
//!
//! Verifies exact equivalence with LinearEntropy on well-separated spectra,
//! self-similarity, empty/edge cases, and modified search.

use mass_spectrometry::prelude::{
    CocaineSpectrum, FlashEntropyIndex, FlashEntropyIndexError, FlashSearchResult, GenericSpectrum,
    GlucoseSpectrum, HydroxyCholesterolSpectrum, LinearEntropy, PhenylalanineSpectrum,
    SalicinSpectrum, ScalarSimilarity, SimilarityComputationError, SimilarityConfigError, Spectrum,
    SpectrumAlloc, SpectrumMut, TopKSearchState,
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
    type Precision = f64;

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

fn top_k_expected(mut results: Vec<FlashSearchResult>, k: usize) -> Vec<FlashSearchResult> {
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.n_matches.cmp(&left.n_matches))
            .then_with(|| left.spectrum_id.cmp(&right.spectrum_id))
    });
    results.truncate(k);
    results
}

fn top_k_threshold_expected(
    mut results: Vec<FlashSearchResult>,
    k: usize,
    score_threshold: f64,
) -> Vec<FlashSearchResult> {
    results.retain(|result| result.score >= score_threshold);
    top_k_expected(results, k)
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

// ---------- self-similarity (weighted) ----------

#[test]
fn self_similarity_weighted() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");

    for (i, (name, spectrum)) in spectra.iter().enumerate() {
        let results = index.search(spectrum).expect("search should succeed");
        let self_result = results
            .iter()
            .find(|r| r.spectrum_id == i as u32)
            .unwrap_or_else(|| panic!("{name}: self-match not found"));
        assert!(
            (1.0 - self_result.score).abs() < 1e-10,
            "{name}: weighted self-similarity expected ~1.0, got {}",
            self_result.score
        );
        assert_eq!(self_result.n_matches, spectrum.len());
    }
}

// ---------- self-similarity (unweighted) ----------

#[test]
fn self_similarity_unweighted() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        false,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");

    for (i, (name, spectrum)) in spectra.iter().enumerate() {
        let results = index.search(spectrum).expect("search should succeed");
        let self_result = results
            .iter()
            .find(|r| r.spectrum_id == i as u32)
            .unwrap_or_else(|| panic!("{name}: self-match not found"));
        assert!(
            (1.0 - self_result.score).abs() < 1e-10,
            "{name}: unweighted self-similarity expected ~1.0, got {}",
            self_result.score
        );
    }
}

// ---------- exact equivalence with LinearEntropy ----------

#[test]
fn equivalence_with_linear_entropy_weighted() {
    let spectra = reference_spectra();
    let linear = LinearEntropy::weighted(0.1_f64).expect("valid scorer config");
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");

    for (qname, query) in spectra.iter() {
        let results = index.search(query).expect("search should succeed");

        for (li, (lname, library)) in spectra.iter().enumerate() {
            let (linear_score, linear_matches): (f64, usize) = linear
                .similarity(query, library)
                .expect("LinearEntropy should succeed");

            let flash_result = results.iter().find(|r| r.spectrum_id == li as u32);

            if linear_matches == 0 {
                if let Some(r) = flash_result {
                    assert!(
                        r.score.abs() < 1e-12,
                        "{qname} vs {lname}: Linear has 0 matches but Flash score = {}",
                        r.score
                    );
                }
            } else {
                let r = flash_result.unwrap_or_else(|| {
                    panic!("{qname} vs {lname}: Linear matches={linear_matches} score={linear_score} but Flash returned no result")
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

#[test]
fn equivalence_with_linear_entropy_unweighted() {
    let spectra = reference_spectra();
    let linear = LinearEntropy::unweighted(0.1_f64).expect("valid scorer config");
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        false,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");

    for (qname, query) in spectra.iter() {
        let results = index.search(query).expect("search should succeed");

        for (li, (lname, library)) in spectra.iter().enumerate() {
            let (linear_score, linear_matches): (f64, usize) = linear
                .similarity(query, library)
                .expect("LinearEntropy should succeed");

            let flash_result = results.iter().find(|r| r.spectrum_id == li as u32);

            if linear_matches == 0 {
                if let Some(r) = flash_result {
                    assert!(
                        r.score.abs() < 1e-12,
                        "{qname} vs {lname}: Linear has 0 matches but Flash score = {}",
                        r.score
                    );
                }
            } else {
                let r = flash_result.unwrap_or_else(|| {
                    panic!("{qname} vs {lname}: Linear matches={linear_matches} score={linear_score} but Flash returned no result")
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

// ---------- empty library / empty query ----------

#[test]
fn empty_library() {
    let empty: Vec<&GenericSpectrum> = Vec::new();
    let index = FlashEntropyIndex::<f64>::new(0.0_f64, 1.0_f64, 0.1_f64, true, empty)
        .expect("empty index should build");
    assert_eq!(index.n_spectra(), 0);

    let query: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
    let results = index.search(&query).expect("search should succeed");
    assert!(results.is_empty());
}

#[test]
fn empty_query() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");

    let empty = make_spectrum_f64(100.0, &[]);
    let results = index.search(&empty).expect("search should succeed");
    assert!(results.is_empty());
}

// ---------- zero-intensity spectrum ----------

#[test]
fn zero_intensity_library_spectrum() {
    // Zero-intensity peaks are now rejected; use an empty spectrum instead.
    let empty = make_spectrum_f64(200.0, &[]);
    let normal = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let library = [empty, normal];

    let index = FlashEntropyIndex::<f64>::new(0.0_f64, 1.0_f64, 0.1_f64, true, library.iter())
        .expect("index build should succeed");

    let query = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let results = index.search(&query).expect("search should succeed");

    // Empty spectrum should not appear or score 0.
    for r in &results {
        if r.spectrum_id == 0 {
            assert!(r.score.abs() < 1e-12);
        }
    }
    // Normal spectrum should have self-similarity ~1.0.
    let normal_result = results.iter().find(|r| r.spectrum_id == 1);
    assert!(normal_result.is_some());
    assert!((1.0 - normal_result.unwrap().score).abs() < 1e-10);
}

#[test]
fn accessors_and_convenience_constructors_match_expected_configuration() {
    let spectra = reference_spectra();

    let weighted = FlashEntropyIndex::<f64>::weighted(0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("weighted index should build");
    assert!(weighted.is_weighted());
    assert_eq!(weighted.tolerance(), 0.1);
    assert_eq!(weighted.n_spectra(), spectra.len() as u32);
    assert_eq!(weighted.mz_power_f64(), 0.0);
    assert_eq!(weighted.intensity_power_f64(), 1.0);

    let unweighted = FlashEntropyIndex::<f64>::unweighted(0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("unweighted index should build");
    assert!(!unweighted.is_weighted());
    assert_eq!(unweighted.tolerance(), 0.1);
    assert_eq!(unweighted.n_spectra(), spectra.len() as u32);
    assert_eq!(unweighted.mz_power_f64(), 0.0);
    assert_eq!(unweighted.intensity_power_f64(), 1.0);
}

#[test]
fn search_with_state_matches_stateless_results_and_state_reuse_is_stable() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::<f64>::new(
        0.5_f64,
        2.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");
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
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");

    for threshold in [0.0_f64, 0.5_f64, 0.7_f64, 0.9_f64, 1.1_f64] {
        let mut state = index.new_search_state();
        for (_, query) in &spectra {
            let expected: Vec<FlashSearchResult> = index
                .search(query)
                .expect("direct search should succeed")
                .into_iter()
                .filter(|result| result.score >= threshold)
                .collect();
            let actual = index
                .search_threshold_with_state(query, threshold, &mut state)
                .expect("thresholded search should succeed");
            assert_results_close(actual, expected, &format!("threshold={threshold}"));
        }
    }

    let error = index
        .search_threshold(&spectra[0].1, f64::NAN)
        .expect_err("non-finite threshold should be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("score_threshold")
    );
}

#[test]
fn top_k_matches_sorted_direct_search_and_reuses_state() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");

    let query_a = &spectra[0].1;
    let query_b = &spectra[1].1;
    let mut state = index.new_search_state();

    let expected_a = top_k_expected(index.search(query_a).expect("search should succeed"), 3);
    let actual_a = index
        .search_top_k_with_state(query_a, 3, &mut state)
        .expect("top-k search should succeed");
    assert_eq!(actual_a, expected_a);

    let mut top_k_state = TopKSearchState::new();
    let mut streamed_a = Vec::new();
    index
        .for_each_top_k_with_state(query_a, 3, &mut state, &mut top_k_state, |result| {
            streamed_a.push(result);
        })
        .expect("streamed top-k search should succeed");
    assert_eq!(streamed_a, expected_a);

    let expected_b = top_k_expected(index.search(query_b).expect("search should succeed"), 2);
    let actual_b = index
        .search_top_k_with_state(query_b, 2, &mut state)
        .expect("top-k state reuse should succeed");
    assert_eq!(actual_b, expected_b);

    assert!(
        index
            .search_top_k(query_a, 0)
            .expect("zero-k search should succeed")
            .is_empty()
    );
}

#[test]
fn thresholded_top_k_matches_filtered_direct_search() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("index build should succeed");
    let query = &spectra[0].1;

    for threshold in [0.0_f64, 0.5_f64, 0.9_f64, 1.1_f64] {
        let expected = top_k_threshold_expected(
            index.search(query).expect("search should succeed"),
            3,
            threshold,
        );
        let actual = index
            .search_top_k_threshold(query, 3, threshold)
            .expect("thresholded top-k should succeed");
        assert_results_close(actual, expected, "thresholded entropy top-k");
    }
}

#[test]
fn indexed_entropy_queries_match_external_queries() {
    let spectra = reference_spectra();

    for weighted in [true, false] {
        let index = FlashEntropyIndex::<f64>::new(
            0.0_f64,
            1.0_f64,
            0.1_f64,
            weighted,
            spectra.iter().map(|(_, s)| s),
        )
        .expect("index build should succeed");
        let query_id = 0_u32;
        let query = &spectra[query_id as usize].1;
        let mut state = index.new_search_state();

        let direct_expected = index.search(query).expect("external search should work");
        let indexed = index
            .search_indexed_with_state(query_id, &mut state)
            .expect("indexed search should work");
        assert_results_close(indexed, direct_expected.clone(), "indexed direct");

        for threshold in [0.0_f64, 0.5, 0.9, 1.1] {
            let threshold_expected = index
                .search_threshold_with_state(query, threshold, &mut state)
                .expect("external threshold search should work");
            let threshold_indexed = index
                .search_threshold_indexed_with_state(query_id, threshold, &mut state)
                .expect("indexed threshold search should work");
            assert_results_close(
                threshold_indexed,
                threshold_expected.clone(),
                &format!("indexed threshold weighted={weighted} threshold={threshold}"),
            );

            let top_k_expected = top_k_threshold_expected(direct_expected.clone(), 3, threshold);
            let top_k_indexed = index
                .search_top_k_threshold_indexed_with_state(query_id, 3, threshold, &mut state)
                .expect("indexed threshold top-k should work");
            assert_results_close(
                top_k_indexed,
                top_k_expected.clone(),
                &format!("indexed threshold top-k weighted={weighted} threshold={threshold}"),
            );

            let mut top_k_state = TopKSearchState::new();
            let mut streamed = Vec::new();
            index
                .for_each_top_k_threshold_indexed_with_state(
                    query_id,
                    3,
                    threshold,
                    &mut state,
                    &mut top_k_state,
                    |result| streamed.push(result),
                )
                .expect("streamed indexed threshold top-k should work");
            assert_results_close(
                streamed,
                top_k_expected,
                &format!(
                    "streamed indexed threshold top-k weighted={weighted} threshold={threshold}"
                ),
            );
        }

        assert!(
            index
                .search_top_k_indexed_with_state(query_id, 0, &mut state)
                .expect("zero-k indexed search should work")
                .is_empty()
        );

        let mut streamed_top_k = Vec::new();
        let mut top_k_state = TopKSearchState::new();
        index
            .for_each_top_k_indexed_with_state(
                query_id,
                2,
                &mut state,
                &mut top_k_state,
                |result| streamed_top_k.push(result),
            )
            .expect("streamed indexed top-k should work");
        assert_results_close(
            streamed_top_k,
            top_k_expected(direct_expected, 2),
            "streamed indexed top-k",
        );
    }
}

#[test]
fn entropy_thresholded_top_k_prunes_low_bound_spectrum_blocks_without_losing_hits() {
    let high_similarity = make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 10.0), (300.0, 10.0)]);
    let low_similarity = make_spectrum_f64(500.0, &[(100.0, 10.0)]);

    let mut spectra = Vec::new();
    for _ in 0..256 {
        spectra.push(high_similarity.clone());
    }
    for _ in 0..4 {
        spectra.push(low_similarity.clone());
    }

    let index = FlashEntropyIndex::<f64>::unweighted(0.1_f64, spectra.iter())
        .expect("entropy index should build");
    let expected = top_k_threshold_expected(
        index
            .search(&spectra[0])
            .expect("direct search should work"),
        4,
        0.9,
    );

    let mut indexed_state = index.new_search_state();
    let indexed = index
        .search_top_k_threshold_indexed_with_state(0, 4, 0.9, &mut indexed_state)
        .expect("indexed entropy top-k should work");
    assert_results_close(
        indexed,
        expected.clone(),
        "indexed entropy block-pruned top-k",
    );

    let indexed_diagnostics = indexed_state.diagnostics();
    assert_eq!(indexed_diagnostics.spectrum_blocks_evaluated, 2);
    assert_eq!(indexed_diagnostics.spectrum_blocks_allowed, 1);
    assert_eq!(indexed_diagnostics.spectrum_blocks_pruned, 1);
    assert_eq!(indexed_diagnostics.candidates_marked, 256);

    let mut external_state = index.new_search_state();
    let external = index
        .search_top_k_threshold_with_state(&spectra[0], 4, 0.9, &mut external_state)
        .expect("external entropy top-k should work");
    assert_results_close(external, expected, "external entropy block-pruned top-k");

    let external_diagnostics = external_state.diagnostics();
    assert_eq!(external_diagnostics.spectrum_blocks_evaluated, 2);
    assert_eq!(external_diagnostics.spectrum_blocks_allowed, 1);
    assert_eq!(external_diagnostics.spectrum_blocks_pruned, 1);
    assert_eq!(external_diagnostics.candidates_marked, 256);
}

#[test]
fn entropy_index_preserves_public_ids_after_default_reordering() {
    let spectra = [
        make_spectrum_f64(700.0, &[(400.0, 10.0), (450.0, 20.0)]),
        make_spectrum_f64(500.0, &[(100.0, 10.0), (150.0, 20.0)]),
        make_spectrum_f64(500.0, &[(100.05, 11.0), (150.05, 19.0)]),
        make_spectrum_f64(700.0, &[(400.05, 11.0), (450.05, 19.0)]),
    ];
    let index = FlashEntropyIndex::<f64>::unweighted(0.1_f64, spectra.iter())
        .expect("entropy index should build");

    for query_id in 0..spectra.len() as u32 {
        let mut state = index.new_search_state();
        let hits = index
            .search_top_k_threshold_indexed_with_state(query_id, 3, 0.9, &mut state)
            .expect("indexed top-k should work");

        assert!(
            hits.iter()
                .any(|hit| hit.spectrum_id == query_id && hit.score > 0.999),
            "query {query_id} should retain its public self id"
        );
    }
}

#[test]
fn indexed_entropy_queries_validate_ids_and_thresholds() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::<f64>::weighted(0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");
    let mut state = index.new_search_state();
    let out_of_bounds_id = spectra.len() as u32;

    assert_eq!(
        index
            .search_indexed_with_state(out_of_bounds_id, &mut state)
            .expect_err("out-of-bounds indexed search should fail"),
        SimilarityComputationError::IndexOverflow
    );
    assert_eq!(
        index
            .search_threshold_indexed_with_state(out_of_bounds_id, 0.5, &mut state)
            .expect_err("out-of-bounds indexed threshold search should fail"),
        SimilarityComputationError::IndexOverflow
    );
    assert_eq!(
        index
            .search_top_k_threshold_indexed_with_state(out_of_bounds_id, 2, 0.5, &mut state)
            .expect_err("out-of-bounds indexed threshold top-k should fail"),
        SimilarityComputationError::IndexOverflow
    );

    assert_eq!(
        index
            .search_threshold_indexed_with_state(0, f64::NAN, &mut state)
            .expect_err("non-finite indexed threshold should fail"),
        SimilarityComputationError::NonFiniteValue("score_threshold")
    );
    assert_eq!(
        index
            .search_top_k_threshold_indexed_with_state(0, 2, f64::NAN, &mut state)
            .expect_err("non-finite indexed top-k threshold should fail"),
        SimilarityComputationError::NonFiniteValue("score_threshold")
    );
}

#[test]
fn modified_search_with_state_reuses_buffers_without_leaking_matches() {
    let library = [
        make_spectrum_f64(300.0, &[(100.0, 10.0), (200.0, 5.0)]),
        make_spectrum_f64(500.0, &[(50.0, 3.0)]),
    ];
    let query = make_spectrum_f64(310.0, &[(100.0, 10.0), (210.0, 5.0)]);
    let nonmatching_query = make_spectrum_f64(700.0, &[(400.0, 9.0)]);

    let index = FlashEntropyIndex::<f64>::new(0.0_f64, 1.0_f64, 0.1_f64, false, library.iter())
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

    let nan_power =
        FlashEntropyIndex::<f64>::new(f64::NAN, 1.0, 0.1, true, spectra.iter().map(|(_, s)| s));
    assert!(matches!(
        nan_power,
        Err(FlashEntropyIndexError::Config(
            SimilarityConfigError::NonFiniteParameter("mz_power")
        ))
    ));

    let inf_intensity = FlashEntropyIndex::<f64>::new(
        0.0,
        f64::INFINITY,
        0.1,
        true,
        spectra.iter().map(|(_, s)| s),
    );
    assert!(matches!(
        inf_intensity,
        Err(FlashEntropyIndexError::Config(
            SimilarityConfigError::NonFiniteParameter("intensity_power")
        ))
    ));

    let nan_tolerance =
        FlashEntropyIndex::<f64>::new(0.0, 1.0, f64::NAN, true, spectra.iter().map(|(_, s)| s));
    assert!(matches!(
        nan_tolerance,
        Err(FlashEntropyIndexError::Config(
            SimilarityConfigError::NonFiniteParameter("mz_tolerance")
        ))
    ));

    let bad_library = RawSpectrum {
        precursor_mz: f64::NAN,
        peaks: vec![(100.0, 1.0)],
    };
    let build_error = FlashEntropyIndex::<f64>::new(0.0, 1.0, 0.1, true, [&bad_library]);
    assert!(matches!(
        build_error,
        Err(FlashEntropyIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("precursor_mz")
        ))
    ));

    let index = FlashEntropyIndex::<f64>::new(
        0.0_f64,
        1.0_f64,
        0.1_f64,
        true,
        spectra.iter().map(|(_, s)| s),
    )
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
    let lib = make_spectrum_f64(300.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let query = make_spectrum_f64(310.0, &[(100.0, 10.0), (210.0, 5.0)]);

    let index = FlashEntropyIndex::<f64>::new(0.0_f64, 1.0_f64, 0.1_f64, false, [&lib])
        .expect("index build should succeed");

    let direct_results = index.search(&query).expect("direct search should succeed");
    let modified_results = index
        .search_modified(&query)
        .expect("modified search should succeed");

    let direct = direct_results.iter().find(|r| r.spectrum_id == 0);
    assert!(direct.is_some());
    assert_eq!(direct.unwrap().n_matches, 1);

    let modified = modified_results.iter().find(|r| r.spectrum_id == 0);
    assert!(modified.is_some());
    assert_eq!(modified.unwrap().n_matches, 2);
    assert!(modified.unwrap().score > direct.unwrap().score);
}

#[test]
fn modified_search_anti_double_counting() {
    let lib = make_spectrum_f64(200.0, &[(100.0, 10.0)]);
    let query = make_spectrum_f64(200.0, &[(100.0, 10.0)]);

    let index = FlashEntropyIndex::<f64>::new(0.0_f64, 1.0_f64, 0.1_f64, false, [&lib])
        .expect("index build should succeed");

    let direct = index.search(&query).expect("search should succeed");
    let modified = index
        .search_modified(&query)
        .expect("modified search should succeed");

    assert_eq!(direct[0].n_matches, 1);
    assert_eq!(modified[0].n_matches, 1);
    assert!((direct[0].score - modified[0].score).abs() < 1e-12);
}

// ---------- well-separated precondition ----------

#[test]
fn rejects_non_well_separated_library() {
    let bad = make_spectrum_f64(200.0, &[(100.0, 10.0), (100.15, 8.0)]);
    let result = FlashEntropyIndex::<f64>::new(0.0_f64, 1.0_f64, 0.1_f64, true, [&bad]);
    assert!(result.is_err());
}

#[test]
fn rejects_non_well_separated_query() {
    let good = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 8.0)]);
    let index = FlashEntropyIndex::<f64>::new(0.0_f64, 1.0_f64, 0.1_f64, true, [&good])
        .expect("index build should succeed");

    let bad = make_spectrum_f64(200.0, &[(100.0, 10.0), (100.15, 8.0)]);
    assert!(index.search(&bad).is_err());
    assert!(index.search_top_k(&bad, 0).is_err());
}
