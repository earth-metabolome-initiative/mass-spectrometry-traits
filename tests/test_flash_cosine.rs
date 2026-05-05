//! Tests for the FlashCosineIndex.
//!
//! Verifies exact equivalence with LinearCosine on well-separated spectra,
//! self-similarity, symmetry, empty/edge cases, and modified search.

use mass_spectrometry::prelude::{
    CocaineSpectrum, FlashCosineIndex, FlashCosineIndexError, FlashCosineThresholdIndex,
    FlashIndexBuildPhase, FlashIndexBuildProgress, FlashSearchResult, GenericSpectrum,
    GlucoseSpectrum, HydroxyCholesterolSpectrum, LinearCosine, PhenylalanineSpectrum,
    SalicinSpectrum, ScalarSimilarity, SimilarityComputationError, SimilarityConfigError,
    SpectraIndex, SpectraIndexBuilder, Spectrum, SpectrumAlloc, SpectrumMut, TopKSearchState,
};
#[cfg(feature = "rayon")]
use mass_spectrometry::prelude::{FlashCosineSelfSimilarityIndex, PepmassFilter};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[path = "support/progress.rs"]
mod progress;

use progress::{ProgressEvent, RecordingProgress, assert_progress_reports_phase};

fn make_spectrum_f64(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor, peaks.len()).expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum.add_peak(mz, intensity).expect("valid sorted peak");
    }
    spectrum
}

fn build_cosine_index<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    spectra: I,
) -> Result<FlashCosineIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineIndex::<f64>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .build(&spectra)
}

fn build_cosine_index_with_progress<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    spectra: I,
    progress: &(dyn FlashIndexBuildProgress + Sync),
) -> Result<FlashCosineIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineIndex::<f64>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .progress(progress)
        .build(&spectra)
}

fn build_threshold_index<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    score_threshold: f64,
    spectra: I,
) -> Result<FlashCosineThresholdIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineThresholdIndex::<f64>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .score_threshold(score_threshold)
        .build(&spectra)
}

fn build_threshold_index_with_progress<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    score_threshold: f64,
    spectra: I,
    progress: &(dyn FlashIndexBuildProgress + Sync),
) -> Result<FlashCosineThresholdIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineThresholdIndex::<f64>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .score_threshold(score_threshold)
        .progress(progress)
        .build(&spectra)
}

fn build_cosine_index_with_pepmass<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    pepmass_tolerance: f64,
    spectra: I,
) -> Result<FlashCosineIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineIndex::<f64>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .pepmass_tolerance(pepmass_tolerance)
        .map_err(FlashCosineIndexError::Config)?
        .build(&spectra)
}

fn build_cosine_index_with_pepmass_progress<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    pepmass_tolerance: f64,
    spectra: I,
    progress: &(dyn FlashIndexBuildProgress + Sync),
) -> Result<FlashCosineIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineIndex::<f64>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .pepmass_tolerance(pepmass_tolerance)
        .map_err(FlashCosineIndexError::Config)?
        .progress(progress)
        .build(&spectra)
}

fn build_threshold_index_with_pepmass<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    score_threshold: f64,
    pepmass_tolerance: f64,
    spectra: I,
) -> Result<FlashCosineThresholdIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineThresholdIndex::<f64>::builder()
        .mz_power(mz_power)
        .intensity_power(intensity_power)
        .mz_tolerance(mz_tolerance)
        .score_threshold(score_threshold)
        .pepmass_tolerance(pepmass_tolerance)
        .map_err(FlashCosineIndexError::Config)?
        .build(&spectra)
}

#[cfg(feature = "rayon")]
fn build_self_similarity_index<'a, S, I>(
    score_threshold: f64,
    top_k: usize,
    pepmass_tolerance: f64,
    spectra: I,
) -> Result<FlashCosineSelfSimilarityIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(score_threshold)
        .top_k(top_k)
        .pepmass_tolerance(pepmass_tolerance)
        .map_err(FlashCosineIndexError::Config)?
        .parallel()
        .build(&spectra)
}

#[cfg(feature = "rayon")]
fn build_self_similarity_index_with_progress<'a, S, I>(
    score_threshold: f64,
    top_k: usize,
    pepmass_tolerance: f64,
    spectra: I,
    progress: &(dyn FlashIndexBuildProgress + Sync),
) -> Result<FlashCosineSelfSimilarityIndex<f64>, FlashCosineIndexError>
where
    S: Spectrum<Precision = f64> + Clone + Sync + 'a,
    I: IntoIterator<Item = &'a S>,
{
    let spectra: Vec<S> = spectra.into_iter().cloned().collect();
    FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(score_threshold)
        .top_k(top_k)
        .pepmass_tolerance(pepmass_tolerance)
        .map_err(FlashCosineIndexError::Config)?
        .parallel()
        .progress(progress)
        .build(&spectra)
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

fn top_k_expected(
    mut results: Vec<FlashSearchResult>,
    k: usize,
    score_threshold: f64,
) -> Vec<FlashSearchResult> {
    results.retain(|result| result.score >= score_threshold);
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

#[test]
fn index_build_progress_reports_construction_phases() {
    let library = [
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(501.0, &[(100.0, 10.0), (300.0, 20.0)]),
    ];
    let precursor_progress_len = 2 * library.len() as u64
        + 6 * library
            .iter()
            .map(|spectrum| spectrum.len() as u64)
            .sum::<u64>();

    let cosine_progress = RecordingProgress::default();
    let cosine = build_cosine_index_with_progress(0.0, 1.0, 0.1, library.iter(), &cosine_progress)
        .expect("cosine index should build");
    assert_eq!(cosine.n_spectra(), 2);
    let cosine_events = cosine_progress.events();
    assert_progress_reports_phase(
        &cosine_events,
        FlashIndexBuildPhase::PrepareSpectra,
        Some(2),
    );
    assert_progress_reports_phase(
        &cosine_events,
        FlashIndexBuildPhase::ReorderSpectra,
        Some(1),
    );
    assert_progress_reports_phase(
        &cosine_events,
        FlashIndexBuildPhase::PackFlashPeaks,
        Some(2),
    );
    assert_progress_reports_phase(
        &cosine_events,
        FlashIndexBuildPhase::SortProductIndex,
        Some(1),
    );
    assert_progress_reports_phase(
        &cosine_events,
        FlashIndexBuildPhase::SortNeutralLossIndex,
        Some(1),
    );
    assert!(
        !cosine_events.iter().any(|event| matches!(
            event,
            ProgressEvent::Phase(FlashIndexBuildPhase::BuildPrecursorIndex, _)
        )),
        "PEPMASS 2D index should not be built during ordinary index construction"
    );
    assert!(cosine_events.contains(&ProgressEvent::Finish));

    let cosine = build_cosine_index_with_pepmass_progress(
        0.0,
        1.0,
        0.1,
        0.5,
        library.iter(),
        &cosine_progress,
    )
    .expect("pepmass filter should be valid");
    assert_eq!(cosine.pepmass_filter().tolerance(), Some(0.5));
    let cosine_events = cosine_progress.events();
    assert_progress_reports_phase(
        &cosine_events,
        FlashIndexBuildPhase::BuildPrecursorIndex,
        Some(precursor_progress_len),
    );
    assert!(
        cosine_events.contains(&ProgressEvent::Inc(precursor_progress_len)),
        "PEPMASS 2D index progress did not advance by expected length: {cosine_events:?}"
    );

    let threshold_progress = RecordingProgress::default();
    build_threshold_index_with_progress(0.0, 1.0, 0.1, 0.8, library.iter(), &threshold_progress)
        .expect("threshold index should build");
    let threshold_events = threshold_progress.events();
    assert_progress_reports_phase(
        &threshold_events,
        FlashIndexBuildPhase::BuildBlockUpperBounds,
        Some(1),
    );
    assert_progress_reports_phase(
        &threshold_events,
        FlashIndexBuildPhase::BuildBlockProductIndex,
        Some(1),
    );
    assert!(threshold_events.contains(&ProgressEvent::Finish));
}

#[cfg(feature = "indicatif")]
#[test]
fn indicatif_progress_bar_can_build_cosine_index() {
    let library = [
        make_spectrum_f64(500.0, &[(100.0, 10.0)]),
        make_spectrum_f64(501.0, &[(100.0, 10.0)]),
    ];
    let progress = indicatif::ProgressBar::hidden();
    let index = build_cosine_index_with_progress(0.0, 1.0, 0.1, library.iter(), &progress)
        .expect("index should build with indicatif progress");
    assert_eq!(index.n_spectra(), 2);

    let index =
        build_cosine_index_with_pepmass_progress(0.0, 1.0, 0.1, 0.5, library.iter(), &progress)
            .expect("same indicatif progress bar should be reusable for PEPMASS build");
    assert_eq!(index.pepmass_filter().tolerance(), Some(0.5));
}

// ---------- self-similarity: flash score ~1.0 for each spectrum ----------

#[test]
fn self_similarity_all_reference() {
    let spectra = reference_spectra();

    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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

    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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
    let index = build_cosine_index(0.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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
    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, empty)
        .expect("empty index build should succeed");
    assert_eq!(index.n_spectra(), 0);

    let query: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
    let results = index.search(&query).expect("search should succeed");
    assert!(results.is_empty());
}

#[test]
fn empty_query() {
    let spectra = reference_spectra();
    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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

    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, library.iter())
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
    let result = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, [&bad]);
    assert!(result.is_err());
}

#[test]
fn rejects_non_well_separated_query() {
    let good = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 8.0)]);
    let index =
        build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, [&good]).expect("index build should succeed");

    let bad = make_spectrum_f64(200.0, &[(100.0, 10.0), (100.15, 8.0)]);
    let result = index.search(&bad);
    assert!(result.is_err());
    assert!(index.search_top_k(&bad, 0).is_err());
}

// ---------- single-spectrum library ----------

#[test]
fn single_spectrum_library() {
    let cocaine: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, [&cocaine])
        .expect("index build should succeed");

    let results = index.search(&cocaine).expect("search should succeed");
    assert_eq!(results.len(), 1);
    assert!((1.0 - results[0].score).abs() < 1e-10);
}

#[test]
fn accessors_and_search_with_state_match_stateless_results() {
    let spectra = reference_spectra();
    let index = build_cosine_index(0.5_f64, 2.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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
    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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
    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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
fn top_k_matches_sorted_direct_search_and_reuses_state() {
    let spectra = reference_spectra();
    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");

    let query_a = &spectra[0].1;
    let query_b = &spectra[1].1;
    let mut state = index.new_search_state();

    let expected_a = top_k_expected(
        index.search(query_a).expect("search should succeed"),
        3,
        0.0,
    );
    let actual_a = index
        .search_top_k_with_state(query_a, 3, &mut state)
        .expect("top-k search should succeed");
    assert_results_close(actual_a, expected_a, "top-k query a");

    let expected_b = top_k_expected(
        index.search(query_b).expect("search should succeed"),
        2,
        0.0,
    );
    let actual_b = index
        .search_top_k_with_state(query_b, 2, &mut state)
        .expect("top-k state reuse should succeed");
    assert_results_close(actual_b, expected_b, "top-k query b");

    assert!(
        index
            .search_top_k(query_a, 0)
            .expect("zero-k search should succeed")
            .is_empty()
    );
}

#[test]
fn top_k_threshold_matches_filtered_direct_search() {
    let spectra = reference_spectra();
    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
        .expect("index build should succeed");
    let query = &spectra[0].1;

    for threshold in [0.0_f64, 0.5_f64, 0.9_f64, 1.1_f64] {
        let expected = top_k_expected(
            index.search(query).expect("search should succeed"),
            4,
            threshold,
        );
        let actual = index
            .search_top_k_threshold(query, 4, threshold)
            .expect("thresholded top-k search should succeed");
        assert_results_close(actual, expected, "thresholded top-k");
    }

    let expected = top_k_expected(index.search(query).expect("search should succeed"), 3, 0.9);
    let mut search_state = index.new_search_state();
    let mut top_k_state = TopKSearchState::new();
    let mut streamed = Vec::new();
    index
        .for_each_top_k_threshold_with_state(
            query,
            3,
            0.9,
            &mut search_state,
            &mut top_k_state,
            |result| streamed.push(result),
        )
        .expect("streamed thresholded top-k should succeed");
    assert_results_close(streamed, expected, "streamed thresholded top-k");

    let error = index
        .search_top_k_threshold(query, 4, f64::NAN)
        .expect_err("non-finite threshold must be rejected");
    assert_eq!(
        error,
        SimilarityComputationError::NonFiniteValue("score_threshold")
    );
}

#[test]
fn threshold_index_matches_filtered_direct_search() {
    let spectra = reference_spectra();
    let direct_index =
        build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
            .expect("direct index build should succeed");

    for threshold in [0.5_f64, 0.7_f64, 0.9_f64] {
        let threshold_index = build_threshold_index(
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
fn threshold_index_top_k_matches_thresholded_results_for_external_and_indexed_queries() {
    let spectra = reference_spectra();
    let direct_index =
        build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
            .expect("direct index build should succeed");

    let query_id = 0_u32;
    let query = &spectra[query_id as usize].1;

    for threshold in [0.85_f64, 0.9, 0.95] {
        let threshold_index = build_threshold_index(
            1.0_f64,
            1.0_f64,
            0.1_f64,
            threshold,
            spectra.iter().map(|(_, s)| s),
        )
        .expect("threshold index build should succeed");

        for k in [0_usize, 1, 2, 16] {
            let expected_external = top_k_expected(
                direct_index.search(query).expect("search should succeed"),
                k,
                threshold_index.score_threshold(),
            );
            let actual_external = threshold_index
                .search_top_k(query, k)
                .expect("external threshold top-k should succeed");
            assert_results_close(
                actual_external,
                expected_external,
                &format!("threshold top-k external query threshold={threshold} k={k}"),
            );

            let mut emitted = Vec::new();
            let mut state = threshold_index.new_search_state();
            threshold_index
                .for_each_indexed_with_state(query_id, &mut state, |hit| emitted.push(hit))
                .expect("indexed threshold search should succeed");
            let expected_indexed = top_k_expected(emitted, k, threshold_index.score_threshold());

            let actual_indexed = threshold_index
                .search_top_k_indexed_with_state(query_id, k, &mut state)
                .expect("indexed threshold top-k should succeed");
            assert_results_close(
                actual_indexed,
                expected_indexed.clone(),
                &format!("threshold top-k indexed query threshold={threshold} k={k}"),
            );

            let mut top_k_state = TopKSearchState::new();
            let mut streamed_indexed = Vec::new();
            threshold_index
                .for_each_top_k_indexed_with_state(
                    query_id,
                    k,
                    &mut state,
                    &mut top_k_state,
                    |result| streamed_indexed.push(result),
                )
                .expect("streamed indexed threshold top-k should succeed");
            assert_results_close(
                streamed_indexed,
                expected_indexed,
                &format!("streamed threshold top-k indexed query threshold={threshold} k={k}"),
            );
        }
    }
}

#[test]
fn pepmass_filter_limits_cosine_search_paths() {
    let library = [
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.4, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(505.0, &[(100.0, 10.0), (200.0, 20.0)]),
    ];
    let query = make_spectrum_f64(500.2, &[(100.0, 10.0), (200.0, 20.0)]);

    let index = build_cosine_index_with_pepmass(0.0, 1.0, 0.1, 0.5, library.iter())
        .expect("index build should succeed");
    assert!(index.pepmass_filter().is_enabled());

    let direct_ids: Vec<_> = sorted_results(index.search(&query).expect("search should work"))
        .into_iter()
        .map(|hit| hit.spectrum_id)
        .collect();
    assert_eq!(direct_ids, vec![0, 1]);

    let top_k_ids: Vec<_> = sorted_results(
        index
            .search_top_k_threshold(&query, 8, 0.8)
            .expect("top-k search should work"),
    )
    .into_iter()
    .map(|hit| hit.spectrum_id)
    .collect();
    assert_eq!(top_k_ids, vec![0, 1]);

    let modified_ids: Vec<_> = sorted_results(
        index
            .search_modified(&query)
            .expect("modified search should work"),
    )
    .into_iter()
    .map(|hit| hit.spectrum_id)
    .collect();
    assert_eq!(modified_ids, vec![0, 1]);

    let bad_query = RawSpectrum {
        precursor_mz: f64::NAN,
        peaks: vec![(100.0, 10.0)],
    };
    assert!(matches!(
        index.search(&bad_query),
        Err(SimilarityComputationError::NonFiniteValue(
            "query_precursor_mz"
        ))
    ));

    let threshold_index =
        build_threshold_index_with_pepmass(0.0, 1.0, 0.1, 0.8, 0.5, library.iter())
            .expect("threshold index should build");

    let threshold_ids: Vec<_> = sorted_results(
        threshold_index
            .search(&query)
            .expect("threshold search should work"),
    )
    .into_iter()
    .map(|hit| hit.spectrum_id)
    .collect();
    assert_eq!(threshold_ids, vec![0, 1]);

    let indexed_top_k_ids: Vec<_> = sorted_results(
        threshold_index
            .search_top_k_indexed(0, 8)
            .expect("indexed top-k should work"),
    )
    .into_iter()
    .map(|hit| hit.spectrum_id)
    .collect();
    assert_eq!(indexed_top_k_ids, vec![0, 1]);
}

#[test]
fn pepmass_filter_handles_bin_boundaries_without_false_hits() {
    let library = [
        make_spectrum_f64(100.0, &[(50.0, 10.0), (60.0, 20.0)]),
        make_spectrum_f64(100.5, &[(50.0, 10.0), (60.0, 20.0)]),
        make_spectrum_f64(100.9, &[(50.0, 10.0), (60.0, 20.0)]),
        make_spectrum_f64(99.5, &[(50.0, 10.0), (60.0, 20.0)]),
        make_spectrum_f64(99.49, &[(50.0, 10.0), (60.0, 20.0)]),
    ];
    let query = make_spectrum_f64(100.0, &[(50.0, 10.0), (60.0, 20.0)]);

    let index = build_cosine_index_with_pepmass(0.0, 1.0, 0.1, 0.5, library.iter())
        .expect("index build should succeed");

    let direct_ids: Vec<_> = sorted_results(index.search(&query).expect("search should work"))
        .into_iter()
        .map(|hit| hit.spectrum_id)
        .collect();
    assert_eq!(direct_ids, vec![0, 1, 3]);

    let modified_ids: Vec<_> = sorted_results(
        index
            .search_modified(&query)
            .expect("modified search should work"),
    )
    .into_iter()
    .map(|hit| hit.spectrum_id)
    .collect();
    assert_eq!(modified_ids, vec![0, 1, 3]);
}

#[test]
fn pepmass_2d_index_builds_with_requested_tolerance() {
    let library = [
        make_spectrum_f64(100.0, &[(50.0, 10.0)]),
        make_spectrum_f64(101.0, &[(50.0, 10.0)]),
    ];
    let progress = RecordingProgress::default();

    let index =
        build_cosine_index_with_pepmass_progress(0.0, 1.0, 0.1, 0.5, library.iter(), &progress)
            .expect("index build should succeed");
    assert_eq!(index.pepmass_filter().tolerance(), Some(0.5));
    let first_build_count = progress
        .events()
        .iter()
        .filter(|event| {
            matches!(
                event,
                ProgressEvent::Phase(FlashIndexBuildPhase::BuildPrecursorIndex, _)
            )
        })
        .count();
    assert_eq!(first_build_count, 1);

    let changed =
        build_cosine_index_with_pepmass_progress(0.0, 1.0, 0.1, 1.0, library.iter(), &progress)
            .expect("changed pepmass tolerance should build a fresh 2D index");
    assert_eq!(changed.pepmass_filter().tolerance(), Some(1.0));
    let changed_tolerance_build_count = progress
        .events()
        .iter()
        .filter(|event| {
            matches!(
                event,
                ProgressEvent::Phase(FlashIndexBuildPhase::BuildPrecursorIndex, _)
            )
        })
        .count();
    assert_eq!(changed_tolerance_build_count, first_build_count + 1);
}

#[test]
fn pepmass_filter_reduces_cosine_posting_scans() {
    let library: Vec<_> = (0..600)
        .map(|index| make_spectrum_f64(500.0 + index as f64, &[(100.0, 10.0)]))
        .collect();

    let unfiltered =
        build_cosine_index(0.0, 1.0, 0.1, library.iter()).expect("index build should succeed");
    let mut unfiltered_state = unfiltered.new_search_state();
    let unfiltered_hits = unfiltered
        .search_with_state(&library[0], &mut unfiltered_state)
        .expect("unfiltered search should work");
    assert_eq!(unfiltered_hits.len(), library.len());
    let unfiltered_visited = unfiltered_state.diagnostics().product_postings_visited;
    assert_eq!(unfiltered_visited, library.len());

    let filtered = build_cosine_index_with_pepmass(0.0, 1.0, 0.1, 0.1, library.iter())
        .expect("index build should succeed");
    let mut filtered_state = filtered.new_search_state();
    let filtered_hits = filtered
        .search_with_state(&library[0], &mut filtered_state)
        .expect("filtered search should work");
    let filtered_ids: Vec<_> = filtered_hits
        .into_iter()
        .map(|hit| hit.spectrum_id)
        .collect();
    assert_eq!(filtered_ids, vec![0]);

    let filtered_visited = filtered_state.diagnostics().product_postings_visited;
    assert!(
        filtered_visited <= unfiltered_visited / 2,
        "PEPMASS 2D index should reduce visited postings: {filtered_visited} vs {unfiltered_visited}"
    );
}

#[test]
fn invalid_pepmass_filter_tolerances_are_rejected() {
    let nan_result = FlashCosineIndex::<f64>::builder().pepmass_tolerance(f64::NAN);
    assert!(matches!(
        nan_result,
        Err(SimilarityConfigError::NonFiniteParameter(
            "pepmass_tolerance"
        ))
    ));

    let negative_result = FlashCosineIndex::<f64>::builder().pepmass_tolerance(-0.1);
    assert!(matches!(
        negative_result,
        Err(SimilarityConfigError::InvalidParameter("pepmass_tolerance"))
    ));
}

#[test]
fn threshold_indexed_top_k_reports_dynamic_block_candidate_diagnostics() {
    let spectra = reference_spectra();
    let threshold_index = build_threshold_index(
        1.0_f64,
        1.0_f64,
        0.1_f64,
        0.9_f64,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("threshold index build should succeed");

    let query_id = 0_u32;
    let mut state = threshold_index.new_search_state();
    let mut emitted = Vec::new();
    threshold_index
        .for_each_indexed_with_state(query_id, &mut state, |hit| emitted.push(hit))
        .expect("indexed threshold search should succeed");
    let threshold_diagnostics = state.diagnostics();

    let mut top_k_state = TopKSearchState::new();
    let mut top_k = Vec::new();
    threshold_index
        .for_each_top_k_indexed_with_state(query_id, 2, &mut state, &mut top_k_state, |hit| {
            top_k.push(hit)
        })
        .expect("indexed top-k threshold search should succeed");
    let top_k_diagnostics = state.diagnostics();

    assert_results_close(
        top_k,
        top_k_expected(emitted, 2, threshold_index.score_threshold()),
        "diagnostic top-k result",
    );
    assert!(top_k_diagnostics.product_postings_visited > 0);
    assert!(top_k_diagnostics.candidates_marked > 0);
    assert_eq!(
        top_k_diagnostics.candidates_rescored,
        top_k_diagnostics.candidates_marked
    );
    assert!(top_k_diagnostics.candidates_marked <= threshold_diagnostics.candidates_marked);
}

#[test]
fn threshold_index_top_k_prunes_low_bound_spectrum_blocks_without_losing_hits() {
    let high_similarity = make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 10.0), (300.0, 10.0)]);
    let low_similarity = make_spectrum_f64(500.0, &[(100.0, 10.0)]);

    let mut spectra = Vec::new();
    for _ in 0..1024 {
        spectra.push(high_similarity.clone());
    }
    for _ in 0..4 {
        spectra.push(low_similarity.clone());
    }

    let direct_index = build_cosine_index(0.0_f64, 1.0_f64, 0.1_f64, spectra.iter())
        .expect("direct index should build");
    let threshold_index = build_threshold_index(0.0_f64, 1.0_f64, 0.1_f64, 0.9_f64, spectra.iter())
        .expect("threshold index should build");

    let expected = top_k_expected(
        direct_index
            .search(&spectra[0])
            .expect("direct search should work"),
        4,
        threshold_index.score_threshold(),
    );

    let mut indexed_state = threshold_index.new_search_state();
    let indexed = threshold_index
        .search_top_k_indexed_with_state(0, 4, &mut indexed_state)
        .expect("indexed top-k should work");
    assert_results_close(indexed, expected.clone(), "indexed block-pruned top-k");

    let indexed_diagnostics = indexed_state.diagnostics();
    assert_eq!(indexed_diagnostics.spectrum_blocks_evaluated, 2);
    assert_eq!(indexed_diagnostics.spectrum_blocks_allowed, 1);
    assert_eq!(indexed_diagnostics.spectrum_blocks_pruned, 1);
    assert_eq!(indexed_diagnostics.candidates_marked, 1024);

    let mut external_state = threshold_index.new_search_state();
    let external = threshold_index
        .search_top_k_with_state(&spectra[0], 4, &mut external_state)
        .expect("external top-k should work");
    assert_results_close(external, expected, "external block-pruned top-k");

    let external_diagnostics = external_state.diagnostics();
    assert_eq!(external_diagnostics.spectrum_blocks_evaluated, 2);
    assert_eq!(external_diagnostics.spectrum_blocks_allowed, 1);
    assert_eq!(external_diagnostics.spectrum_blocks_pruned, 1);
    assert_eq!(external_diagnostics.candidates_marked, 1024);
}

#[test]
fn threshold_index_preserves_public_ids_after_default_reordering() {
    let spectra = [
        make_spectrum_f64(700.0, &[(400.0, 10.0), (450.0, 20.0)]),
        make_spectrum_f64(500.0, &[(100.0, 10.0), (150.0, 20.0)]),
        make_spectrum_f64(500.0, &[(100.05, 11.0), (150.05, 19.0)]),
        make_spectrum_f64(700.0, &[(400.05, 11.0), (450.05, 19.0)]),
    ];
    let index = build_threshold_index(0.0, 1.0, 0.1, 0.9, spectra.iter())
        .expect("threshold index should build");

    for query_id in 0..spectra.len() as u32 {
        let mut state = index.new_search_state();
        let hits = index
            .search_top_k_indexed_with_state(query_id, 3, &mut state)
            .expect("indexed top-k should work");

        assert!(
            hits.iter()
                .any(|hit| hit.spectrum_id == query_id && hit.score > 0.999),
            "query {query_id} should retain its public self id"
        );
    }
}

#[test]
fn threshold_index_validates_threshold_and_query_id() {
    let spectra = reference_spectra();
    let nan_threshold = build_threshold_index(
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

    let threshold_index = build_threshold_index(
        1.0_f64,
        1.0_f64,
        0.1_f64,
        1.1_f64,
        spectra.iter().map(|(_, s)| s),
    )
    .expect("threshold above one should build");

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

    let top_k_out_of_bounds = threshold_index
        .search_top_k_indexed_with_state(spectra.len() as u32, 1, &mut state)
        .expect_err("out-of-bounds top-k query id should fail");
    assert_eq!(
        top_k_out_of_bounds,
        SimilarityComputationError::IndexOverflow
    );
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_matches_threshold_index_rows_and_excludes_self() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0), (300.05, 30.0)]),
        make_spectrum_f64(500.2, &[(100.05, 10.0), (200.05, 20.0)]),
        make_spectrum_f64(502.0, &[(100.05, 10.0), (200.05, 20.0), (300.05, 30.0)]),
        make_spectrum_f64(700.0, &[(400.0, 10.0), (450.0, 20.0)]),
    ];
    let threshold = 0.5_f64;
    let top_k = 2_usize;
    let pepmass_tolerance = 0.5_f64;
    let threshold_index =
        build_threshold_index_with_pepmass(0.0, 1.0, 0.1, threshold, pepmass_tolerance, &spectra)
            .expect("threshold index should build");
    let self_index = build_self_similarity_index(threshold, top_k, pepmass_tolerance, &spectra)
        .expect("self-similarity index should build");

    assert_eq!(self_index.n_spectra(), spectra.len() as u32);
    assert_eq!(self_index.top_k(), top_k);
    assert_eq!(self_index.score_threshold(), threshold);
    assert_eq!(
        self_index.pepmass_filter().tolerance(),
        Some(pepmass_tolerance)
    );

    let mut rows: Vec<_> = (&self_index).into_par_iter().map(Result::unwrap).collect();
    rows.sort_by_key(|row| row.0);
    assert_eq!(
        rows.iter().map(|row| row.0).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );

    for (query_id, row) in rows {
        assert!(
            row.iter().all(|hit| hit.spectrum_id != query_id),
            "row {query_id} should not contain the query spectrum"
        );

        let mut state = threshold_index.new_search_state();
        let mut expected = Vec::new();
        threshold_index
            .for_each_indexed_with_state(query_id, &mut state, |hit| expected.push(hit))
            .expect("baseline threshold row should work");
        expected.retain(|hit| hit.spectrum_id != query_id);
        let expected = top_k_expected(expected, top_k, threshold);
        assert_results_close(row, expected, &format!("self row {query_id}"));
    }
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_matches_threshold_index_at_high_threshold_and_broad_pepmass() {
    let spectra = vec![
        make_spectrum_f64(
            500.0,
            &[(100.0, 20.0), (150.0, 80.0), (200.0, 120.0), (250.0, 40.0)],
        ),
        make_spectrum_f64(
            505.0,
            &[
                (100.004, 20.0),
                (150.004, 80.0),
                (200.004, 120.0),
                (250.004, 40.0),
            ],
        ),
        make_spectrum_f64(
            507.5,
            &[
                (99.996, 18.0),
                (149.996, 76.0),
                (199.996, 118.0),
                (249.996, 42.0),
            ],
        ),
        make_spectrum_f64(
            530.0,
            &[
                (100.002, 20.0),
                (150.002, 80.0),
                (200.002, 120.0),
                (250.002, 40.0),
            ],
        ),
        make_spectrum_f64(
            503.0,
            &[(101.0, 20.0), (151.0, 80.0), (201.0, 120.0), (251.0, 40.0)],
        ),
        make_spectrum_f64(
            900.0,
            &[(300.0, 20.0), (350.0, 80.0), (400.0, 120.0), (450.0, 40.0)],
        ),
    ];
    let mz_tolerance = 0.01_f64;
    let threshold = 0.9_f64;
    let top_k = 4_usize;
    let pepmass_tolerance = 10.0_f64;

    let threshold_index = build_threshold_index_with_pepmass(
        0.0,
        1.0,
        mz_tolerance,
        threshold,
        pepmass_tolerance,
        &spectra,
    )
    .expect("threshold index should build");
    let self_index = FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(mz_tolerance)
        .score_threshold(threshold)
        .top_k(top_k)
        .pepmass_tolerance(pepmass_tolerance)
        .unwrap()
        .parallel()
        .build(&spectra)
        .expect("self-similarity index should build");

    let mut rows: Vec<_> = (&self_index).into_par_iter().map(Result::unwrap).collect();
    rows.sort_by_key(|row| row.0);

    let mut expected_total_hits = 0_usize;
    let mut actual_total_hits = 0_usize;
    for (query_id, row) in rows {
        let mut state = threshold_index.new_search_state();
        let mut expected = Vec::new();
        threshold_index
            .for_each_indexed_with_state(query_id, &mut state, |hit| expected.push(hit))
            .expect("baseline threshold row should work");
        expected.retain(|hit| hit.spectrum_id != query_id);
        let expected = top_k_expected(expected, top_k, threshold);
        expected_total_hits += expected.len();
        actual_total_hits += row.len();
        assert_results_close(
            row,
            expected,
            &format!("high threshold self row {query_id}"),
        );
    }
    assert!(
        expected_total_hits > 0,
        "baseline threshold index should produce non-self hits"
    );
    assert!(
        actual_total_hits > 0,
        "high-threshold one-shot self-similarity should produce non-self hits"
    );
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_reports_construction_progress() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
    ];
    let progress = RecordingProgress::default();
    let index = build_self_similarity_index_with_progress(0.8, 1, 0.5, &spectra, &progress)
        .expect("self-similarity index should build");

    assert_eq!(index.n_spectra(), 2);
    let events = progress.events();
    assert_progress_reports_phase(&events, FlashIndexBuildPhase::PrepareSpectra, Some(2));
    assert_progress_reports_phase(
        &events,
        FlashIndexBuildPhase::BuildBlockUpperBounds,
        Some(1),
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            ProgressEvent::Phase(FlashIndexBuildPhase::BuildPrecursorIndex, _)
        )),
        "missing precursor index progress: {events:?}"
    );
    assert!(events.contains(&ProgressEvent::Finish));
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_can_iterate_a_contiguous_row_range() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
        make_spectrum_f64(500.2, &[(100.05, 8.0), (200.05, 18.0)]),
        make_spectrum_f64(700.0, &[(400.0, 10.0), (450.0, 20.0)]),
    ];
    let index = build_self_similarity_index(0.5, 2, 0.5, &spectra)
        .expect("self-similarity index should build");

    let mut rows: Vec<_> = index
        .rows()
        .range(1..3)
        .into_par_iter()
        .map(Result::unwrap)
        .collect();
    rows.sort_by_key(|row| row.0);

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
    assert_eq!(rows[1].0, 2);
    assert!(
        rows.iter()
            .all(|(query_id, row)| row.iter().all(|hit| hit.spectrum_id != *query_id))
    );
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_can_iterate_an_explicit_row_set() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
        make_spectrum_f64(500.2, &[(100.05, 8.0), (200.05, 18.0)]),
        make_spectrum_f64(700.0, &[(400.0, 10.0), (450.0, 20.0)]),
    ];
    let index = build_self_similarity_index(0.5, 2, 0.5, &spectra)
        .expect("self-similarity index should build");

    let row_ids = [2, 0];
    let mut rows: Vec<_> = index
        .rows()
        .ids(&row_ids)
        .into_par_iter()
        .map(Result::unwrap)
        .collect();
    rows.sort_by_key(|row| row.0);

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 0);
    assert_eq!(rows[1].0, 2);
    assert!(
        rows.iter()
            .all(|(query_id, row)| row.iter().all(|hit| hit.spectrum_id != *query_id))
    );
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_accessors_and_all_rows_work_after_sequential_build() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
        make_spectrum_f64(700.0, &[(400.0, 10.0), (450.0, 20.0)]),
    ];
    let index = FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(0.5)
        .top_k(2)
        .pepmass_tolerance(0.5)
        .unwrap()
        .sequential()
        .build(&spectra)
        .expect("sequential self-similarity index should build");

    assert_eq!(index.mz_power(), 0.0);
    assert_eq!(index.intensity_power(), 1.0);
    assert_eq!(index.tolerance(), 0.1);
    assert_eq!(index.score_threshold(), 0.5);
    assert_eq!(index.top_k(), 2);
    assert_eq!(index.n_spectra(), spectra.len() as u32);
    assert_eq!(index.pepmass_filter().tolerance(), Some(0.5));

    let rows: Vec<_> = index
        .rows()
        .all()
        .into_par_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(rows.len(), spectra.len());
    assert!(
        rows.iter()
            .all(|(query_id, row)| row.iter().all(|hit| hit.spectrum_id != *query_id))
    );
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_reports_row_iteration_progress() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
        make_spectrum_f64(500.2, &[(100.05, 8.0), (200.05, 18.0)]),
    ];
    let index = build_self_similarity_index(0.5, 2, 0.5, &spectra)
        .expect("self-similarity index should build");
    let progress = RecordingProgress::default();

    let row_ids = [2, 0];
    let rows: Vec<_> = index
        .rows()
        .ids(&row_ids)
        .progress(&progress)
        .into_par_iter()
        .map(Result::unwrap)
        .collect();

    assert_eq!(rows.len(), 2);
    let events = progress.events();
    assert!(events.contains(&ProgressEvent::RowStart(2)));
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                ProgressEvent::RowInc(delta) => Some(*delta),
                _ => None,
            })
            .sum::<u64>(),
        2
    );
    assert!(events.contains(&ProgressEvent::RowFinish));
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_reports_empty_row_iteration_progress() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
    ];
    let index = build_self_similarity_index(0.5, 2, 0.5, &spectra)
        .expect("self-similarity index should build");
    let progress = RecordingProgress::default();

    let rows: Vec<_> = index
        .rows()
        .range(1..1)
        .progress(&progress)
        .into_par_iter()
        .map(Result::unwrap)
        .collect();

    assert!(rows.is_empty());
    assert_eq!(
        progress.events(),
        vec![ProgressEvent::RowStart(0), ProgressEvent::RowFinish]
    );
}

#[cfg(feature = "rayon")]
#[test]
fn self_similarity_index_validates_fixed_profile() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
    ];

    let disabled_filter = FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(0.8)
        .top_k(1)
        .pepmass_filter(PepmassFilter::disabled())
        .build(&spectra);
    assert!(matches!(
        disabled_filter,
        Err(FlashCosineIndexError::Config(
            SimilarityConfigError::InvalidParameter("pepmass_filter")
        ))
    ));

    let zero_k = FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(0.8)
        .top_k(0)
        .pepmass_tolerance(0.5)
        .unwrap()
        .build(&spectra);
    assert!(matches!(
        zero_k,
        Err(FlashCosineIndexError::Config(
            SimilarityConfigError::InvalidParameter("top_k")
        ))
    ));

    let nan_threshold = FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(f64::NAN)
        .top_k(1)
        .pepmass_tolerance(0.5)
        .unwrap()
        .build(&spectra);
    assert!(matches!(
        nan_threshold,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("score_threshold")
        ))
    ));

    let invalid_pepmass = FlashCosineSelfSimilarityIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(0.8)
        .top_k(1)
        .pepmass_tolerance(-0.5)
        .map_err(FlashCosineIndexError::Config)
        .and_then(|builder| builder.build(&spectra));
    assert!(matches!(
        invalid_pepmass,
        Err(FlashCosineIndexError::Config(
            SimilarityConfigError::InvalidParameter("pepmass_tolerance")
        ))
    ));
}

#[cfg(all(feature = "rayon", feature = "indicatif"))]
#[test]
fn indicatif_progress_bar_can_build_and_iterate_self_similarity_index() {
    let spectra = vec![
        make_spectrum_f64(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum_f64(500.1, &[(100.05, 10.0), (200.05, 20.0)]),
    ];
    let build_progress = indicatif::ProgressBar::hidden();
    let index = build_self_similarity_index_with_progress(0.8, 1, 0.5, &spectra, &build_progress)
        .expect("self-similarity index should build with indicatif progress");
    let row_progress = indicatif::ProgressBar::hidden();
    let rows: Vec<_> = index
        .rows()
        .progress(&row_progress)
        .into_par_iter()
        .map(Result::unwrap)
        .collect();

    assert_eq!(rows.len(), 2);
}

#[test]
fn modified_search_with_state_reuses_buffers_without_leaking_matches() {
    let library = [
        make_spectrum_f64(300.0, &[(100.0, 10.0), (200.0, 5.0)]),
        make_spectrum_f64(500.0, &[(50.0, 3.0)]),
    ];
    let query = make_spectrum_f64(310.0, &[(100.0, 10.0), (210.0, 5.0)]);
    let nonmatching_query = make_spectrum_f64(700.0, &[(400.0, 9.0)]);

    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, library.iter())
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

    let nan_power = build_cosine_index(f64::NAN, 1.0, 0.1, spectra.iter().map(|(_, s)| s));
    assert!(matches!(
        nan_power,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("mz_power")
        ))
    ));

    let inf_intensity = build_cosine_index(1.0, f64::INFINITY, 0.1, spectra.iter().map(|(_, s)| s));
    assert!(matches!(
        inf_intensity,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("intensity_power")
        ))
    ));

    let nan_tolerance = build_cosine_index(1.0, 1.0, f64::NAN, spectra.iter().map(|(_, s)| s));
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
    let build_error = build_cosine_index(1.0, 1.0, 0.1, [&bad_library]);
    assert!(matches!(
        build_error,
        Err(FlashCosineIndexError::Computation(
            SimilarityComputationError::NonFiniteValue("precursor_mz")
        ))
    ));

    let index = build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, spectra.iter().map(|(_, s)| s))
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

    let index =
        build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, [&lib]).expect("index build should succeed");

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

    let index =
        build_cosine_index(1.0_f64, 1.0_f64, 0.1_f64, [&lib]).expect("index build should succeed");

    let direct = index.search(&query).expect("search should succeed");
    let modified = index
        .search_modified(&query)
        .expect("modified search should succeed");

    // Both should yield exactly 1 match due to anti-double-counting.
    assert_eq!(direct[0].n_matches, 1);
    assert_eq!(modified[0].n_matches, 1);
    assert!((direct[0].score - modified[0].score).abs() < 1e-12);
}
