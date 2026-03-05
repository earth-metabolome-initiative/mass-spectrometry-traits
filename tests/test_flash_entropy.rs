//! Tests for the FlashEntropyIndex.
//!
//! Verifies exact equivalence with LinearEntropy on well-separated spectra,
//! self-similarity, empty/edge cases, and modified search.

use mass_spectrometry::prelude::{
    CocaineSpectrum, FlashEntropyIndex, GenericSpectrum, GlucoseSpectrum,
    HydroxyCholesterolSpectrum, LinearEntropy, PhenylalanineSpectrum, SalicinSpectrum,
    ScalarSimilarity, Spectrum, SpectrumAlloc, SpectrumMut,
};

fn make_spectrum_f64(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum<f64, f64> {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor, peaks.len()).expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum.add_peak(mz, intensity).expect("valid sorted peak");
    }
    spectrum
}

fn reference_spectra() -> Vec<(&'static str, GenericSpectrum<f64, f64>)> {
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

// ---------- self-similarity (weighted) ----------

#[test]
fn self_similarity_weighted() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::new(
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
    let index = FlashEntropyIndex::new(
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
    let index = FlashEntropyIndex::new(
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
    let index = FlashEntropyIndex::new(
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
    let empty: Vec<&GenericSpectrum<f64, f64>> = Vec::new();
    let index = FlashEntropyIndex::new(0.0_f64, 1.0_f64, 0.1_f64, true, empty)
        .expect("empty index should build");
    assert_eq!(index.n_spectra(), 0);

    let query: GenericSpectrum<f64, f64> = GenericSpectrum::cocaine().unwrap();
    let results = index.search(&query).expect("search should succeed");
    assert!(results.is_empty());
}

#[test]
fn empty_query() {
    let spectra = reference_spectra();
    let index = FlashEntropyIndex::new(
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
    let zero = make_spectrum_f64(200.0, &[(100.0, 0.0), (200.0, 0.0)]);
    let normal = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let library = [zero, normal];

    let index = FlashEntropyIndex::new(0.0_f64, 1.0_f64, 0.1_f64, true, library.iter())
        .expect("index build should succeed");

    let query = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let results = index.search(&query).expect("search should succeed");

    // Zero-intensity spectrum should not appear or score 0.
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

// ---------- modified search ----------

#[test]
fn modified_search_includes_shifted_matches() {
    let lib = make_spectrum_f64(300.0, &[(100.0, 10.0), (200.0, 5.0)]);
    let query = make_spectrum_f64(310.0, &[(100.0, 10.0), (210.0, 5.0)]);

    let index = FlashEntropyIndex::new(0.0_f64, 1.0_f64, 0.1_f64, false, [&lib])
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

    let index = FlashEntropyIndex::new(0.0_f64, 1.0_f64, 0.1_f64, false, [&lib])
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
    let result = FlashEntropyIndex::new(0.0_f64, 1.0_f64, 0.1_f64, true, [&bad]);
    assert!(result.is_err());
}

#[test]
fn rejects_non_well_separated_query() {
    let good = make_spectrum_f64(200.0, &[(100.0, 10.0), (200.0, 8.0)]);
    let index = FlashEntropyIndex::new(0.0_f64, 1.0_f64, 0.1_f64, true, [&good])
        .expect("index build should succeed");

    let bad = make_spectrum_f64(200.0, &[(100.0, 10.0), (100.15, 8.0)]);
    assert!(index.search(&bad).is_err());
}
