//! Tests for the FlashCosineIndex.
//!
//! Verifies exact equivalence with LinearCosine on well-separated spectra,
//! self-similarity, symmetry, empty/edge cases, and modified search.

use mass_spectrometry::prelude::{
    CocaineSpectrum, FlashCosineIndex, GenericSpectrum, GlucoseSpectrum,
    HydroxyCholesterolSpectrum, LinearCosine, PhenylalanineSpectrum, SalicinSpectrum,
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
    let empty: Vec<&GenericSpectrum<f64, f64>> = Vec::new();
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, empty)
        .expect("empty index build should succeed");
    assert_eq!(index.n_spectra(), 0);

    let query: GenericSpectrum<f64, f64> = GenericSpectrum::cocaine().unwrap();
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
            assert!(
                r.score.abs() < 1e-12,
                "empty spectrum should score 0"
            );
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
    let cocaine: GenericSpectrum<f64, f64> = GenericSpectrum::cocaine().unwrap();
    let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, [&cocaine])
        .expect("index build should succeed");

    let results = index.search(&cocaine).expect("search should succeed");
    assert_eq!(results.len(), 1);
    assert!((1.0 - results[0].score).abs() < 1e-10);
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
