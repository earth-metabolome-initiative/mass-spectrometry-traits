//! Tests for the common SpectraIndex trait across Flash index variants.

use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineThresholdIndex, FlashEntropyIndex, GenericSpectrum,
    SimilarityConfigError, SpectraIndex, SpectraIndexBuilder, SpectrumAlloc, SpectrumMut,
    TopKSearchState,
};

fn make_spectrum(precursor: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor, peaks.len()).expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum.add_peak(mz, intensity).expect("valid sorted peak");
    }
    spectrum
}

fn spectra() -> Vec<GenericSpectrum> {
    vec![
        make_spectrum(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
        make_spectrum(500.0, &[(100.05, 10.0), (200.05, 20.0)]),
        make_spectrum(500.0, &[(300.0, 10.0), (400.0, 20.0)]),
    ]
}

fn assert_trait_surface<I>(index: &I, query: &GenericSpectrum)
where
    I: SpectraIndex,
{
    assert_eq!(index.n_spectra(), 3);
    assert_eq!(index.tolerance(), 0.1);

    let direct_hits = index.search(query).expect("trait search should succeed");
    assert!(direct_hits.iter().any(|hit| hit.spectrum_id == 0));

    let direct_top = index
        .search_top_k(query, 1)
        .expect("trait top-k should succeed");
    assert_eq!(direct_top.len(), 1);
    assert_eq!(direct_top[0].spectrum_id, 0);

    let mut state = index.new_search_state();
    let hits = index
        .search_with_state(query, &mut state)
        .expect("trait search should succeed");
    assert_eq!(hits, direct_hits);

    let top = index
        .search_top_k_with_state(query, 1, &mut state)
        .expect("trait top-k should succeed");
    assert_eq!(top, direct_top);

    let mut top_k_state = TopKSearchState::new();
    let mut streamed = Vec::new();
    index
        .for_each_top_k_with_state(query, 1, &mut state, &mut top_k_state, |hit| {
            streamed.push(hit);
        })
        .expect("trait streaming top-k should succeed");
    assert_eq!(streamed, top);
}

fn assert_trait_pepmass_defaults<I>(index: I)
where
    I: SpectraIndex,
{
    assert!(!index.pepmass_filter().is_enabled());
}

#[test]
fn spectra_index_trait_covers_all_flash_index_variants() {
    let spectra = spectra();
    let query = &spectra[0];

    let cosine = FlashCosineIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .build(&spectra)
        .unwrap();
    assert_trait_surface(&cosine, query);

    let threshold_cosine = FlashCosineThresholdIndex::<f64>::builder()
        .mz_power(0.0)
        .intensity_power(1.0)
        .mz_tolerance(0.1)
        .score_threshold(0.8)
        .build(&spectra)
        .unwrap();
    assert_trait_surface(&threshold_cosine, query);

    let entropy = FlashEntropyIndex::<f64>::builder()
        .mz_tolerance(0.1)
        .build(&spectra)
        .unwrap();
    assert_trait_surface(&entropy, query);
}

#[test]
fn spectra_index_trait_covers_pepmass_filter_defaults() {
    let spectra = spectra();

    assert_trait_pepmass_defaults(
        FlashCosineIndex::<f64>::builder()
            .mz_tolerance(0.1)
            .build(&spectra)
            .unwrap(),
    );
    assert_trait_pepmass_defaults(
        FlashCosineThresholdIndex::<f64>::builder()
            .mz_tolerance(0.1)
            .score_threshold(0.8)
            .build(&spectra)
            .unwrap(),
    );
    assert_trait_pepmass_defaults(
        FlashEntropyIndex::<f64>::builder()
            .mz_tolerance(0.1)
            .build(&spectra)
            .unwrap(),
    );

    assert!(matches!(
        FlashCosineIndex::<f64>::builder().pepmass_tolerance(f64::NAN),
        Err(SimilarityConfigError::NonFiniteParameter(
            "pepmass_tolerance"
        ))
    ));
}

#[test]
fn index_builders_configure_pepmass_filters() {
    let spectra = spectra();

    let cosine = FlashCosineIndex::<f64>::builder()
        .mz_tolerance(0.1)
        .pepmass_tolerance(0.5)
        .unwrap()
        .build(&spectra)
        .unwrap();
    assert_eq!(cosine.pepmass_filter().tolerance(), Some(0.5));
}

#[test]
fn index_builders_can_force_sequential_execution() {
    let spectra = spectra();

    let cosine = FlashCosineIndex::<f64>::builder()
        .mz_tolerance(0.1)
        .sequential()
        .build(&spectra)
        .unwrap();
    assert_trait_surface(&cosine, &spectra[0]);

    let threshold_cosine = FlashCosineThresholdIndex::<f64>::builder()
        .mz_tolerance(0.1)
        .score_threshold(0.8)
        .sequential()
        .build(&spectra)
        .unwrap();
    assert_trait_surface(&threshold_cosine, &spectra[0]);

    let entropy = FlashEntropyIndex::<f64>::builder()
        .mz_tolerance(0.1)
        .sequential()
        .build(&spectra)
        .unwrap();
    assert_trait_surface(&entropy, &spectra[0]);
}
