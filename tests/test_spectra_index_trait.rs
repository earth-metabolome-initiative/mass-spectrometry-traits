//! Tests for the common SpectraIndex trait across Flash index variants.

use mass_spectrometry::prelude::{
    FlashCosineIndex, FlashCosineThresholdIndex, FlashEntropyIndex, GenericSpectrum,
    SimilarityConfigError, SpectraIndex, SpectraIndexSetupError, SpectrumAlloc, SpectrumMut,
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

    let mut state = index.new_search_state();
    let hits = index
        .search_with_state(query, &mut state)
        .expect("trait search should succeed");
    assert!(hits.iter().any(|hit| hit.spectrum_id == 0));

    let top = index
        .search_top_k_with_state(query, 1, &mut state)
        .expect("trait top-k should succeed");
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].spectrum_id, 0);

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

    let index = index
        .with_pepmass_tolerance(0.5)
        .expect("valid PEPMASS tolerance should be accepted through SpectraIndex");
    assert_eq!(index.pepmass_filter().tolerance(), Some(0.5));

    let index = index.without_pepmass_filter();
    assert!(!index.pepmass_filter().is_enabled());
}

fn assert_trait_rejects_invalid_pepmass_tolerance<I>(index: I)
where
    I: SpectraIndex,
{
    assert!(matches!(
        index.with_pepmass_tolerance(f64::NAN),
        Err(SpectraIndexSetupError::Config(
            SimilarityConfigError::NonFiniteParameter("pepmass_tolerance")
        ))
    ));
}

#[test]
fn spectra_index_trait_covers_all_flash_index_variants() {
    let spectra = spectra();
    let query = &spectra[0];

    let cosine = FlashCosineIndex::<f64>::new(0.0, 1.0, 0.1, &spectra).unwrap();
    assert_trait_surface(&cosine, query);

    let threshold_cosine =
        FlashCosineThresholdIndex::<f64>::new(0.0, 1.0, 0.1, 0.8, &spectra).unwrap();
    assert_trait_surface(&threshold_cosine, query);

    let entropy = FlashEntropyIndex::<f64>::weighted(0.1, &spectra).unwrap();
    assert_trait_surface(&entropy, query);
}

#[test]
fn spectra_index_trait_covers_pepmass_filter_defaults() {
    let spectra = spectra();

    assert_trait_pepmass_defaults(FlashCosineIndex::<f64>::new(0.0, 1.0, 0.1, &spectra).unwrap());
    assert_trait_pepmass_defaults(
        FlashCosineThresholdIndex::<f64>::new(0.0, 1.0, 0.1, 0.8, &spectra).unwrap(),
    );
    assert_trait_pepmass_defaults(FlashEntropyIndex::<f64>::weighted(0.1, &spectra).unwrap());

    assert_trait_rejects_invalid_pepmass_tolerance(
        FlashCosineIndex::<f64>::new(0.0, 1.0, 0.1, &spectra).unwrap(),
    );
    assert_trait_rejects_invalid_pepmass_tolerance(
        FlashCosineThresholdIndex::<f64>::new(0.0, 1.0, 0.1, 0.8, &spectra).unwrap(),
    );
    assert_trait_rejects_invalid_pepmass_tolerance(
        FlashEntropyIndex::<f64>::weighted(0.1, &spectra).unwrap(),
    );
}
