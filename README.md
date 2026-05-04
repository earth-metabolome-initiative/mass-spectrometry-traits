# Mass Spectrometry

[![Crates.io](https://img.shields.io/crates/v/mass_spectrometry.svg)](https://crates.io/crates/mass_spectrometry)
[![Documentation](https://docs.rs/mass_spectrometry/badge.svg)](https://docs.rs/mass_spectrometry)
[![License](https://img.shields.io/crates/l/mass_spectrometry.svg)](https://github.com/earth-metabolome-initiative/mass-spectrometry-traits/blob/main/LICENSE)
[![no_std](https://img.shields.io/badge/no__std-default-success.svg)](https://docs.rust-embedded.org/book/intro/no-std.html)

A `no_std` crate for mass spectra, spectral similarities, fast search indices,
and SPLASH fingerprints. `GenericSpectrum<P>` stores sorted `(m/z, intensity)` peaks with `f64`, `f32`,
or `half::f16` precision. The crate includes cosine and entropy similarities,
fast search indices including cutoff-specialized cosine indices and exact top-k
queries for direct graph construction, SPLASH generation, and built-in reference
spectra for examples and regression tests.

## Examples

### Spectral Similarities

```rust
use mass_spectrometry::prelude::*;

let mut left: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
left.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();

let mut right: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
right.add_peaks([(100.05, 10.0), (200.05, 20.0)]).unwrap();

let cosine = LinearCosine::new(0.0, 1.0, 0.1).unwrap();
let (score, matches) = cosine.similarity(&left, &right).unwrap();

assert_eq!(matches, 2);
assert!(score > 0.99);

let cocaine: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
let (_reference_score, reference_matches) = cosine.similarity(&cocaine, &cocaine).unwrap();
assert!(reference_matches > 0);
```

### SPLASH

```rust
use mass_spectrometry::prelude::*;

let mut spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(250.0, 2).unwrap();
spectrum.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();

assert_eq!(
    spectrum.splash().unwrap(),
    "splash10-0udi-0490000000-4425acda10ed7d4709bd"
);
```

### Indices

The regular cosine and entropy indices accept a cutoff at query time. `FlashCosineThresholdIndex` bakes one cutoff into the cosine index and is the intended path for thresholded indexed self-similarity; entropy keeps query-time thresholding only. `FlashCosineIndex`, `FlashCosineThresholdIndex`, and `FlashEntropyIndex` implement `SpectraIndex` for external-query search and top-k search with reusable scratch state. Flash indices can also enable an optional PEPmass/precursor m/z filter; when enabled, search uses a lazily built precursor-ordered posting index so spectra outside the configured precursor-mass window are skipped before most product-ion postings are scanned.

Flash indices are generic over their stored peak precision, so the default `f64` examples below can be switched to `FlashCosineIndex::<f32>`, `FlashCosineThresholdIndex::<f32>`, or `FlashEntropyIndex::<f32>` when index memory is the limiting factor; half precision is also available for spectra whose m/z and intensity values remain representable at that precision.

Index construction can report progress through `new_with_progress` constructors. The crate provides a small `FlashIndexBuildProgress` trait for custom reporters, and with the `indicatif` feature enabled an `indicatif::ProgressBar` can be passed directly. The PEPMASS reverse index is only built when a PEPMASS filter is enabled; use the `with_pepmass_tolerance_and_progress` variants if that lazy build should also report progress.

```rust
use mass_spectrometry::prelude::*;

let mut a: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
a.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();

let mut b: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
b.add_peaks([(100.05, 10.0), (200.05, 20.0)]).unwrap();

let mut c: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
c.add_peaks([(300.0, 10.0), (400.0, 20.0)]).unwrap();

let spectra = vec![a, b, c];

let index = FlashCosineIndex::<f64>::new(0.0, 1.0, 0.1, &spectra).unwrap();
let hits = index.search_threshold(&spectra[0], 0.8).unwrap();
assert!(hits.iter().any(|hit| hit.spectrum_id == 1));

let pepmass_index = FlashCosineIndex::<f64>::new(0.0, 1.0, 0.1, &spectra)
    .unwrap()
    .with_pepmass_tolerance(0.5)
    .unwrap();
assert_eq!(pepmass_index.pepmass_filter().tolerance(), Some(0.5));

let best = index.search_top_k_threshold(&spectra[0], 2, 0.8).unwrap();
assert_eq!(best[0].spectrum_id, 0);
assert!(best.iter().any(|hit| hit.spectrum_id == 1));

let threshold_index =
    FlashCosineThresholdIndex::<f64>::new(0.0, 1.0, 0.1, 0.8, &spectra).unwrap();
let indexed_best = threshold_index.search_top_k_indexed(0, 2).unwrap();
assert_eq!(indexed_best[0].spectrum_id, 0);
assert!(indexed_best.iter().any(|hit| hit.spectrum_id == 1));

// Reuse both scratch buffers across queries when scanning an indexed library.
// `SearchState` owns the per-query candidate buffers and diagnostics.
let mut state = threshold_index.new_search_state();
// `TopKSearchState` owns the heap/result buffers for maintaining the best k hits.
let mut top_k_state = TopKSearchState::new();
let mut edges = Vec::new();

for query_id in 0..threshold_index.n_spectra() {
    threshold_index
        .for_each_top_k_indexed_with_state(
            query_id,
            2,
            &mut state,
            &mut top_k_state,
            |hit| {
                if hit.spectrum_id > query_id {
                    edges.push((query_id, hit.spectrum_id, hit.score));
                }
            },
        )
        .unwrap();
}

assert_eq!(edges.len(), 1);
assert_eq!(edges[0].0, 0);
assert_eq!(edges[0].1, 1);
assert!(edges[0].2 > 0.99);

let entropy_index = FlashEntropyIndex::<f64>::weighted(0.1, &spectra).unwrap();
let entropy_best = entropy_index.search_top_k_threshold(&spectra[0], 2, 0.8).unwrap();
assert_eq!(entropy_best[0].spectrum_id, 0);
assert!(entropy_best.iter().any(|hit| hit.spectrum_id == 1));

#[cfg(feature = "indicatif")]
{
    let progress = indicatif::ProgressBar::new(0);
    let _index =
        FlashCosineIndex::<f64>::new_with_progress(0.0, 1.0, 0.1, &spectra, &progress)
            .unwrap()
            .with_pepmass_tolerance_and_progress(0.5, &progress)
            .unwrap();
}
```
