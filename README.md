# Mass Spectrometry

[![Crates.io](https://img.shields.io/crates/v/mass_spectrometry.svg)](https://crates.io/crates/mass_spectrometry)
[![Documentation](https://docs.rs/mass_spectrometry/badge.svg)](https://docs.rs/mass_spectrometry)
[![License](https://img.shields.io/crates/l/mass_spectrometry.svg)](https://github.com/earth-metabolome-initiative/mass-spectrometry-traits/blob/main/LICENSE)
[![no_std](https://img.shields.io/badge/no__std-default-success.svg)](https://docs.rust-embedded.org/book/intro/no-std.html)

A `no_std` crate for mass spectra, spectral similarities, fast search indices,
and SPLASH fingerprints. `GenericSpectrum<P>` stores sorted `(m/z, intensity)` peaks with `f64`, `f32`,
or `half::f16` precision. The crate includes cosine and entropy similarities,
fast search indices including cutoff-specialized cosine indices for direct graph
construction, SPLASH generation, and built-in reference spectra for examples and
regression tests.

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

let splash = splash_from_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();
assert_eq!(splash, "splash10-0udi-0490000000-4425acda10ed7d4709bd");

let mut spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(250.0, 2).unwrap();
spectrum.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();

assert_eq!(spectrum.splash().unwrap(), splash);
```

### Indices

```rust
use mass_spectrometry::prelude::*;

let mut a: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
a.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();

let mut b: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
b.add_peaks([(100.05, 10.0), (200.05, 20.0)]).unwrap();

let mut c: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
c.add_peaks([(300.0, 10.0), (400.0, 20.0)]).unwrap();

let spectra = vec![a, b, c];

let index = FlashCosineIndex::new(0.0, 1.0, 0.1, &spectra).unwrap();
let hits = index.search_threshold(&spectra[0], 0.8).unwrap();
assert!(hits.iter().any(|hit| hit.spectrum_id == 1));

let threshold_index =
    FlashCosineThresholdIndex::new(0.0, 1.0, 0.1, 0.8, &spectra).unwrap();
let mut state = threshold_index.new_search_state();
let mut edges = Vec::new();

for query_id in 0..threshold_index.n_spectra() {
    threshold_index
        .for_each_indexed_with_state(query_id, &mut state, |hit| {
            if hit.spectrum_id > query_id {
                edges.push((query_id, hit.spectrum_id, hit.score));
            }
        })
        .unwrap();
}

assert_eq!(edges.len(), 1);
assert_eq!(edges[0].0, 0);
assert_eq!(edges[0].1, 1);
assert!(edges[0].2 > 0.99);
```
