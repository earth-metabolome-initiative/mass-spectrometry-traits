//! Regression tests for matching peak graph edge-case safety.

use geometric_traits::prelude::*;
use mass_spectrometry::prelude::{GenericSpectrum, Spectrum, SpectrumAlloc, SpectrumMut};

fn one_peak_spectrum(precursor_mz: f64, mz: f64) -> GenericSpectrum {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor_mz, 1).expect("valid spectrum allocation");
    spectrum.add_peak(mz, 1.0).expect("sorted single peak");
    spectrum
}

#[test]
fn matching_peaks_does_not_panic_on_tolerance_edge() {
    let left = one_peak_spectrum(100.0, 10.0);
    let right = one_peak_spectrum(100.0, 12.0);

    let result = std::panic::catch_unwind(|| left.matching_peaks(&right, 1.0));
    assert!(result.is_ok(), "matching_peaks panicked");

    let graph = result
        .expect("catch_unwind succeeded")
        .expect("graph construction succeeded");
    assert_eq!(graph.number_of_rows(), 1);
    assert_eq!(graph.number_of_columns(), 1);
    assert_eq!(graph.number_of_defined_values(), 0);
}

#[test]
fn modified_matching_peaks_does_not_panic_with_shift() {
    let left = one_peak_spectrum(100.0, 10.0);
    let right = one_peak_spectrum(100.0, 2_000_000.0);

    let result =
        std::panic::catch_unwind(|| left.modified_matching_peaks(&right, 1.0, 100.0, 100.0));
    assert!(result.is_ok(), "modified_matching_peaks panicked");

    let graph = result
        .expect("catch_unwind succeeded")
        .expect("graph construction succeeded");
    assert_eq!(graph.number_of_rows(), 1);
    assert_eq!(graph.number_of_columns(), 1);
    assert_eq!(graph.number_of_defined_values(), 0);
}
