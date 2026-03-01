//! Regression tests for unsigned `Mz` overflow safety in matching APIs.

use geometric_traits::prelude::*;
use mass_spectrometry::prelude::{GenericSpectrum, Spectrum, SpectrumAlloc, SpectrumMut};

fn one_peak_u32_spectrum(precursor_mz: u32, mz: u32) -> GenericSpectrum<u32, u32> {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor_mz, 1).expect("valid spectrum allocation");
    spectrum.add_peak(mz, 1).expect("sorted single peak");
    spectrum
}

#[test]
fn matching_peaks_u32_does_not_panic_on_tolerance_edge() {
    let left = one_peak_u32_spectrum(100, 10);
    let right = one_peak_u32_spectrum(100, 12);

    let result = std::panic::catch_unwind(|| left.matching_peaks(&right, 1));
    assert!(result.is_ok(), "matching_peaks panicked for unsigned mz");

    let graph = result
        .expect("catch_unwind succeeded")
        .expect("graph construction succeeded");
    assert_eq!(graph.number_of_rows(), 1);
    assert_eq!(graph.number_of_columns(), 1);
    assert_eq!(graph.number_of_defined_values(), 0);
}

#[test]
fn modified_matching_peaks_u32_does_not_panic_with_shift() {
    let left = one_peak_u32_spectrum(100, 10);
    let right = one_peak_u32_spectrum(100, u32::MAX);

    let result = std::panic::catch_unwind(|| left.modified_matching_peaks(&right, 1, 1));
    assert!(
        result.is_ok(),
        "modified_matching_peaks panicked for unsigned mz"
    );

    let graph = result
        .expect("catch_unwind succeeded")
        .expect("graph construction succeeded");
    assert_eq!(graph.number_of_rows(), 1);
    assert_eq!(graph.number_of_columns(), 1);
    assert_eq!(graph.number_of_defined_values(), 0);
}
