//! Regression tests for `GreedySharedPeaks`.

use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};
use mass_spectrometry::structs::iterators::shared_peaks::GreedySharedPeaksBuilder;

fn spectrum_from_peaks(precursor_mz: f32, peaks: &[(f32, f32)]) -> GenericSpectrum<f32, f32> {
    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz, peaks.len());
    for &(mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .expect("test peaks must be sorted by m/z");
    }
    spectrum
}

#[test]
fn shifted_progression_matches_all_pairs() {
    let left = spectrum_from_peaks(500.0, &[(100.0, 1.0), (140.0, 2.0)]);
    let right = spectrum_from_peaks(500.0, &[(150.0, 1.0), (200.0, 2.0)]);

    // After applying right_shift=-60:
    // right peaks become [90, 140], so only left[140] <-> right[200] should match.
    let matches = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(0.1)
        .right_shift(-60.0)
        .build()
        .expect("builder is complete")
        .collect::<Vec<_>>();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0.0, 140.0);
    assert_eq!(matches[0].1.0, 200.0);
}

#[test]
fn includes_exact_tolerance_boundary() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.1, 1.0)]);

    let matches = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(0.1)
        .right_shift(0.0)
        .build()
        .expect("builder is complete")
        .collect::<Vec<_>>();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0.0, 100.0);
    assert_eq!(matches[0].1.0, 100.1);
}

#[test]
fn long_nonmatching_sequence_completes() {
    let n = 200_000usize;
    let mut left = GenericSpectrum::with_capacity(1_000_000.0, n);
    let mut right = GenericSpectrum::with_capacity(2_000_000.0, n);

    for i in 0..n {
        left.add_peak(i as f32, 1.0).expect("sorted");
        right
            .add_peak(10_000_000.0 + i as f32, 1.0)
            .expect("sorted");
    }

    let count = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(0.0)
        .right_shift(0.0)
        .build()
        .expect("builder is complete")
        .count();

    assert_eq!(count, 0);
}

#[test]
fn unsigned_mz_shift_addition_does_not_overflow() {
    let mut left = GenericSpectrum::with_capacity(100_u32, 1);
    left.add_peak(10_u32, 1_u32).expect("sorted");

    let mut right = GenericSpectrum::with_capacity(100_u32, 1);
    right.add_peak(u32::MAX, 1_u32).expect("sorted");

    let result = std::panic::catch_unwind(|| {
        GreedySharedPeaksBuilder::default()
            .left(&left)
            .right(&right)
            .tolerance(1_u32)
            .right_shift(1_u32)
            .build()
            .expect("builder is complete")
            .count()
    });

    assert!(result.is_ok(), "shared peaks iteration panicked");
    assert_eq!(result.expect("catch_unwind succeeded"), 0);
}
