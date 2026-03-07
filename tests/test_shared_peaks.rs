//! Regression tests for `GreedySharedPeaks`.

use mass_spectrometry::prelude::{GenericSpectrum, Spectrum, SpectrumAlloc, SpectrumMut};
use mass_spectrometry::structs::iterators::shared_peaks::{
    GreedySharedPeaksBuilder, GreedySharedPeaksBuilderError,
};

#[derive(Clone)]
struct RawSpectrum {
    precursor_mz: f64,
    peaks: Vec<(f64, f64)>,
}

impl Spectrum for RawSpectrum {
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

fn spectrum_from_peaks(precursor_mz: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz, peaks.len())
        .expect("valid spectrum allocation");
    for &(mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .expect("test peaks must be sorted by m/z");
    }
    spectrum
}

fn raw_spectrum_from_peaks(precursor_mz: f64, peaks: &[(f64, f64)]) -> RawSpectrum {
    RawSpectrum {
        precursor_mz,
        peaks: peaks.to_vec(),
    }
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
    let mut left =
        GenericSpectrum::with_capacity(1_000_000.0, n).expect("valid spectrum allocation");
    let mut right =
        GenericSpectrum::with_capacity(2_000_000.0, n).expect("valid spectrum allocation");

    for i in 0..n {
        left.add_peak((i + 1) as f64, 1.0).expect("sorted");
        right
            .add_peak(1_000_000.0 + (i + 1) as f64, 1.0)
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
fn rejects_nan_tolerance() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(f64::NAN)
        .right_shift(0.0)
        .build();
    let error = match error {
        Ok(_) => panic!("NaN tolerance should be rejected"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        GreedySharedPeaksBuilderError::NonFiniteValue("tolerance")
    );
}

#[test]
fn rejects_non_finite_right_shift() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(0.1)
        .right_shift(f64::INFINITY)
        .build();
    let error = match error {
        Ok(_) => panic!("non-finite shift should be rejected"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        GreedySharedPeaksBuilderError::NonFiniteValue("right_shift")
    );
}

#[test]
fn rejects_non_finite_left_peak_mz() {
    let left = raw_spectrum_from_peaks(100.0, &[(f64::INFINITY, 1.0)]);
    let right = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);

    let error = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(0.1)
        .right_shift(0.0)
        .build();
    let error = match error {
        Ok(_) => panic!("non-finite left mz should be rejected"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        GreedySharedPeaksBuilderError::NonFiniteValue("left_mz")
    );
}

#[test]
fn rejects_non_finite_right_peak_mz() {
    let left = spectrum_from_peaks(100.0, &[(100.0, 1.0)]);
    let right = raw_spectrum_from_peaks(100.0, &[(f64::INFINITY, 1.0)]);

    let error = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(0.1)
        .right_shift(0.0)
        .build();
    let error = match error {
        Ok(_) => panic!("non-finite right mz should be rejected"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        GreedySharedPeaksBuilderError::NonFiniteValue("right_mz")
    );
}

#[test]
fn rejects_shifted_right_mz_overflow_to_infinity() {
    // Use RawSpectrum to bypass GenericSpectrum validation — f64::MAX
    // exceeds MAX_MZ but is needed to trigger overflow when shifted.
    let left = RawSpectrum {
        precursor_mz: 1.0,
        peaks: vec![(1.0, 1.0)],
    };
    let right = RawSpectrum {
        precursor_mz: 1.0,
        peaks: vec![(f64::MAX, 1.0)],
    };

    let error = GreedySharedPeaksBuilder::default()
        .left(&left)
        .right(&right)
        .tolerance(0.1)
        .right_shift(f64::MAX)
        .build();
    let error = match error {
        Ok(_) => panic!("shifted right mz overflow should be rejected"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        GreedySharedPeaksBuilderError::NonFiniteValue("shifted_right_mz")
    );
}
