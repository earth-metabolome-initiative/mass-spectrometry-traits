//! Submodule defining a Spectra collection trait.

use super::spectrum::Spectrum;

/// Trait for a collection of Spectra.
pub trait Spectra {
    /// The type of the Spectrum.
    type Spectrum: Spectrum;
    /// The type of the borrowed Spectrum iterator.
    type SpectraIter<'a>: Iterator<Item = &'a Self::Spectrum>
    where
        Self: 'a,
        Self::Spectrum: 'a;

    /// Returns an iterator over the Spectra.
    fn spectra(&self) -> Self::SpectraIter<'_>;

    /// Returns the number of Spectra.
    fn len(&self) -> usize;

    /// Returns true if the Spectra is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::traits::Spectrum;

    struct TestSpectrum {
        precursor_mz: f64,
        peaks: alloc::vec::Vec<(f64, f64)>,
    }

    impl Spectrum for TestSpectrum {
        type Precision = f64;

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

    struct TestSpectra(alloc::vec::Vec<TestSpectrum>);

    impl Spectra for TestSpectra {
        type Spectrum = TestSpectrum;
        type SpectraIter<'a>
            = core::slice::Iter<'a, TestSpectrum>
        where
            Self: 'a,
            Self::Spectrum: 'a;

        fn spectra(&self) -> Self::SpectraIter<'_> {
            self.0.iter()
        }

        fn len(&self) -> usize {
            self.0.len()
        }
    }

    #[test]
    fn is_empty_reflects_len_and_iteration_order() {
        let empty = TestSpectra(vec![]);
        assert!(empty.is_empty());

        let collection = TestSpectra(vec![
            TestSpectrum {
                precursor_mz: 100.0,
                peaks: vec![(10.0, 1.0)],
            },
            TestSpectrum {
                precursor_mz: 200.0,
                peaks: vec![(20.0, 2.0)],
            },
        ]);
        assert!(!collection.is_empty());
        assert_eq!(collection.len(), 2);

        let precursors: alloc::vec::Vec<f64> = collection
            .spectra()
            .map(|spectrum| spectrum.precursor_mz())
            .collect();
        assert_eq!(precursors.len(), 2);
        assert_eq!(precursors[0], collection.0[0].precursor_mz());
        assert_eq!(precursors[1], collection.0[1].precursor_mz());
    }
}
