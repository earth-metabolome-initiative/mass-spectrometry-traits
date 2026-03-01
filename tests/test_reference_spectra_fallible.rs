use core::fmt;

use mass_spectrometry::prelude::{
    AdenineSpectrum, GenericSpectrum, Spectrum, SpectrumAlloc, SpectrumMut,
};

#[derive(Debug, Clone, Copy)]
struct AlwaysErr;

impl fmt::Display for AlwaysErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "always fail")
    }
}

impl std::error::Error for AlwaysErr {}

struct FailingSpectrum {
    inner: GenericSpectrum<f32, f32>,
}

impl Spectrum for FailingSpectrum {
    type Intensity = f32;
    type Mz = f32;
    type SortedIntensitiesIter<'a>
        = <GenericSpectrum<f32, f32> as Spectrum>::SortedIntensitiesIter<'a>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = <GenericSpectrum<f32, f32> as Spectrum>::SortedMzIter<'a>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = <GenericSpectrum<f32, f32> as Spectrum>::SortedPeaksIter<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.inner.intensities()
    }

    fn intensity_nth(&self, n: usize) -> Self::Intensity {
        self.inner.intensity_nth(n)
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.inner.mz()
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.inner.mz_from(index)
    }

    fn mz_nth(&self, n: usize) -> Self::Mz {
        self.inner.mz_nth(n)
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.inner.peaks()
    }

    fn peak_nth(&self, n: usize) -> (Self::Mz, Self::Intensity) {
        self.inner.peak_nth(n)
    }

    fn precursor_mz(&self) -> Self::Mz {
        self.inner.precursor_mz()
    }
}

impl SpectrumMut for FailingSpectrum {
    type MutationError = AlwaysErr;

    fn add_peak(
        &mut self,
        _mz: Self::Mz,
        _intensity: Self::Intensity,
    ) -> Result<(), Self::MutationError> {
        Err(AlwaysErr)
    }
}

impl SpectrumAlloc for FailingSpectrum {
    fn with_capacity(precursor_mz: Self::Mz, capacity: usize) -> Self {
        Self {
            inner: GenericSpectrum::with_capacity(precursor_mz, capacity),
        }
    }
}

#[test]
fn reference_constructor_returns_err_for_failing_spectrum_mutation() {
    let result = FailingSpectrum::adenine();
    assert!(
        result.is_err(),
        "expected failing constructor to return Err"
    );
}

#[test]
fn generic_spectrum_reference_constructor_returns_ok() {
    let spectrum = GenericSpectrum::<f32, f32>::adenine().expect("reference spectrum should build");
    assert!(!spectrum.is_empty());
}
