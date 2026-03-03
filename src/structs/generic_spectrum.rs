//! A naively implemented generic spectrum struct.

use alloc::vec::Vec;

use geometric_traits::prelude::{Finite, Number, SortedVec};

use crate::traits::{Spectrum, SpectrumAlloc, SpectrumMut};

/// A generic spectrum struct.
pub struct GenericSpectrum<Mz, Intensity> {
    mz: SortedVec<Mz>,
    intensity: Vec<Intensity>,
    precursor_mz: Mz,
}

/// Error returned when mutating a [`GenericSpectrum`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GenericSpectrumMutationError {
    /// Peaks must be added in sorted m/z order.
    #[error("mz values must be added in sorted order")]
    UnsortedMz,
    /// Duplicate peak m/z values are not allowed.
    #[error("mz values must be strictly increasing; duplicate mz values are not allowed")]
    DuplicateMz,
    /// Peak m/z values must be finite.
    #[error("mz values must be finite")]
    NonFiniteMz,
    /// Peak m/z values must be zero or positive.
    #[error("mz values must be >= 0")]
    NegativeMz,
    /// Precursor m/z values must be finite.
    #[error("precursor_mz must be finite")]
    NonFinitePrecursorMz,
    /// Precursor m/z values must be zero or positive.
    #[error("precursor_mz must be >= 0")]
    NegativePrecursorMz,
    /// Intensities must be finite.
    #[error("intensity values must be finite")]
    NonFiniteIntensity,
    /// Intensities must be zero or positive.
    #[error("intensity values must be >= 0")]
    NegativeIntensity,
}

impl<Mz, Intensity> GenericSpectrum<Mz, Intensity>
where
    Mz: Number + PartialOrd + Finite,
    Intensity: Number + PartialOrd + Finite,
{
    /// Creates a new `GenericSpectrum` with a given capacity.
    ///
    /// # Errors
    ///
    /// Returns [`GenericSpectrumMutationError::NonFinitePrecursorMz`] if
    /// `precursor_mz` is not finite, and
    /// [`GenericSpectrumMutationError::NegativePrecursorMz`] if
    /// `precursor_mz` is negative.
    pub fn try_with_capacity(
        precursor_mz: Mz,
        capacity: usize,
    ) -> Result<Self, GenericSpectrumMutationError> {
        if !precursor_mz.is_finite() {
            return Err(GenericSpectrumMutationError::NonFinitePrecursorMz);
        }
        if precursor_mz < Mz::zero() {
            return Err(GenericSpectrumMutationError::NegativePrecursorMz);
        }
        Ok(Self {
            mz: SortedVec::with_capacity(capacity),
            intensity: Vec::with_capacity(capacity),
            precursor_mz,
        })
    }
}

impl<Mz, Intensity> Spectrum for GenericSpectrum<Mz, Intensity>
where
    Mz: Number,
    Intensity: Number,
{
    type Intensity = Intensity;
    type Mz = Mz;
    type SortedIntensitiesIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, Intensity>>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, Mz>>
    where
        Self: 'a;

    type SortedPeaksIter<'a>
        = core::iter::Zip<Self::SortedMzIter<'a>, Self::SortedIntensitiesIter<'a>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.mz.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.intensity.iter().copied()
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.mz.iter().copied()
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.mz().zip(self.intensities())
    }

    fn precursor_mz(&self) -> Self::Mz {
        self.precursor_mz
    }

    fn intensity_nth(&self, n: usize) -> Self::Intensity {
        self.intensity[n]
    }

    fn mz_nth(&self, n: usize) -> Self::Mz {
        self.mz[n]
    }

    fn peak_nth(&self, n: usize) -> (Self::Mz, Self::Intensity) {
        (self.mz_nth(n), self.intensity_nth(n))
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.mz[index..].iter().copied()
    }
}

impl<Mz, Intensity> SpectrumMut for GenericSpectrum<Mz, Intensity>
where
    Mz: Number + PartialOrd + Finite,
    Intensity: Number + PartialOrd + Finite,
{
    type MutationError = GenericSpectrumMutationError;

    fn add_peak(
        &mut self,
        mz: Self::Mz,
        intensity: Self::Intensity,
    ) -> Result<(), Self::MutationError> {
        if !mz.is_finite() {
            return Err(GenericSpectrumMutationError::NonFiniteMz);
        }
        if mz < Self::Mz::zero() {
            return Err(GenericSpectrumMutationError::NegativeMz);
        }
        if !intensity.is_finite() {
            return Err(GenericSpectrumMutationError::NonFiniteIntensity);
        }
        if intensity < Self::Intensity::zero() {
            return Err(GenericSpectrumMutationError::NegativeIntensity);
        }
        if let Some(last_mz) = self.mz.last() {
            if mz == *last_mz {
                return Err(GenericSpectrumMutationError::DuplicateMz);
            }
            if mz < *last_mz {
                return Err(GenericSpectrumMutationError::UnsortedMz);
            }
        }
        self.mz
            .push(mz)
            .map_err(|_| GenericSpectrumMutationError::UnsortedMz)?;
        self.intensity.push(intensity);
        Ok(())
    }
}

impl<Mz, Intensity> SpectrumAlloc for GenericSpectrum<Mz, Intensity>
where
    Mz: Number + PartialOrd + Finite,
    Intensity: Number + PartialOrd + Finite,
{
    fn with_capacity(precursor_mz: Self::Mz, capacity: usize) -> Result<Self, Self::MutationError> {
        Self::try_with_capacity(precursor_mz, capacity)
    }
}
