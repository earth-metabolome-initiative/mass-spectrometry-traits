//! A naively implemented generic spectrum struct.

use alloc::vec::Vec;
use arbitrary::{Arbitrary as ByteArbitrary, Unstructured};

use geometric_traits::prelude::SortedVec;
#[cfg(feature = "proptest")]
use proptest::{
    arbitrary::Arbitrary as ProptestArbitrary,
    collection,
    strategy::{BoxedStrategy, Strategy},
};

use crate::numeric_validation::{ELECTRON_MASS, MAX_MZ};
use crate::traits::{Spectrum, SpectrumAlloc, SpectrumMut};

/// A generic spectrum struct.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, PartialEq)]
pub struct GenericSpectrum {
    mz: SortedVec<f64>,
    intensity: Vec<f64>,
    precursor_mz: f64,
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
    /// Peak m/z value is below the minimum physically meaningful value.
    #[error("mz must be >= ELECTRON_MASS (5.486e-4 Da)")]
    MzBelowMinimum,
    /// Peak m/z value exceeds the maximum allowed value.
    #[error("mz must be <= MAX_MZ (2,000,000 Da)")]
    MzAboveMaximum,
    /// Precursor m/z values must be finite.
    #[error("precursor_mz must be finite")]
    NonFinitePrecursorMz,
    /// Precursor m/z value is below the minimum physically meaningful value.
    #[error("precursor_mz must be >= ELECTRON_MASS (5.486e-4 Da)")]
    PrecursorMzBelowMinimum,
    /// Precursor m/z value exceeds the maximum allowed value.
    #[error("precursor_mz must be <= MAX_MZ (2,000,000 Da)")]
    PrecursorMzAboveMaximum,
    /// Intensities must be finite.
    #[error("intensity values must be finite")]
    NonFiniteIntensity,
    /// Intensities must be strictly positive.
    #[error("intensity must be > 0")]
    NonPositiveIntensity,
}

impl GenericSpectrum {
    /// Creates a new `GenericSpectrum` with a given capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if `precursor_mz` is non-finite or outside the valid
    /// range `[ELECTRON_MASS, MAX_MZ]`.
    pub fn try_with_capacity(
        precursor_mz: f64,
        capacity: usize,
    ) -> Result<Self, GenericSpectrumMutationError> {
        if !precursor_mz.is_finite() {
            return Err(GenericSpectrumMutationError::NonFinitePrecursorMz);
        }
        if precursor_mz < ELECTRON_MASS {
            return Err(GenericSpectrumMutationError::PrecursorMzBelowMinimum);
        }
        if precursor_mz > MAX_MZ {
            return Err(GenericSpectrumMutationError::PrecursorMzAboveMaximum);
        }
        Ok(Self {
            mz: SortedVec::with_capacity(capacity),
            intensity: Vec::with_capacity(capacity),
            precursor_mz,
        })
    }

    fn from_untrusted_parts(precursor_mz: f64, peaks: Vec<(f64, f64)>) -> Self {
        let precursor_mz =
            if precursor_mz.is_finite() && (ELECTRON_MASS..=MAX_MZ).contains(&precursor_mz) {
                precursor_mz
            } else {
                1.0
            };

        let mut sanitized: Vec<(f64, f64)> = peaks
            .into_iter()
            .filter_map(|(mz, intensity)| {
                if !mz.is_finite() {
                    return None;
                }
                if !(ELECTRON_MASS..=MAX_MZ).contains(&mz) {
                    return None;
                }
                if !intensity.is_finite() || intensity <= 0.0 {
                    return None;
                }
                Some((mz, intensity))
            })
            .collect();

        sanitized.sort_by(|(left_mz, _), (right_mz, _)| {
            left_mz
                .partial_cmp(right_mz)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut spectrum = GenericSpectrum {
            mz: SortedVec::with_capacity(sanitized.len()),
            intensity: Vec::with_capacity(sanitized.len()),
            precursor_mz,
        };

        for (mz, intensity) in sanitized {
            if spectrum.add_peak(mz, intensity).is_err() {
                continue;
            }
        }

        spectrum
    }
}

impl<'a> ByteArbitrary<'a> for GenericSpectrum {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let precursor_mz = <f64 as ByteArbitrary<'a>>::arbitrary(u)?;
        let peak_count = u.int_in_range(0..=64usize)?;
        let mut peaks = Vec::with_capacity(peak_count);
        for _ in 0..peak_count {
            peaks.push((
                <f64 as ByteArbitrary<'a>>::arbitrary(u)?,
                <f64 as ByteArbitrary<'a>>::arbitrary(u)?,
            ));
        }
        Ok(Self::from_untrusted_parts(precursor_mz, peaks))
    }
}

impl Spectrum for GenericSpectrum {
    type SortedIntensitiesIter<'a> = core::iter::Copied<core::slice::Iter<'a, f64>>;
    type SortedMzIter<'a> = core::iter::Copied<core::slice::Iter<'a, f64>>;
    type SortedPeaksIter<'a> =
        core::iter::Zip<Self::SortedMzIter<'a>, Self::SortedIntensitiesIter<'a>>;

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

    fn precursor_mz(&self) -> f64 {
        self.precursor_mz
    }

    fn intensity_nth(&self, n: usize) -> f64 {
        self.intensity[n]
    }

    fn mz_nth(&self, n: usize) -> f64 {
        self.mz[n]
    }

    fn peak_nth(&self, n: usize) -> (f64, f64) {
        (self.mz_nth(n), self.intensity_nth(n))
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.mz[index..].iter().copied()
    }
}

impl SpectrumMut for GenericSpectrum {
    type MutationError = GenericSpectrumMutationError;

    fn add_peak(&mut self, mz: f64, intensity: f64) -> Result<(), Self::MutationError> {
        if !mz.is_finite() {
            return Err(GenericSpectrumMutationError::NonFiniteMz);
        }
        if mz < ELECTRON_MASS {
            return Err(GenericSpectrumMutationError::MzBelowMinimum);
        }
        if mz > MAX_MZ {
            return Err(GenericSpectrumMutationError::MzAboveMaximum);
        }
        if !intensity.is_finite() {
            return Err(GenericSpectrumMutationError::NonFiniteIntensity);
        }
        if intensity <= 0.0 {
            return Err(GenericSpectrumMutationError::NonPositiveIntensity);
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

impl SpectrumAlloc for GenericSpectrum {
    fn with_capacity(precursor_mz: f64, capacity: usize) -> Result<Self, Self::MutationError> {
        Self::try_with_capacity(precursor_mz, capacity)
    }
}

#[cfg(feature = "proptest")]
impl ProptestArbitrary for GenericSpectrum {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        let precursor = <f64 as ProptestArbitrary>::arbitrary().prop_map(|value| {
            if value.is_finite() && value >= 0.0 {
                value
            } else {
                0.0
            }
        });
        let peaks = collection::vec(
            (
                <f64 as ProptestArbitrary>::arbitrary(),
                <f64 as ProptestArbitrary>::arbitrary(),
            ),
            0..64,
        );

        (precursor, peaks)
            .prop_map(|(precursor_mz, peaks)| {
                GenericSpectrum::from_untrusted_parts(precursor_mz, peaks)
            })
            .boxed()
    }
}
