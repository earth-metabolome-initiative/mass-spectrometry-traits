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
use crate::traits::{Spectrum, SpectrumAlloc, SpectrumFloat, SpectrumMut};

/// A generic spectrum struct.
///
/// The storage precision defaults to `f64`. Use `GenericSpectrum<f32>` or
/// `GenericSpectrum<half::f16>` when lower memory use is more important than
/// stored peak precision.
///
/// # Example
///
/// ```
/// use half::f16;
/// use mass_spectrometry::prelude::*;
///
/// let mut f32_spectrum: GenericSpectrum<f32> =
///     GenericSpectrum::try_with_capacity(250.0, 1).unwrap();
/// f32_spectrum.add_peak(100.0, 2.0).unwrap();
/// assert_eq!(f32_spectrum.mz_nth(0), 100.0_f32);
///
/// let mut f16_spectrum: GenericSpectrum<f16> =
///     GenericSpectrum::try_with_capacity(250.0, 1).unwrap();
/// f16_spectrum.add_peak(100.0, 2.0).unwrap();
/// assert_eq!(f16_spectrum.intensity_nth(0), f16::from_f64(2.0));
/// ```
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, PartialEq)]
pub struct GenericSpectrum<P: SpectrumFloat = f64> {
    peaks: SortedVec<(P, P)>,
    precursor_mz: P,
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

impl<P: SpectrumFloat> GenericSpectrum<P> {
    /// Creates a new `GenericSpectrum` with a given capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if `precursor_mz` is non-finite or outside the valid
    /// range `[ELECTRON_MASS, MAX_MZ]`, or if it cannot be represented as a
    /// finite value in the selected precision.
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
        let precursor_mz =
            P::from_f64(precursor_mz).ok_or(GenericSpectrumMutationError::NonFinitePrecursorMz)?;
        let stored_precursor_mz = precursor_mz.to_f64();
        if stored_precursor_mz < ELECTRON_MASS {
            return Err(GenericSpectrumMutationError::PrecursorMzBelowMinimum);
        }
        if stored_precursor_mz > MAX_MZ {
            return Err(GenericSpectrumMutationError::PrecursorMzAboveMaximum);
        }
        Ok(Self {
            peaks: SortedVec::with_capacity(capacity),
            precursor_mz,
        })
    }

    fn from_untrusted_parts(precursor_mz: f64, peaks: Vec<(f64, f64)>) -> Self {
        let precursor_mz =
            if precursor_mz.is_finite() && (ELECTRON_MASS..=MAX_MZ).contains(&precursor_mz) {
                P::from_f64(precursor_mz).unwrap_or_else(|| {
                    P::from_f64(1.0).expect("1.0 must be representable as spectrum precision")
                })
            } else {
                P::from_f64(1.0).expect("1.0 must be representable as spectrum precision")
            };

        let mut sanitized: Vec<(P, P)> = peaks
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
                let mz = P::from_f64(mz)?;
                let intensity = P::from_f64(intensity)?;
                Some((mz, intensity))
            })
            .collect();

        sanitized.sort_by(|(left_mz, _), (right_mz, _)| {
            left_mz
                .partial_cmp(right_mz)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut spectrum = GenericSpectrum {
            peaks: SortedVec::with_capacity(sanitized.len()),
            precursor_mz,
        };

        for (mz, intensity) in sanitized {
            if spectrum.add_peak(mz.to_f64(), intensity.to_f64()).is_err() {
                continue;
            }
        }

        spectrum
    }
}

#[inline]
fn peak_mz<P: SpectrumFloat>(peak: &(P, P)) -> P {
    peak.0
}

#[inline]
fn peak_intensity<P: SpectrumFloat>(peak: &(P, P)) -> P {
    peak.1
}

impl<'a> ByteArbitrary<'a> for GenericSpectrum<f64> {
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

impl<P: SpectrumFloat> Spectrum for GenericSpectrum<P> {
    type Precision = P;
    type SortedIntensitiesIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (P, P)>, fn(&(P, P)) -> P>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (P, P)>, fn(&(P, P)) -> P>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, (P, P)>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.peaks.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.peaks
            .iter()
            .map(peak_intensity::<P> as fn(&(P, P)) -> P)
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.peaks.iter().map(peak_mz::<P> as fn(&(P, P)) -> P)
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.peaks.iter().copied()
    }

    fn precursor_mz(&self) -> Self::Precision {
        self.precursor_mz
    }

    fn intensity_nth(&self, n: usize) -> Self::Precision {
        self.peaks[n].1
    }

    fn mz_nth(&self, n: usize) -> Self::Precision {
        self.peaks[n].0
    }

    fn peak_nth(&self, n: usize) -> (Self::Precision, Self::Precision) {
        self.peaks[n]
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.peaks[index..]
            .iter()
            .map(peak_mz::<P> as fn(&(P, P)) -> P)
    }
}

impl<P: SpectrumFloat> SpectrumMut for GenericSpectrum<P> {
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
        let mz = P::from_f64(mz).ok_or(GenericSpectrumMutationError::NonFiniteMz)?;
        let intensity =
            P::from_f64(intensity).ok_or(GenericSpectrumMutationError::NonFiniteIntensity)?;
        let stored_mz = mz.to_f64();
        if stored_mz < ELECTRON_MASS {
            return Err(GenericSpectrumMutationError::MzBelowMinimum);
        }
        if stored_mz > MAX_MZ {
            return Err(GenericSpectrumMutationError::MzAboveMaximum);
        }
        if intensity.to_f64() <= 0.0 {
            return Err(GenericSpectrumMutationError::NonPositiveIntensity);
        }
        // Tuple ordering would allow equal m/z with increasing intensity, so
        // keep the stricter spectrum invariant explicit.
        if let Some(&(last_mz, _)) = self.peaks.last() {
            if mz == last_mz {
                return Err(GenericSpectrumMutationError::DuplicateMz);
            }
            if mz < last_mz {
                return Err(GenericSpectrumMutationError::UnsortedMz);
            }
        }
        self.peaks
            .push((mz, intensity))
            .map_err(|_| GenericSpectrumMutationError::UnsortedMz)?;
        Ok(())
    }
}

impl<P: SpectrumFloat> SpectrumAlloc for GenericSpectrum<P> {
    fn with_capacity(precursor_mz: f64, capacity: usize) -> Result<Self, Self::MutationError> {
        Self::try_with_capacity(precursor_mz, capacity)
    }
}

#[cfg(feature = "proptest")]
impl ProptestArbitrary for GenericSpectrum<f64> {
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
