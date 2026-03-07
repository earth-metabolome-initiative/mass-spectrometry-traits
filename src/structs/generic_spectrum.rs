//! A naively implemented generic spectrum struct.

use alloc::vec::Vec;
use arbitrary::{Arbitrary as ByteArbitrary, Unstructured};

use geometric_traits::prelude::{Finite, Number, SortedVec};
use num_traits::ToPrimitive;
#[cfg(feature = "proptest")]
use proptest::{
    arbitrary::Arbitrary as ProptestArbitrary,
    collection,
    strategy::{BoxedStrategy, Strategy},
};

use crate::numeric_validation::{ELECTRON_MASS, MAX_MZ};
use crate::traits::{Spectrum, SpectrumAlloc, SpectrumMut};

/// A generic spectrum struct.
#[derive(Debug)]
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
    /// Peak m/z value is below the minimum physically meaningful value.
    #[error("mz must be >= ELECTRON_MASS (5.486e-4 Da)")]
    MzBelowMinimum,
    /// Peak m/z value exceeds the maximum allowed value.
    #[error("mz must be <= MAX_MZ (2,000,000 Da)")]
    MzAboveMaximum,
    /// Peak m/z value cannot be represented as f64.
    #[error("mz value not representable as f64")]
    MzNotRepresentable,
    /// Precursor m/z values must be finite.
    #[error("precursor_mz must be finite")]
    NonFinitePrecursorMz,
    /// Precursor m/z value is below the minimum physically meaningful value.
    #[error("precursor_mz must be >= ELECTRON_MASS (5.486e-4 Da)")]
    PrecursorMzBelowMinimum,
    /// Precursor m/z value exceeds the maximum allowed value.
    #[error("precursor_mz must be <= MAX_MZ (2,000,000 Da)")]
    PrecursorMzAboveMaximum,
    /// Precursor m/z value cannot be represented as f64.
    #[error("precursor_mz not representable as f64")]
    PrecursorMzNotRepresentable,
    /// Intensities must be finite.
    #[error("intensity values must be finite")]
    NonFiniteIntensity,
    /// Intensities must be strictly positive.
    #[error("intensity must be > 0")]
    NonPositiveIntensity,
}

impl<Mz, Intensity> GenericSpectrum<Mz, Intensity>
where
    Mz: Number + PartialOrd + Finite + ToPrimitive,
    Intensity: Number + PartialOrd + Finite,
{
    /// Creates a new `GenericSpectrum` with a given capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if `precursor_mz` is non-finite, not representable
    /// as f64, or outside the valid range `[ELECTRON_MASS, MAX_MZ]`.
    pub fn try_with_capacity(
        precursor_mz: Mz,
        capacity: usize,
    ) -> Result<Self, GenericSpectrumMutationError> {
        if !precursor_mz.is_finite() {
            return Err(GenericSpectrumMutationError::NonFinitePrecursorMz);
        }
        let precursor_f64 = precursor_mz
            .to_f64()
            .ok_or(GenericSpectrumMutationError::PrecursorMzNotRepresentable)?;
        if precursor_f64 < ELECTRON_MASS {
            return Err(GenericSpectrumMutationError::PrecursorMzBelowMinimum);
        }
        if precursor_f64 > MAX_MZ {
            return Err(GenericSpectrumMutationError::PrecursorMzAboveMaximum);
        }
        Ok(Self {
            mz: SortedVec::with_capacity(capacity),
            intensity: Vec::with_capacity(capacity),
            precursor_mz,
        })
    }
}

impl<Mz, Intensity> GenericSpectrum<Mz, Intensity>
where
    Mz: Number + PartialOrd + Finite + ToPrimitive,
    Intensity: Number + PartialOrd + Finite,
{
    fn from_untrusted_parts(precursor_mz: Mz, peaks: Vec<(Mz, Intensity)>) -> Self {
        let precursor_mz = if precursor_mz.is_finite() {
            let f = precursor_mz.to_f64().unwrap_or(0.0);
            if (ELECTRON_MASS..=MAX_MZ).contains(&f) {
                precursor_mz
            } else {
                Mz::one()
            }
        } else {
            Mz::one()
        };

        let mut sanitized: Vec<(Mz, Intensity)> = peaks
            .into_iter()
            .filter_map(|(mz, intensity)| {
                if !mz.is_finite() {
                    return None;
                }
                let mz_f64 = mz.to_f64()?;
                if !(ELECTRON_MASS..=MAX_MZ).contains(&mz_f64) {
                    return None;
                }
                if !intensity.is_finite() || intensity <= Intensity::zero() {
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

impl<'a, Mz, Intensity> ByteArbitrary<'a> for GenericSpectrum<Mz, Intensity>
where
    Mz: ByteArbitrary<'a> + Number + PartialOrd + Finite + ToPrimitive,
    Intensity: ByteArbitrary<'a> + Number + PartialOrd + Finite,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let precursor_mz = Mz::arbitrary(u)?;
        let peak_count = u.int_in_range(0..=64usize)?;
        let mut peaks = Vec::with_capacity(peak_count);
        for _ in 0..peak_count {
            peaks.push((Mz::arbitrary(u)?, Intensity::arbitrary(u)?));
        }
        Ok(Self::from_untrusted_parts(precursor_mz, peaks))
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
    Mz: Number + PartialOrd + Finite + ToPrimitive,
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
        let mz_f64 = mz
            .to_f64()
            .ok_or(GenericSpectrumMutationError::MzNotRepresentable)?;
        if mz_f64 < ELECTRON_MASS {
            return Err(GenericSpectrumMutationError::MzBelowMinimum);
        }
        if mz_f64 > MAX_MZ {
            return Err(GenericSpectrumMutationError::MzAboveMaximum);
        }
        if !intensity.is_finite() {
            return Err(GenericSpectrumMutationError::NonFiniteIntensity);
        }
        if intensity <= Self::Intensity::zero() {
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

impl<Mz, Intensity> SpectrumAlloc for GenericSpectrum<Mz, Intensity>
where
    Mz: Number + PartialOrd + Finite + ToPrimitive,
    Intensity: Number + PartialOrd + Finite,
{
    fn with_capacity(precursor_mz: Self::Mz, capacity: usize) -> Result<Self, Self::MutationError> {
        Self::try_with_capacity(precursor_mz, capacity)
    }
}

#[cfg(feature = "proptest")]
impl<Mz, Intensity> ProptestArbitrary for GenericSpectrum<Mz, Intensity>
where
    Mz: ProptestArbitrary + Number + PartialOrd + Finite + ToPrimitive + core::fmt::Debug + 'static,
    Intensity: ProptestArbitrary + Number + PartialOrd + Finite + core::fmt::Debug + 'static,
    <Mz as ProptestArbitrary>::Parameters: Default,
    <Intensity as ProptestArbitrary>::Parameters: Default,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        let precursor = Mz::arbitrary().prop_map(|value| {
            if value.is_finite() && value >= Mz::zero() {
                value
            } else {
                Mz::zero()
            }
        });
        let peaks = collection::vec((Mz::arbitrary(), Intensity::arbitrary()), 0..64);

        (precursor, peaks)
            .prop_map(|(precursor_mz, peaks)| {
                GenericSpectrum::from_untrusted_parts(precursor_mz, peaks)
            })
            .boxed()
    }
}
