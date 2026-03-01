//! Iterator over the shared peaks in two spectra, within a given tolerance.

use core::iter::Peekable;

use num_traits::{ToPrimitive, Zero};

use crate::prelude::Spectrum;

#[inline]
fn to_f64_checked<T: ToPrimitive>(
    value: T,
    name: &'static str,
) -> Result<f64, GreedySharedPeaksBuilderError> {
    let value = value
        .to_f64()
        .ok_or(GreedySharedPeaksBuilderError::ValueNotRepresentable(name))?;
    if !value.is_finite() {
        return Err(GreedySharedPeaksBuilderError::NonFiniteValue(name));
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Attribute for the [`GreedySharedPeaks`] iterator.
pub enum GreedySharedPeaksAttribute {
    /// The left spectrum.
    Left,
    /// The right spectrum.
    Right,
    /// The tolerance.
    Tolerance,
    /// The right spectra shift.
    RightShift,
}

impl core::fmt::Display for GreedySharedPeaksAttribute {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Self::Left => write!(f, "left"),
            Self::Right => write!(f, "right"),
            Self::Tolerance => write!(f, "tolerance"),
            Self::RightShift => write!(f, "right_shift"),
        }
    }
}

/// Error type for building a [`GreedySharedPeaks`] iterator.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum GreedySharedPeaksBuilderError {
    /// The builder is missing a required attribute.
    #[error("Incomplete build: missing attribute `{0}`")]
    IncompleteBuild(GreedySharedPeaksAttribute),
    /// A value required for matching could not be represented as `f64`.
    #[error("value `{0}` must be representable as f64")]
    ValueNotRepresentable(&'static str),
    /// A value required for matching was not finite.
    #[error("value `{0}` must be finite")]
    NonFiniteValue(&'static str),
    /// Tolerance must be zero or positive.
    #[error("value `tolerance` must be >= 0")]
    NegativeTolerance,
}

/// Iterator over the shared peaks in two spectra, within a given tolerance.
pub struct GreedySharedPeaks<'a, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum + 'a,
    RightSpectrum: Spectrum<Mz = LeftSpectrum::Mz> + 'a,
{
    left: Peekable<LeftSpectrum::SortedPeaksIter<'a>>,
    right: Peekable<RightSpectrum::SortedPeaksIter<'a>>,
    tolerance_f64: f64,
    right_shift_f64: f64,
}

impl<LeftSpectrum, RightSpectrum> Iterator for GreedySharedPeaks<'_, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum,
    RightSpectrum: Spectrum<Mz = LeftSpectrum::Mz>,
    LeftSpectrum::Mz: ToPrimitive,
{
    type Item = (
        (LeftSpectrum::Mz, LeftSpectrum::Intensity),
        (RightSpectrum::Mz, RightSpectrum::Intensity),
    );

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (Some((left_mz, _)), Some((right_mz, _))) = (self.left.peek(), self.right.peek())
            else {
                return None;
            };

            let left_mz_f64 = left_mz.to_f64().expect("left_mz validated by builder");
            let right_mz_f64 = right_mz.to_f64().expect("right_mz validated by builder");
            let shifted_right_mz = right_mz_f64 + self.right_shift_f64;

            if shifted_right_mz <= left_mz_f64 + self.tolerance_f64
                && left_mz_f64 <= shifted_right_mz + self.tolerance_f64
            {
                let (Some(left), Some(right)) = (self.left.next(), self.right.next()) else {
                    return None;
                };
                return Some((left, right));
            }

            if shifted_right_mz < left_mz_f64 {
                self.right.next();
            } else {
                self.left.next();
            }
        }
    }
}

#[derive(Clone, Debug)]
/// Builder for the [`GreedySharedPeaks`] iterator.
pub struct GreedySharedPeaksBuilder<'spectra, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum + 'spectra,
    RightSpectrum: Spectrum<Mz = LeftSpectrum::Mz> + 'spectra,
{
    left: Option<&'spectra LeftSpectrum>,
    right: Option<&'spectra RightSpectrum>,
    tolerance: Option<LeftSpectrum::Mz>,
    right_shift: LeftSpectrum::Mz,
}

impl<LeftSpectrum, RightSpectrum> Default
    for GreedySharedPeaksBuilder<'_, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum,
    RightSpectrum: Spectrum<Mz = LeftSpectrum::Mz>,
{
    fn default() -> Self {
        Self {
            left: None,
            right: None,
            tolerance: None,
            right_shift: LeftSpectrum::Mz::zero(),
        }
    }
}

impl<'spectra, LeftSpectrum, RightSpectrum>
    GreedySharedPeaksBuilder<'spectra, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum,
    RightSpectrum: Spectrum<Mz = LeftSpectrum::Mz>,
{
    /// Sets the left spectrum.
    pub fn left(mut self, left: &'spectra LeftSpectrum) -> Self {
        self.left = Some(left);
        self
    }

    /// Sets the right spectrum.
    pub fn right(mut self, right: &'spectra RightSpectrum) -> Self {
        self.right = Some(right);
        self
    }

    /// Sets the tolerance.
    pub fn tolerance(mut self, tolerance: LeftSpectrum::Mz) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Sets the shift for the right spectrum.
    pub fn right_shift(mut self, right_shift: LeftSpectrum::Mz) -> Self {
        self.right_shift = right_shift;
        self
    }

    /// Builds the [`GreedySharedPeaks`] iterator.
    pub fn build(
        self,
    ) -> Result<
        GreedySharedPeaks<'spectra, LeftSpectrum, RightSpectrum>,
        GreedySharedPeaksBuilderError,
    >
    where
        LeftSpectrum::Mz: ToPrimitive,
    {
        let left = self
            .left
            .ok_or(GreedySharedPeaksBuilderError::IncompleteBuild(
                GreedySharedPeaksAttribute::Left,
            ))?;
        let right = self
            .right
            .ok_or(GreedySharedPeaksBuilderError::IncompleteBuild(
                GreedySharedPeaksAttribute::Right,
            ))?;
        let tolerance = self
            .tolerance
            .ok_or(GreedySharedPeaksBuilderError::IncompleteBuild(
                GreedySharedPeaksAttribute::Tolerance,
            ))?;
        let tolerance_f64 = to_f64_checked(tolerance, "tolerance")?;
        if tolerance_f64 < 0.0 {
            return Err(GreedySharedPeaksBuilderError::NegativeTolerance);
        }
        let right_shift_f64 = to_f64_checked(self.right_shift, "right_shift")?;

        for (mz, _) in left.peaks() {
            to_f64_checked(mz, "left_mz")?;
        }
        for (mz, _) in right.peaks() {
            let right_mz_f64 = to_f64_checked(mz, "right_mz")?;
            let shifted_right_mz = right_mz_f64 + right_shift_f64;
            if !shifted_right_mz.is_finite() {
                return Err(GreedySharedPeaksBuilderError::NonFiniteValue(
                    "shifted_right_mz",
                ));
            }
        }

        Ok(GreedySharedPeaks {
            left: left.peaks().peekable(),
            right: right.peaks().peekable(),
            tolerance_f64,
            right_shift_f64,
        })
    }
}
