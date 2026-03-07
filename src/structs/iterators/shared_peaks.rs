//! Iterator over the shared peaks in two spectra, within a given tolerance.

use core::iter::Peekable;

use crate::prelude::Spectrum;

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
    RightSpectrum: Spectrum + 'a,
{
    left: Peekable<LeftSpectrum::SortedPeaksIter<'a>>,
    right: Peekable<RightSpectrum::SortedPeaksIter<'a>>,
    tolerance_f64: f64,
    right_shift_f64: f64,
}

impl<LeftSpectrum, RightSpectrum> Iterator for GreedySharedPeaks<'_, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum,
    RightSpectrum: Spectrum,
{
    type Item = ((f64, f64), (f64, f64));

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (Some(&(left_mz, _)), Some(&(right_mz, _))) = (self.left.peek(), self.right.peek())
            else {
                return None;
            };

            let shifted_right_mz = right_mz + self.right_shift_f64;

            if shifted_right_mz <= left_mz + self.tolerance_f64
                && left_mz <= shifted_right_mz + self.tolerance_f64
            {
                let (Some(left), Some(right)) = (self.left.next(), self.right.next()) else {
                    return None;
                };
                return Some((left, right));
            }

            if shifted_right_mz < left_mz {
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
    RightSpectrum: Spectrum + 'spectra,
{
    left: Option<&'spectra LeftSpectrum>,
    right: Option<&'spectra RightSpectrum>,
    tolerance: Option<f64>,
    right_shift: f64,
}

impl<LeftSpectrum, RightSpectrum> Default
    for GreedySharedPeaksBuilder<'_, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum,
    RightSpectrum: Spectrum,
{
    fn default() -> Self {
        Self {
            left: None,
            right: None,
            tolerance: None,
            right_shift: 0.0,
        }
    }
}

impl<'spectra, LeftSpectrum, RightSpectrum>
    GreedySharedPeaksBuilder<'spectra, LeftSpectrum, RightSpectrum>
where
    LeftSpectrum: Spectrum,
    RightSpectrum: Spectrum,
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
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Sets the shift for the right spectrum.
    pub fn right_shift(mut self, right_shift: f64) -> Self {
        self.right_shift = right_shift;
        self
    }

    /// Builds the [`GreedySharedPeaks`] iterator.
    pub fn build(
        self,
    ) -> Result<
        GreedySharedPeaks<'spectra, LeftSpectrum, RightSpectrum>,
        GreedySharedPeaksBuilderError,
    > {
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
        let tolerance_f64 =
            self.tolerance
                .ok_or(GreedySharedPeaksBuilderError::IncompleteBuild(
                    GreedySharedPeaksAttribute::Tolerance,
                ))?;
        if !tolerance_f64.is_finite() {
            return Err(GreedySharedPeaksBuilderError::NonFiniteValue("tolerance"));
        }
        if tolerance_f64 < 0.0 {
            return Err(GreedySharedPeaksBuilderError::NegativeTolerance);
        }
        let right_shift_f64 = self.right_shift;
        if !right_shift_f64.is_finite() {
            return Err(GreedySharedPeaksBuilderError::NonFiniteValue("right_shift"));
        }

        for (mz, _) in left.peaks() {
            if !mz.is_finite() {
                return Err(GreedySharedPeaksBuilderError::NonFiniteValue("left_mz"));
            }
        }

        for (mz, _) in right.peaks() {
            if !mz.is_finite() {
                return Err(GreedySharedPeaksBuilderError::NonFiniteValue("right_mz"));
            }
            let shifted = mz + right_shift_f64;
            if !shifted.is_finite() {
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
