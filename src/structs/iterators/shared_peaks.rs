//! Iterator over the shared peaks in two spectra, within a given tolerance.

use core::iter::Peekable;

use crate::prelude::{Spectrum, SpectrumFloat};

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
    type Item = (
        (LeftSpectrum::Precision, LeftSpectrum::Precision),
        (RightSpectrum::Precision, RightSpectrum::Precision),
    );

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (Some(&(left_mz, _)), Some(&(right_mz, _))) = (self.left.peek(), self.right.peek())
            else {
                return None;
            };

            let left_mz_f64 = left_mz.to_f64();
            let shifted_right_mz = right_mz.to_f64() + self.right_shift_f64;

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
            let shifted = mz.to_f64() + right_shift_f64;
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

#[cfg(test)]
mod tests {
    use alloc::{string::ToString, vec, vec::Vec};

    use super::*;

    #[derive(Clone)]
    struct RawSpectrum {
        precursor_mz: f64,
        peaks: Vec<(f64, f64)>,
    }

    impl Spectrum for RawSpectrum {
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

    #[test]
    fn attribute_display_strings_are_stable() {
        assert_eq!(GreedySharedPeaksAttribute::Left.to_string(), "left");
        assert_eq!(GreedySharedPeaksAttribute::Right.to_string(), "right");
        assert_eq!(
            GreedySharedPeaksAttribute::Tolerance.to_string(),
            "tolerance"
        );
        assert_eq!(
            GreedySharedPeaksAttribute::RightShift.to_string(),
            "right_shift"
        );
    }

    #[test]
    fn builder_reports_missing_required_attributes() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(1.0, 1.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(1.0, 1.0)],
        };

        let missing_left = GreedySharedPeaksBuilder::<RawSpectrum, RawSpectrum>::default()
            .right(&right)
            .tolerance(0.1)
            .build();
        let missing_left = match missing_left {
            Ok(_) => panic!("missing left must fail"),
            Err(error) => error,
        };
        assert_eq!(
            missing_left,
            GreedySharedPeaksBuilderError::IncompleteBuild(GreedySharedPeaksAttribute::Left)
        );

        let missing_right = GreedySharedPeaksBuilder::<RawSpectrum, RawSpectrum>::default()
            .left(&left)
            .tolerance(0.1)
            .build();
        let missing_right = match missing_right {
            Ok(_) => panic!("missing right must fail"),
            Err(error) => error,
        };
        assert_eq!(
            missing_right,
            GreedySharedPeaksBuilderError::IncompleteBuild(GreedySharedPeaksAttribute::Right)
        );

        let missing_tolerance = GreedySharedPeaksBuilder::<RawSpectrum, RawSpectrum>::default()
            .left(&left)
            .right(&right)
            .build();
        let missing_tolerance = match missing_tolerance {
            Ok(_) => panic!("missing tolerance must fail"),
            Err(error) => error,
        };
        assert_eq!(
            missing_tolerance,
            GreedySharedPeaksBuilderError::IncompleteBuild(GreedySharedPeaksAttribute::Tolerance)
        );
        assert_eq!(
            missing_tolerance.to_string(),
            "Incomplete build: missing attribute `tolerance`"
        );
    }

    #[test]
    fn default_right_shift_of_zero_matches_direct_peaks() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(1.0, 1.0), (2.0, 1.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(1.05, 2.0), (2.05, 2.0)],
        };

        let pairs = GreedySharedPeaksBuilder::default()
            .left(&left)
            .right(&right)
            .tolerance(0.1)
            .build()
            .expect("builder should succeed")
            .collect::<Vec<_>>();

        assert_eq!(
            pairs,
            vec![((1.0, 1.0), (1.05, 2.0)), ((2.0, 1.0), (2.05, 2.0))]
        );
    }

    #[test]
    fn iterator_returns_none_after_advancing_past_non_matches() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(5.0, 1.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(1.0, 1.0), (2.0, 1.0)],
        };

        let mut iter = GreedySharedPeaksBuilder::default()
            .left(&left)
            .right(&right)
            .tolerance(0.1)
            .build()
            .expect("builder should succeed");

        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn empty_input_iterators_return_none_immediately() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![],
        };
        let right = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(1.0, 1.0)],
        };

        let mut iter = GreedySharedPeaksBuilder::default()
            .left(&left)
            .right(&right)
            .tolerance(0.1)
            .build()
            .expect("builder should succeed");

        assert!(iter.next().is_none());
    }

    #[test]
    fn builder_rejects_negative_tolerance() {
        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(1.0, 1.0)],
        };

        let error = GreedySharedPeaksBuilder::default()
            .left(&spectrum)
            .right(&spectrum)
            .tolerance(-0.1)
            .build();
        let error = match error {
            Ok(_) => panic!("negative tolerance should be rejected"),
            Err(error) => error,
        };
        assert_eq!(error, GreedySharedPeaksBuilderError::NegativeTolerance);
    }
}
