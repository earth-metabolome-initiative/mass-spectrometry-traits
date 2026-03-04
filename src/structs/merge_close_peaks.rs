//! Spectral processor that merges peaks closer than `2 * mz_tolerance`.
//!
//! Implements the SIRIUS greedy merge: process peaks from highest to lowest
//! intensity, absorbing all neighbors within the merge window and summing
//! their intensities while keeping the dominant peak's m/z.

use alloc::vec;
use alloc::vec::Vec;

use geometric_traits::prelude::{Finite, Number};
use num_traits::{Float, ToPrimitive};

use super::cosine_common::validate_non_negative_tolerance;
use super::similarity_errors::SimilarityConfigError;
use crate::structs::GenericSpectrum;
use crate::traits::{SpectralProcessor, Spectrum, SpectrumMut};

/// Merges peaks that are closer than `2 * mz_tolerance` in m/z.
///
/// After processing, consecutive peaks are guaranteed to be greater than
/// `2 * mz_tolerance` apart, satisfying the well-separated precondition
/// required by [`super::LinearCosine`].
///
/// The algorithm processes peaks from highest to lowest intensity. For each
/// surviving peak, all unconsumed neighbors within the merge window have their
/// intensities summed into the dominant peak, which keeps its original m/z.
/// If the running sum overflows to a non-finite value, it is clamped to the
/// maximum finite value for `MZ`.
pub struct MergeClosePeaks<MZ> {
    mz_tolerance: MZ,
}

impl<MZ: Number> MergeClosePeaks<MZ> {
    /// Returns the m/z tolerance used for merging.
    #[inline]
    pub fn mz_tolerance(&self) -> MZ {
        self.mz_tolerance
    }
}

impl<MZ> MergeClosePeaks<MZ>
where
    MZ: Number + ToPrimitive + PartialOrd,
{
    /// Creates a new `MergeClosePeaks` processor.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `mz_tolerance` is negative,
    /// non-finite, or not representable as `f64`.
    #[inline]
    pub fn new(mz_tolerance: MZ) -> Result<Self, SimilarityConfigError> {
        validate_non_negative_tolerance(mz_tolerance)?;
        Ok(Self { mz_tolerance })
    }
}

impl<MZ> SpectralProcessor for MergeClosePeaks<MZ>
where
    MZ: Float + Number + Finite + ToPrimitive,
{
    type Spectrum = GenericSpectrum<MZ, MZ>;

    fn process(&self, spectrum: &Self::Spectrum) -> Self::Spectrum {
        let n = spectrum.len();
        if n == 0 {
            return GenericSpectrum::try_with_capacity(spectrum.precursor_mz(), 0)
                .expect("precursor_mz from valid spectrum must be valid");
        }

        let merge_window = self.mz_tolerance + self.mz_tolerance;

        // Collect peaks into a working vec.
        let peaks: Vec<(MZ, MZ)> = spectrum.peaks().collect();

        // Build indices sorted by descending intensity.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            peaks[b]
                .1
                .partial_cmp(&peaks[a].1)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut consumed = vec![false; n];
        let mut survivors: Vec<(MZ, MZ)> = Vec::with_capacity(n);

        for &idx in &order {
            if consumed[idx] {
                continue;
            }
            consumed[idx] = true;
            let dominant_mz = peaks[idx].0;
            let mut summed_intensity = peaks[idx].1;

            // Scan left from idx.
            let mut j = idx;
            while j > 0 {
                j -= 1;
                if consumed[j] {
                    continue;
                }
                if dominant_mz - peaks[j].0 <= merge_window {
                    let merged = summed_intensity + peaks[j].1;
                    summed_intensity = if merged.is_finite() {
                        merged
                    } else {
                        <MZ as Float>::max_value()
                    };
                    consumed[j] = true;
                } else {
                    break;
                }
            }

            // Scan right from idx.
            for k in (idx + 1)..n {
                if consumed[k] {
                    continue;
                }
                if peaks[k].0 - dominant_mz <= merge_window {
                    let merged = summed_intensity + peaks[k].1;
                    summed_intensity = if merged.is_finite() {
                        merged
                    } else {
                        <MZ as Float>::max_value()
                    };
                    consumed[k] = true;
                } else {
                    break;
                }
            }

            survivors.push((dominant_mz, summed_intensity));
        }

        // Sort survivors by ascending m/z.
        survivors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let mut result =
            GenericSpectrum::try_with_capacity(spectrum.precursor_mz(), survivors.len())
                .expect("precursor_mz from valid spectrum must be valid");
        for (mz, intensity) in survivors {
            result
                .add_peak(mz, intensity)
                .expect("merged peaks should be valid and in sorted order");
        }
        result
    }
}
