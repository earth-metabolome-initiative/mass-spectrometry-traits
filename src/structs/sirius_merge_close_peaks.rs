//! Spectral processor that merges peaks closer than `2 * mz_tolerance`.
//!
//! Implements the SIRIUS greedy merge: process peaks from highest to lowest
//! intensity, absorbing all neighbors within the merge window and summing
//! their intensities while keeping the dominant peak's m/z.

use alloc::vec;
use alloc::vec::Vec;

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
/// If the running sum overflows to a non-finite value, it is clamped to
/// `f64::MAX`.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct SiriusMergeClosePeaks {
    mz_tolerance: f64,
}

impl SiriusMergeClosePeaks {
    /// Returns the m/z tolerance used for merging.
    #[inline]
    pub fn mz_tolerance(&self) -> f64 {
        self.mz_tolerance
    }

    /// Creates a new `SiriusMergeClosePeaks` processor.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `mz_tolerance` is negative or
    /// non-finite.
    #[inline]
    pub fn new(mz_tolerance: f64) -> Result<Self, SimilarityConfigError> {
        validate_non_negative_tolerance(mz_tolerance)?;
        Ok(Self { mz_tolerance })
    }
}

impl SpectralProcessor for SiriusMergeClosePeaks {
    type Spectrum = GenericSpectrum;

    fn process(&self, spectrum: &Self::Spectrum) -> Self::Spectrum {
        let n = spectrum.len();
        if n == 0 {
            return GenericSpectrum::try_with_capacity(spectrum.precursor_mz(), 0)
                .expect("precursor_mz from valid spectrum must be valid");
        }

        let merge_window = self.mz_tolerance + self.mz_tolerance;

        // Collect peaks into a working vec.
        let peaks: Vec<(f64, f64)> = spectrum.peaks().collect();

        // Build indices sorted by descending intensity.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            peaks[b]
                .1
                .partial_cmp(&peaks[a].1)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut consumed = vec![false; n];
        let mut survivors: Vec<(f64, f64)> = Vec::with_capacity(n);

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
                    summed_intensity = if merged.is_finite() { merged } else { f64::MAX };
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
                    summed_intensity = if merged.is_finite() { merged } else { f64::MAX };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_round_trips_tolerance() {
        let processor = SiriusMergeClosePeaks::new(0.25).expect("tolerance should be valid");
        assert_eq!(processor.mz_tolerance(), 0.25);
    }
}
