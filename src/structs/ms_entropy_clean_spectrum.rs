//! Spectral processor that mirrors `ms_entropy.clean_spectrum` semantics.
//!
//! The default configuration matches `ms_entropy` defaults:
//! - `min_mz`: disabled
//! - `max_mz`: disabled
//! - `noise_threshold`: `0.01`
//! - `min_ms2_difference_in_da`: `0.05`
//! - `min_ms2_difference_in_ppm`: disabled
//! - `max_peak_num`: disabled
//! - `normalize_intensity`: `true`

use alloc::vec::Vec;

use super::cosine_common::validate_numeric_parameter;
use super::similarity_errors::SimilarityConfigError;
use crate::structs::GenericSpectrum;
use crate::traits::{SpectralProcessor, Spectrum, SpectrumMut};

/// Spectral processor mirroring `ms_entropy.clean_spectrum` behavior.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct MsEntropyCleanSpectrum {
    min_mz: Option<f64>,
    max_mz: Option<f64>,
    noise_threshold: Option<f64>,
    min_ms2_difference_in_da: f64,
    min_ms2_difference_in_ppm: Option<f64>,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
}

impl MsEntropyCleanSpectrum {
    /// Returns a builder configured with `ms_entropy` defaults.
    #[inline]
    pub fn builder() -> MsEntropyCleanSpectrumBuilder {
        MsEntropyCleanSpectrumBuilder::default()
    }

    /// Returns the configured minimum mz filter (enabled only when > 0).
    #[inline]
    pub fn min_mz(&self) -> Option<f64> {
        self.min_mz
    }

    /// Returns the configured maximum mz filter (enabled only when > 0).
    #[inline]
    pub fn max_mz(&self) -> Option<f64> {
        self.max_mz
    }

    /// Returns the configured relative noise threshold.
    #[inline]
    pub fn noise_threshold(&self) -> Option<f64> {
        self.noise_threshold
    }

    /// Returns the configured Da centroid threshold.
    #[inline]
    pub fn min_ms2_difference_in_da(&self) -> f64 {
        self.min_ms2_difference_in_da
    }

    /// Returns the configured ppm centroid threshold.
    #[inline]
    pub fn min_ms2_difference_in_ppm(&self) -> Option<f64> {
        self.min_ms2_difference_in_ppm
    }

    /// Returns the configured maximum retained peak count.
    #[inline]
    pub fn max_peak_num(&self) -> Option<usize> {
        self.max_peak_num
    }

    /// Returns whether output intensities are normalized to sum to 1.
    #[inline]
    pub fn normalize_intensity(&self) -> bool {
        self.normalize_intensity
    }

    fn clean_peaks(&self, mut peaks: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        // Step 1. Remove empty peaks.
        peaks.retain(|(mz, intensity)| *mz > 0.0 && *intensity > 0.0);

        // Step 2. Min/max mz filtering.
        if let Some(min_mz) = self.min_mz
            && min_mz > 0.0
        {
            peaks.retain(|(mz, _)| *mz >= min_mz);
        }
        if let Some(max_mz) = self.max_mz
            && max_mz > 0.0
        {
            peaks.retain(|(mz, _)| *mz <= max_mz);
        }

        if peaks.is_empty() {
            return peaks;
        }

        // Step 3. Centroiding.
        let ppm = self.min_ms2_difference_in_ppm.filter(|&v| v > 0.0);
        peaks = centroid_spectrum(peaks, self.min_ms2_difference_in_da, ppm);

        if peaks.is_empty() {
            return peaks;
        }

        // Step 4. Noise filtering.
        if let Some(noise_threshold) = self.noise_threshold {
            let max_intensity = peaks
                .iter()
                .map(|&(_, intensity)| intensity)
                .fold(0.0_f64, f64::max);
            let threshold = noise_threshold * max_intensity;
            peaks.retain(|&(_, intensity)| intensity >= threshold);
        }

        if peaks.is_empty() {
            return peaks;
        }

        // Step 5. Keep top-N peaks.
        if let Some(max_peak_num) = self.max_peak_num
            && max_peak_num > 0
            && peaks.len() > max_peak_num
        {
            peaks.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).expect("non-NaN intensity"));
            peaks = peaks[peaks.len() - max_peak_num..].to_vec();
            peaks.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN mz"));
        }

        // Step 6. Normalize to sum 1.
        if self.normalize_intensity {
            let spectrum_sum: f64 = peaks.iter().map(|&(_, intensity)| intensity).sum();
            if spectrum_sum > 0.0 {
                for (_, intensity) in &mut peaks {
                    *intensity /= spectrum_sum;
                }
                // Division can produce zero/subnormal for tiny intensities
                // relative to a large sum.  Drop any non-positive results.
                peaks.retain(|&(_, intensity)| intensity > 0.0);
            } else {
                peaks.clear();
            }
        }

        peaks
    }
}

impl SpectralProcessor for MsEntropyCleanSpectrum {
    type Spectrum = GenericSpectrum;

    fn process(&self, spectrum: &Self::Spectrum) -> Self::Spectrum {
        let input_peaks: Vec<(f64, f64)> = spectrum.peaks().collect();

        let cleaned = self.clean_peaks(input_peaks);

        let mut result = GenericSpectrum::try_with_capacity(spectrum.precursor_mz(), cleaned.len())
            .expect("precursor_mz from valid spectrum must be valid");

        for (mz, intensity) in cleaned {
            result
                .add_peak(mz, intensity)
                .expect("cleaned peaks should be valid and sorted by m/z");
        }

        result
    }
}

/// Builder for [`MsEntropyCleanSpectrum`].
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct MsEntropyCleanSpectrumBuilder {
    min_mz: Option<f64>,
    max_mz: Option<f64>,
    noise_threshold: Option<f64>,
    min_ms2_difference_in_da: f64,
    min_ms2_difference_in_ppm: Option<f64>,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
}

impl Default for MsEntropyCleanSpectrumBuilder {
    fn default() -> Self {
        Self {
            min_mz: None,
            max_mz: None,
            noise_threshold: Some(0.01),
            min_ms2_difference_in_da: 0.05,
            min_ms2_difference_in_ppm: None,
            max_peak_num: None,
            normalize_intensity: true,
        }
    }
}

impl MsEntropyCleanSpectrumBuilder {
    /// Sets the minimum m/z filter.
    #[inline]
    pub fn min_mz(mut self, min_mz: Option<f64>) -> Result<Self, SimilarityConfigError> {
        if let Some(v) = min_mz {
            validate_numeric_parameter(v, "min_mz")?;
        }
        self.min_mz = min_mz;
        Ok(self)
    }

    /// Sets the maximum m/z filter.
    #[inline]
    pub fn max_mz(mut self, max_mz: Option<f64>) -> Result<Self, SimilarityConfigError> {
        if let Some(v) = max_mz {
            validate_numeric_parameter(v, "max_mz")?;
        }
        self.max_mz = max_mz;
        Ok(self)
    }

    /// Sets the relative noise threshold.
    #[inline]
    pub fn noise_threshold(
        mut self,
        noise_threshold: Option<f64>,
    ) -> Result<Self, SimilarityConfigError> {
        if let Some(v) = noise_threshold {
            validate_numeric_parameter(v, "noise_threshold")?;
        }
        self.noise_threshold = noise_threshold;
        Ok(self)
    }

    /// Sets the minimum Da centroid distance.
    #[inline]
    pub fn min_ms2_difference_in_da(
        mut self,
        min_ms2_difference_in_da: f64,
    ) -> Result<Self, SimilarityConfigError> {
        validate_numeric_parameter(min_ms2_difference_in_da, "min_ms2_difference_in_da")?;
        self.min_ms2_difference_in_da = min_ms2_difference_in_da;
        Ok(self)
    }

    /// Sets the minimum ppm centroid distance.
    #[inline]
    pub fn min_ms2_difference_in_ppm(
        mut self,
        min_ms2_difference_in_ppm: Option<f64>,
    ) -> Result<Self, SimilarityConfigError> {
        if let Some(v) = min_ms2_difference_in_ppm {
            validate_numeric_parameter(v, "min_ms2_difference_in_ppm")?;
        }
        self.min_ms2_difference_in_ppm = min_ms2_difference_in_ppm;
        Ok(self)
    }

    /// Sets the maximum number of retained peaks.
    #[inline]
    pub fn max_peak_num(
        mut self,
        max_peak_num: Option<usize>,
    ) -> Result<Self, SimilarityConfigError> {
        if matches!(max_peak_num, Some(0)) {
            return Err(SimilarityConfigError::InvalidParameter("max_peak_num"));
        }
        self.max_peak_num = max_peak_num;
        Ok(self)
    }

    /// Enables or disables output intensity normalization.
    #[inline]
    pub fn normalize_intensity(
        mut self,
        normalize_intensity: bool,
    ) -> Result<Self, SimilarityConfigError> {
        self.normalize_intensity = normalize_intensity;
        Ok(self)
    }

    /// Builds the cleaner from validated builder fields.
    pub fn build(self) -> Result<MsEntropyCleanSpectrum, SimilarityConfigError> {
        let ppm_positive = self.min_ms2_difference_in_ppm.is_some_and(|ppm| ppm > 0.0);
        let da_positive = self.min_ms2_difference_in_da > 0.0;
        if !ppm_positive && !da_positive {
            return Err(SimilarityConfigError::InvalidParameter(
                "min_ms2_difference_in_da/min_ms2_difference_in_ppm",
            ));
        }

        Ok(MsEntropyCleanSpectrum {
            min_mz: self.min_mz,
            max_mz: self.max_mz,
            noise_threshold: self.noise_threshold,
            min_ms2_difference_in_da: self.min_ms2_difference_in_da,
            min_ms2_difference_in_ppm: self.min_ms2_difference_in_ppm,
            max_peak_num: self.max_peak_num,
            normalize_intensity: self.normalize_intensity,
        })
    }
}

#[inline]
fn check_centroid(peaks: &[(f64, f64)], ms2_da: f64, ms2_ppm: Option<f64>) -> bool {
    if peaks.len() <= 1 {
        return true;
    }
    if let Some(ppm) = ms2_ppm {
        peaks
            .windows(2)
            .all(|w| (w[1].0 - w[0].0) > (w[1].0 * ppm * 1e-6))
    } else {
        peaks.windows(2).all(|w| (w[1].0 - w[0].0) > ms2_da)
    }
}

fn centroid_spectrum(
    mut peaks: Vec<(f64, f64)>,
    ms2_da: f64,
    ms2_ppm: Option<f64>,
) -> Vec<(f64, f64)> {
    peaks.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN mz"));
    while !check_centroid(&peaks, ms2_da, ms2_ppm) {
        peaks = centroid_once(peaks, ms2_da, ms2_ppm);
    }
    peaks
}

fn centroid_once(mut peaks: Vec<(f64, f64)>, ms2_da: f64, ms2_ppm: Option<f64>) -> Vec<(f64, f64)> {
    let n = peaks.len();
    let mut intensity_order: Vec<usize> = (0..n).collect();
    intensity_order.sort_unstable_by(|&a, &b| {
        peaks[b]
            .1
            .partial_cmp(&peaks[a].1)
            .expect("non-NaN intensity")
    });

    for &idx in &intensity_order {
        if peaks[idx].1 <= 0.0 {
            continue;
        }

        let mz = peaks[idx].0;
        let (delta_left, delta_right) = if let Some(ppm) = ms2_ppm {
            let left = mz * ppm * 1e-6;
            let right = mz * ppm / (1e6 - ppm);
            (left, right)
        } else {
            (ms2_da, ms2_da)
        };

        let mut left_idx = idx;
        while left_idx > 0 && (peaks[idx].0 - peaks[left_idx - 1].0) <= delta_left {
            left_idx -= 1;
        }

        let mut right_idx = idx + 1;
        while right_idx < n && (peaks[right_idx].0 - peaks[idx].0) <= delta_right {
            right_idx += 1;
        }

        // Skip the weighted-average recomputation when no live neighbors
        // exist. Running `(int * mz) / int` on a lone peak with subnormal
        // intensity can lose enough precision to push mz below the valid
        // minimum.
        let has_live_neighbor = (left_idx..right_idx).any(|j| j != idx && peaks[j].1 > 0.0);
        if !has_live_neighbor {
            continue;
        }

        let mut intensity_sum = 0.0_f64;
        let mut intensity_weighted_mz_sum = 0.0_f64;
        for peak in peaks.iter_mut().take(right_idx).skip(left_idx) {
            intensity_sum += peak.1;
            intensity_weighted_mz_sum += peak.1 * peak.0;
            peak.1 = 0.0;
        }

        if intensity_sum > 0.0 {
            peaks[idx].0 = intensity_weighted_mz_sum / intensity_sum;
        } else {
            peaks[idx].0 = 0.0;
        }
        peaks[idx].1 = intensity_sum;
    }

    peaks.retain(|&(mz, intensity)| mz.is_finite() && mz > 0.0 && intensity > 0.0);
    peaks.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN mz"));
    peaks
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn default_cleaner() -> MsEntropyCleanSpectrum {
        MsEntropyCleanSpectrum::builder()
            .build()
            .expect("default cleaner should build")
    }

    #[test]
    fn clean_peaks_returns_early_after_mz_filtering() {
        let cleaner = MsEntropyCleanSpectrum::builder()
            .min_mz(Some(200.0))
            .expect("min_mz should be valid")
            .build()
            .expect("cleaner should build");

        let cleaned = cleaner.clean_peaks(vec![(50.0, 1.0), (75.0, 2.0)]);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn builder_rejects_zero_max_peak_num() {
        let error = match MsEntropyCleanSpectrum::builder().max_peak_num(Some(0)) {
            Ok(_) => panic!("zero max_peak_num should be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            SimilarityConfigError::InvalidParameter("max_peak_num")
        );
    }

    #[test]
    fn check_centroid_uses_ppm_thresholds() {
        assert!(!check_centroid(
            &[(100.0, 1.0), (100.00005, 1.0)],
            0.0,
            Some(1.0)
        ));
        assert!(check_centroid(
            &[(100.0, 1.0), (100.0003, 1.0)],
            0.0,
            Some(1.0)
        ));
    }

    #[test]
    fn centroid_once_drops_zero_sum_clusters() {
        let peaks = centroid_once(vec![(100.0, 0.0), (100.05, 0.0)], 0.1, None);
        assert!(peaks.is_empty());
    }

    #[test]
    fn centroid_once_clears_clusters_whose_total_intensity_cancels_out() {
        let peaks = centroid_once(vec![(100.0, 1.0), (100.05, -1.5), (100.08, 0.5)], 0.1, None);
        assert!(peaks.is_empty());
    }

    #[test]
    fn clean_peaks_keeps_non_positive_bounds_inactive() {
        let cleaner = MsEntropyCleanSpectrum::builder()
            .min_mz(Some(0.0))
            .expect("zero min_mz should be accepted")
            .max_mz(Some(0.0))
            .expect("zero max_mz should be accepted")
            .build()
            .expect("cleaner should build");
        let cleaned = cleaner.clean_peaks(vec![(50.0, 1.0)]);
        assert_eq!(cleaned, default_cleaner().clean_peaks(vec![(50.0, 1.0)]));
    }
}
