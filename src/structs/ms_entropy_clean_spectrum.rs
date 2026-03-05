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
use core::marker::PhantomData;

use geometric_traits::prelude::{Finite, Number};
use num_traits::{Float, NumCast, ToPrimitive};

use super::cosine_common::{to_f64_checked_for_computation, validate_numeric_parameter};
use super::similarity_errors::SimilarityConfigError;
use crate::structs::GenericSpectrum;
use crate::traits::{SpectralProcessor, Spectrum, SpectrumMut};

/// Spectral processor mirroring `ms_entropy.clean_spectrum` behavior.
pub struct MsEntropyCleanSpectrum<MZ> {
    min_mz: Option<f64>,
    max_mz: Option<f64>,
    noise_threshold: Option<f64>,
    min_ms2_difference_in_da: f64,
    min_ms2_difference_in_ppm: Option<f64>,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
    _marker: PhantomData<MZ>,
}

impl<MZ> MsEntropyCleanSpectrum<MZ> {
    /// Returns a builder configured with `ms_entropy` defaults.
    #[inline]
    pub fn builder() -> MsEntropyCleanSpectrumBuilder<MZ> {
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

    #[inline]
    fn centroid_tolerance(&self) -> (f64, Option<f64>) {
        let ppm = self.min_ms2_difference_in_ppm.filter(|v| *v > 0.0);
        (self.min_ms2_difference_in_da, ppm)
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
        let (da, ppm) = self.centroid_tolerance();
        peaks = centroid_spectrum(peaks, da, ppm);

        if peaks.is_empty() {
            return peaks;
        }

        // Step 4. Noise filtering.
        if let Some(noise_threshold) = self.noise_threshold {
            let max_intensity = peaks
                .iter()
                .map(|(_, intensity)| *intensity)
                .fold(0.0, f64::max);
            peaks.retain(|(_, intensity)| *intensity >= noise_threshold * max_intensity);
        }

        if peaks.is_empty() {
            return peaks;
        }

        // Step 5. Keep top-N peaks.
        if let Some(max_peak_num) = self.max_peak_num
            && max_peak_num > 0
            && peaks.len() > max_peak_num
        {
            peaks.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            peaks = peaks[peaks.len() - max_peak_num..].to_vec();
            peaks.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        }

        // Step 6. Normalize to sum 1.
        if self.normalize_intensity {
            let spectrum_sum: f64 = peaks.iter().map(|(_, intensity)| *intensity).sum();
            if spectrum_sum > 0.0 {
                for (_, intensity) in &mut peaks {
                    *intensity /= spectrum_sum;
                }
            } else {
                peaks.clear();
            }
        }

        peaks
    }
}

impl<MZ> SpectralProcessor for MsEntropyCleanSpectrum<MZ>
where
    MZ: Float + NumCast + Number + Finite + ToPrimitive,
{
    type Spectrum = GenericSpectrum<MZ, MZ>;

    fn process(&self, spectrum: &Self::Spectrum) -> Self::Spectrum {
        let input_peaks: Vec<(f64, f64)> = spectrum
            .peaks()
            .map(|(mz, intensity)| {
                let mz = to_f64_checked_for_computation(mz, "mz")
                    .expect("input spectrum m/z values must be finite/representable");
                let intensity = to_f64_checked_for_computation(intensity, "intensity")
                    .expect("input spectrum intensities must be finite/representable");
                (mz, intensity)
            })
            .collect();

        let cleaned = self.clean_peaks(input_peaks);

        let mut result = GenericSpectrum::try_with_capacity(spectrum.precursor_mz(), cleaned.len())
            .expect("precursor_mz from valid spectrum must be valid");

        for (mz, intensity) in cleaned {
            let mz_cast: MZ = NumCast::from(mz)
                .expect("cleaned m/z must be representable in output numeric type");
            let intensity_cast: MZ = NumCast::from(intensity)
                .expect("cleaned intensity must be representable in output numeric type");
            result
                .add_peak(mz_cast, intensity_cast)
                .expect("cleaned peaks should be valid and sorted by m/z");
        }

        result
    }
}

/// Builder for [`MsEntropyCleanSpectrum`].
pub struct MsEntropyCleanSpectrumBuilder<MZ> {
    min_mz: Option<f64>,
    max_mz: Option<f64>,
    noise_threshold: Option<f64>,
    min_ms2_difference_in_da: f64,
    min_ms2_difference_in_ppm: Option<f64>,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
    _marker: PhantomData<MZ>,
}

impl<MZ> Default for MsEntropyCleanSpectrumBuilder<MZ> {
    fn default() -> Self {
        Self {
            min_mz: None,
            max_mz: None,
            noise_threshold: Some(0.01),
            min_ms2_difference_in_da: 0.05,
            min_ms2_difference_in_ppm: None,
            max_peak_num: None,
            normalize_intensity: true,
            _marker: PhantomData,
        }
    }
}

impl<MZ> MsEntropyCleanSpectrumBuilder<MZ> {
    #[inline]
    pub fn min_mz(mut self, min_mz: Option<f64>) -> Self {
        self.min_mz = min_mz;
        self
    }

    #[inline]
    pub fn max_mz(mut self, max_mz: Option<f64>) -> Self {
        self.max_mz = max_mz;
        self
    }

    #[inline]
    pub fn noise_threshold(mut self, noise_threshold: Option<f64>) -> Self {
        self.noise_threshold = noise_threshold;
        self
    }

    #[inline]
    pub fn min_ms2_difference_in_da(mut self, min_ms2_difference_in_da: f64) -> Self {
        self.min_ms2_difference_in_da = min_ms2_difference_in_da;
        self
    }

    #[inline]
    pub fn min_ms2_difference_in_ppm(mut self, min_ms2_difference_in_ppm: Option<f64>) -> Self {
        self.min_ms2_difference_in_ppm = min_ms2_difference_in_ppm;
        self
    }

    #[inline]
    pub fn max_peak_num(mut self, max_peak_num: Option<usize>) -> Self {
        self.max_peak_num = max_peak_num;
        self
    }

    #[inline]
    pub fn normalize_intensity(mut self, normalize_intensity: bool) -> Self {
        self.normalize_intensity = normalize_intensity;
        self
    }

    pub fn build(self) -> Result<MsEntropyCleanSpectrum<MZ>, SimilarityConfigError> {
        if let Some(v) = self.min_mz {
            validate_numeric_parameter(v, "min_mz")?;
        }
        if let Some(v) = self.max_mz {
            validate_numeric_parameter(v, "max_mz")?;
        }
        if let Some(v) = self.noise_threshold {
            validate_numeric_parameter(v, "noise_threshold")?;
        }
        validate_numeric_parameter(self.min_ms2_difference_in_da, "min_ms2_difference_in_da")?;
        if let Some(v) = self.min_ms2_difference_in_ppm {
            validate_numeric_parameter(v, "min_ms2_difference_in_ppm")?;
        }

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
            _marker: PhantomData,
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
            .all(|w| (w[1].0 - w[0].0) >= (w[1].0 * ppm * 1e-6))
    } else {
        peaks.windows(2).all(|w| (w[1].0 - w[0].0) >= ms2_da)
    }
}

fn centroid_spectrum(
    mut peaks: Vec<(f64, f64)>,
    ms2_da: f64,
    ms2_ppm: Option<f64>,
) -> Vec<(f64, f64)> {
    peaks.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    while !check_centroid(&peaks, ms2_da, ms2_ppm) {
        peaks = centroid_once(peaks, ms2_da, ms2_ppm);
    }
    peaks
}

fn centroid_once(mut peaks: Vec<(f64, f64)>, ms2_da: f64, ms2_ppm: Option<f64>) -> Vec<(f64, f64)> {
    let mut intensity_order: Vec<usize> = (0..peaks.len()).collect();
    intensity_order.sort_unstable_by(|&a, &b| peaks[a].1.total_cmp(&peaks[b].1));

    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(peaks.len());

    for &idx in intensity_order.iter().rev() {
        if peaks[idx].1 <= 0.0 {
            continue;
        }

        let mz = peaks[idx].0;
        let (mz_delta_allowed_left, mz_delta_allowed_right) = if let Some(ppm) = ms2_ppm {
            let left = mz * ppm * 1e-6;
            let right = mz / (1.0 - ppm * 1e-6) - mz;
            (left, right)
        } else {
            (ms2_da, ms2_da)
        };

        let mut left_idx = idx;
        while left_idx > 0 && (mz - peaks[left_idx - 1].0) <= mz_delta_allowed_left {
            left_idx -= 1;
        }

        let mut right_idx = idx + 1;
        while right_idx < peaks.len() && (peaks[right_idx].0 - mz) <= mz_delta_allowed_right {
            right_idx += 1;
        }

        let mut intensity_sum = 0.0;
        let mut intensity_weighted_mz_sum = 0.0;
        for (peak_mz, peak_intensity) in peaks[left_idx..right_idx].iter().copied() {
            intensity_sum += peak_intensity;
            intensity_weighted_mz_sum += peak_mz * peak_intensity;
        }

        if intensity_sum > 0.0 {
            merged.push((intensity_weighted_mz_sum / intensity_sum, intensity_sum));
        }

        for peak in peaks[left_idx..right_idx].iter_mut() {
            peak.1 = 0.0;
        }
    }

    merged.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    merged
}
