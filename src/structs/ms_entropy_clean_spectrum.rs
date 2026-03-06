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

use geometric_traits::prelude::{Finite, Number};
use num_traits::{Float, NumCast, ToPrimitive};

use super::cosine_common::{to_f64_checked_for_computation, validate_numeric_parameter};
use super::similarity_errors::SimilarityConfigError;
use crate::structs::GenericSpectrum;
use crate::traits::{SpectralProcessor, Spectrum, SpectrumMut};

/// Spectral processor mirroring `ms_entropy.clean_spectrum` behavior.
pub struct MsEntropyCleanSpectrum<MZ> {
    min_mz: Option<MZ>,
    max_mz: Option<MZ>,
    noise_threshold: Option<MZ>,
    min_ms2_difference_in_da: MZ,
    min_ms2_difference_in_ppm: Option<MZ>,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
}

impl<MZ> MsEntropyCleanSpectrum<MZ>
where
    MZ: NumCast,
{
    /// Returns a builder configured with `ms_entropy` defaults.
    #[inline]
    pub fn builder() -> MsEntropyCleanSpectrumBuilder<MZ> {
        MsEntropyCleanSpectrumBuilder::default()
    }
}

impl<MZ> MsEntropyCleanSpectrum<MZ>
where
    MZ: Number,
{
    /// Returns the configured minimum mz filter (enabled only when > 0).
    #[inline]
    pub fn min_mz(&self) -> Option<MZ> {
        self.min_mz
    }

    /// Returns the configured maximum mz filter (enabled only when > 0).
    #[inline]
    pub fn max_mz(&self) -> Option<MZ> {
        self.max_mz
    }

    /// Returns the configured relative noise threshold.
    #[inline]
    pub fn noise_threshold(&self) -> Option<MZ> {
        self.noise_threshold
    }

    /// Returns the configured Da centroid threshold.
    #[inline]
    pub fn min_ms2_difference_in_da(&self) -> MZ {
        self.min_ms2_difference_in_da
    }

    /// Returns the configured ppm centroid threshold.
    #[inline]
    pub fn min_ms2_difference_in_ppm(&self) -> Option<MZ> {
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
}

impl<MZ> MsEntropyCleanSpectrum<MZ>
where
    MZ: Number + ToPrimitive,
{
    #[inline]
    fn centroid_tolerance(&self) -> (f64, Option<f64>) {
        let da = to_f64_checked_for_computation(
            self.min_ms2_difference_in_da,
            "min_ms2_difference_in_da",
        )
        .expect("validated min_ms2_difference_in_da must be finite/representable");

        let ppm = self.min_ms2_difference_in_ppm.and_then(|value| {
            let ppm = to_f64_checked_for_computation(value, "min_ms2_difference_in_ppm")
                .expect("validated min_ms2_difference_in_ppm must be finite/representable");
            (ppm > 0.0).then_some(ppm)
        });

        (da, ppm)
    }

    fn clean_peaks(&self, mut peaks: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        // Step 1. Remove empty peaks.
        peaks.retain(|(mz, intensity)| *mz > 0.0 && *intensity > 0.0);

        // Step 2. Min/max mz filtering.
        if let Some(min_mz) = self.min_mz.and_then(|value| {
            let min_mz = to_f64_checked_for_computation(value, "min_mz")
                .expect("validated min_mz must be finite/representable");
            (min_mz > 0.0).then_some(min_mz)
        }) {
            peaks.retain(|(mz, _)| *mz >= min_mz);
        }
        if let Some(max_mz) = self.max_mz.and_then(|value| {
            let max_mz = to_f64_checked_for_computation(value, "max_mz")
                .expect("validated max_mz must be finite/representable");
            (max_mz > 0.0).then_some(max_mz)
        }) {
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
        if let Some(noise_threshold) = self.noise_threshold.map(|value| {
            to_f64_checked_for_computation(value, "noise_threshold")
                .expect("validated noise_threshold must be finite/representable")
        }) {
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
    min_mz: Option<MZ>,
    max_mz: Option<MZ>,
    noise_threshold: Option<MZ>,
    min_ms2_difference_in_da: MZ,
    min_ms2_difference_in_ppm: Option<MZ>,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
}

impl<MZ> Default for MsEntropyCleanSpectrumBuilder<MZ>
where
    MZ: NumCast,
{
    fn default() -> Self {
        Self {
            min_mz: None,
            max_mz: None,
            noise_threshold: Some(
                NumCast::from(0.01)
                    .expect("0.01 must be representable in the configured numeric type"),
            ),
            min_ms2_difference_in_da: NumCast::from(0.05)
                .expect("0.05 must be representable in the configured numeric type"),
            min_ms2_difference_in_ppm: None,
            max_peak_num: None,
            normalize_intensity: true,
        }
    }
}

impl<MZ> MsEntropyCleanSpectrumBuilder<MZ> {
    /// Sets the minimum m/z filter.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `min_mz` is not finite or not representable as `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mass_spectrometry::prelude::{MsEntropyCleanSpectrum, SimilarityConfigError};
    ///
    /// let cleaner = MsEntropyCleanSpectrum::<f32>::builder()
    ///     .min_mz(Some(100.0))
    ///     .expect("finite value")
    ///     .build()
    ///     .expect("valid configuration");
    /// assert_eq!(cleaner.min_mz(), Some(100.0));
    ///
    /// let err = MsEntropyCleanSpectrum::<f32>::builder().min_mz(Some(f32::NAN));
    /// assert!(matches!(
    ///     err,
    ///     Err(SimilarityConfigError::NonFiniteParameter("min_mz"))
    /// ));
    /// ```
    #[inline]
    pub fn min_mz(mut self, min_mz: Option<MZ>) -> Result<Self, SimilarityConfigError>
    where
        MZ: ToPrimitive + Copy,
    {
        if let Some(v) = min_mz {
            validate_numeric_parameter(v, "min_mz")?;
        }
        self.min_mz = min_mz;
        Ok(self)
    }

    /// Sets the maximum m/z filter.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `max_mz` is not finite or not representable as `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mass_spectrometry::prelude::{MsEntropyCleanSpectrum, SimilarityConfigError};
    ///
    /// let cleaner = MsEntropyCleanSpectrum::<f32>::builder()
    ///     .max_mz(Some(500.0))
    ///     .expect("finite value")
    ///     .build()
    ///     .expect("valid configuration");
    /// assert_eq!(cleaner.max_mz(), Some(500.0));
    ///
    /// let err = MsEntropyCleanSpectrum::<f32>::builder().max_mz(Some(f32::INFINITY));
    /// assert!(matches!(
    ///     err,
    ///     Err(SimilarityConfigError::NonFiniteParameter("max_mz"))
    /// ));
    /// ```
    #[inline]
    pub fn max_mz(mut self, max_mz: Option<MZ>) -> Result<Self, SimilarityConfigError>
    where
        MZ: ToPrimitive + Copy,
    {
        if let Some(v) = max_mz {
            validate_numeric_parameter(v, "max_mz")?;
        }
        self.max_mz = max_mz;
        Ok(self)
    }

    /// Sets the relative noise threshold.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `noise_threshold` is not finite or not representable
    /// as `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mass_spectrometry::prelude::{MsEntropyCleanSpectrum, SimilarityConfigError};
    ///
    /// let cleaner = MsEntropyCleanSpectrum::<f32>::builder()
    ///     .noise_threshold(Some(0.02))
    ///     .expect("finite value")
    ///     .build()
    ///     .expect("valid configuration");
    /// assert_eq!(cleaner.noise_threshold(), Some(0.02));
    ///
    /// let err = MsEntropyCleanSpectrum::<f32>::builder().noise_threshold(Some(f32::NAN));
    /// assert!(matches!(
    ///     err,
    ///     Err(SimilarityConfigError::NonFiniteParameter("noise_threshold"))
    /// ));
    /// ```
    #[inline]
    pub fn noise_threshold(
        mut self,
        noise_threshold: Option<MZ>,
    ) -> Result<Self, SimilarityConfigError>
    where
        MZ: ToPrimitive + Copy,
    {
        if let Some(v) = noise_threshold {
            validate_numeric_parameter(v, "noise_threshold")?;
        }
        self.noise_threshold = noise_threshold;
        Ok(self)
    }

    /// Sets the minimum Da centroid distance.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `min_ms2_difference_in_da` is not finite or not
    /// representable as `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mass_spectrometry::prelude::{MsEntropyCleanSpectrum, SimilarityConfigError};
    ///
    /// let cleaner = MsEntropyCleanSpectrum::<f32>::builder()
    ///     .min_ms2_difference_in_da(0.1)
    ///     .expect("finite value")
    ///     .build()
    ///     .expect("valid configuration");
    /// assert_eq!(cleaner.min_ms2_difference_in_da(), 0.1);
    ///
    /// let err = MsEntropyCleanSpectrum::<f32>::builder().min_ms2_difference_in_da(f32::NAN);
    /// assert!(matches!(
    ///     err,
    ///     Err(SimilarityConfigError::NonFiniteParameter(
    ///         "min_ms2_difference_in_da"
    ///     ))
    /// ));
    /// ```
    #[inline]
    pub fn min_ms2_difference_in_da(
        mut self,
        min_ms2_difference_in_da: MZ,
    ) -> Result<Self, SimilarityConfigError>
    where
        MZ: ToPrimitive + Copy,
    {
        validate_numeric_parameter(min_ms2_difference_in_da, "min_ms2_difference_in_da")?;
        self.min_ms2_difference_in_da = min_ms2_difference_in_da;
        Ok(self)
    }

    /// Sets the minimum ppm centroid distance.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `min_ms2_difference_in_ppm` is not finite or not
    /// representable as `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mass_spectrometry::prelude::{MsEntropyCleanSpectrum, SimilarityConfigError};
    ///
    /// let cleaner = MsEntropyCleanSpectrum::<f32>::builder()
    ///     .min_ms2_difference_in_ppm(Some(20.0))
    ///     .expect("finite value")
    ///     .build()
    ///     .expect("valid configuration");
    /// assert_eq!(cleaner.min_ms2_difference_in_ppm(), Some(20.0));
    ///
    /// let err =
    ///     MsEntropyCleanSpectrum::<f32>::builder().min_ms2_difference_in_ppm(Some(f32::NAN));
    /// assert!(matches!(
    ///     err,
    ///     Err(SimilarityConfigError::NonFiniteParameter(
    ///         "min_ms2_difference_in_ppm"
    ///     ))
    /// ));
    /// ```
    #[inline]
    pub fn min_ms2_difference_in_ppm(
        mut self,
        min_ms2_difference_in_ppm: Option<MZ>,
    ) -> Result<Self, SimilarityConfigError>
    where
        MZ: ToPrimitive + Copy,
    {
        if let Some(v) = min_ms2_difference_in_ppm {
            validate_numeric_parameter(v, "min_ms2_difference_in_ppm")?;
        }
        self.min_ms2_difference_in_ppm = min_ms2_difference_in_ppm;
        Ok(self)
    }

    /// Sets the maximum number of retained peaks.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError::InvalidParameter`] when `max_peak_num` is `Some(0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mass_spectrometry::prelude::{MsEntropyCleanSpectrum, SimilarityConfigError};
    ///
    /// let cleaner = MsEntropyCleanSpectrum::<f32>::builder()
    ///     .max_peak_num(Some(10))
    ///     .expect("positive value")
    ///     .build()
    ///     .expect("valid configuration");
    /// assert_eq!(cleaner.max_peak_num(), Some(10));
    ///
    /// let err = MsEntropyCleanSpectrum::<f32>::builder().max_peak_num(Some(0));
    /// assert!(matches!(
    ///     err,
    ///     Err(SimilarityConfigError::InvalidParameter("max_peak_num"))
    /// ));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mass_spectrometry::prelude::MsEntropyCleanSpectrum;
    ///
    /// let cleaner = MsEntropyCleanSpectrum::<f32>::builder()
    ///     .normalize_intensity(false)
    ///     .expect("always valid")
    ///     .build()
    ///     .expect("valid configuration");
    /// assert!(!cleaner.normalize_intensity());
    /// ```
    #[inline]
    pub fn normalize_intensity(
        mut self,
        normalize_intensity: bool,
    ) -> Result<Self, SimilarityConfigError> {
        self.normalize_intensity = normalize_intensity;
        Ok(self)
    }

    /// Builds the cleaner from validated builder fields.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError::InvalidParameter`] when the centroiding configuration
    /// is invalid, i.e. when both `min_ms2_difference_in_da` and
    /// `min_ms2_difference_in_ppm` are non-positive/disabled.
    ///
    /// Per-field numeric validation is performed in setter methods.
    pub fn build(self) -> Result<MsEntropyCleanSpectrum<MZ>, SimilarityConfigError>
    where
        MZ: Float + Number + ToPrimitive,
    {
        let ppm_positive = self
            .min_ms2_difference_in_ppm
            .is_some_and(|ppm| ppm > MZ::zero());
        let da_positive = self.min_ms2_difference_in_da > MZ::zero();
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
