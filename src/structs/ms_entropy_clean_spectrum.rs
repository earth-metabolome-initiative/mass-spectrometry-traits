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

use super::cosine_common::validate_numeric_parameter;
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
    MZ: Float + Number + ToPrimitive,
{
    fn clean_peaks(&self, mut peaks: Vec<(MZ, MZ)>) -> Vec<(MZ, MZ)> {
        // Step 1. Remove empty peaks.
        peaks.retain(|(mz, intensity)| *mz > MZ::zero() && *intensity > MZ::zero());

        // Step 2. Min/max mz filtering.
        if let Some(min_mz) = self.min_mz
            && min_mz > MZ::zero()
        {
            peaks.retain(|(mz, _)| *mz >= min_mz);
        }
        if let Some(max_mz) = self.max_mz
            && max_mz > MZ::zero()
        {
            peaks.retain(|(mz, _)| *mz <= max_mz);
        }

        if peaks.is_empty() {
            return peaks;
        }

        // Step 3. Centroiding.
        let ppm = self.min_ms2_difference_in_ppm.filter(|&v| v > MZ::zero());
        peaks = centroid_spectrum(peaks, self.min_ms2_difference_in_da, ppm);

        if peaks.is_empty() {
            return peaks;
        }

        // Step 4. Noise filtering.
        if let Some(noise_threshold) = self.noise_threshold {
            let max_intensity = peaks
                .iter()
                .map(|&(_, intensity)| intensity)
                .fold(MZ::zero(), |a, b| if b > a { b } else { a });
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
            let spectrum_sum = peaks
                .iter()
                .fold(MZ::zero(), |acc, &(_, intensity)| acc + intensity);
            if spectrum_sum > MZ::zero() {
                for (_, intensity) in &mut peaks {
                    *intensity = *intensity / spectrum_sum;
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
        let input_peaks: Vec<(MZ, MZ)> = spectrum.peaks().collect();

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

/// Returns `true` when all consecutive mz gaps strictly exceed the tolerance,
/// matching the C `need_centroid` function which triggers on `<=`.
#[inline]
fn check_centroid<F: Float>(peaks: &[(F, F)], ms2_da: F, ms2_ppm: Option<F>) -> bool {
    if peaks.len() <= 1 {
        return true;
    }
    if let Some(ppm) = ms2_ppm {
        let ppm_factor: F = F::from(1e-6).expect("1e-6 representable");
        peaks
            .windows(2)
            .all(|w| (w[1].0 - w[0].0) > (w[1].0 * ppm * ppm_factor))
    } else {
        peaks.windows(2).all(|w| (w[1].0 - w[0].0) > ms2_da)
    }
}

fn centroid_spectrum<F: Float>(
    mut peaks: Vec<(F, F)>,
    ms2_da: F,
    ms2_ppm: Option<F>,
) -> Vec<(F, F)> {
    peaks.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN mz"));
    while !check_centroid(&peaks, ms2_da, ms2_ppm) {
        peaks = centroid_once(peaks, ms2_da, ms2_ppm);
    }
    peaks
}

/// In-place centroiding matching the C implementation in ms_entropy.
///
/// Peaks are modified in-place: merged peaks keep the weighted-average mz and
/// summed intensity at the center index; all other peaks in the merge range
/// have their intensity zeroed. After the pass, zeroed peaks are removed and
/// the remaining peaks are sorted by mz.
fn centroid_once<F: Float>(mut peaks: Vec<(F, F)>, ms2_da: F, ms2_ppm: Option<F>) -> Vec<(F, F)> {
    let n = peaks.len();
    let mut intensity_order: Vec<usize> = (0..n).collect();
    // Sort descending by intensity (highest first), matching the C quicksort
    // which sorts descending then iterates forward.
    intensity_order.sort_unstable_by(|&a, &b| {
        peaks[b]
            .1
            .partial_cmp(&peaks[a].1)
            .expect("non-NaN intensity")
    });

    let ppm_factor: F = F::from(1e-6).expect("1e-6 representable");

    for &idx in &intensity_order {
        if peaks[idx].1 <= F::zero() {
            continue;
        }

        let mz = peaks[idx].0;
        let (delta_left, delta_right) = if let Some(ppm) = ms2_ppm {
            let left = mz * ppm * ppm_factor;
            let right = mz * ppm / (F::from(1e6).expect("1e6 representable") - ppm);
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

        let mut intensity_sum = F::zero();
        let mut intensity_weighted_mz_sum = F::zero();
        for peak in peaks.iter_mut().take(right_idx).skip(left_idx) {
            intensity_sum = intensity_sum + peak.1;
            intensity_weighted_mz_sum = intensity_weighted_mz_sum + peak.1 * peak.0;
            peak.1 = F::zero();
        }

        if intensity_sum > F::zero() {
            peaks[idx].0 = intensity_weighted_mz_sum / intensity_sum;
        } else {
            peaks[idx].0 = F::zero();
        }
        peaks[idx].1 = intensity_sum;
    }

    // Remove zeroed-out peaks and sort by mz (matching sort_spectrum_by_mz_and_zero_intensity).
    peaks.retain(|&(_, intensity)| intensity > F::zero());
    peaks.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN mz"));
    peaks
}
