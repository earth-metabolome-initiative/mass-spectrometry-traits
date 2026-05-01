//! SPLASH spectral hash generation.
//!
//! SPLASH is a database-independent hashed identifier for spectra, introduced
//! to make spectra easier to exchange, compare, and de-duplicate across
//! libraries. A SPLASH string has four dash-separated blocks:
//!
//! 1. the version and spectrum type block, such as `splash10` for MS data;
//! 2. a short prefilter block from the most prominent ions;
//! 3. a coarser similarity histogram block;
//! 4. a truncated SHA-256 hash of the canonicalized peak list.
//!
//! This module implements the SPLASH10 algorithm used by the Java/Python
//! reference implementations for MS spectra.
//!
//! References:
//!
//! * Wohlgemuth et al., "SPLASH, a hashed identifier for mass spectra",
//!   Nature Biotechnology 34, 1099-1101 (2016),
//!   <https://doi.org/10.1038/nbt.3689>.
//! * SPLASH project page: <https://splash.fiehnlab.ucdavis.edu/>.

use alloc::{string::String, vec::Vec};
use core::cmp::Ordering;

use sha2::{Digest, Sha256};

use crate::traits::{Spectrum, SpectrumFloat};

const EPS_CORRECTION: f64 = 1.0e-7;
const RELATIVE_INTENSITY_SCALE: f64 = 100.0;
const MAX_HASH_CHARACTERS: usize = 20;
const MZ_PRECISION_FACTOR: f64 = 1_000_000.0;
const INTENSITY_PRECISION_FACTOR: f64 = 1.0;
const PREFILTER_BASE: usize = 3;
const PREFILTER_LENGTH: usize = 10;
const PREFILTER_BIN_SIZE: f64 = 5.0;
const PREFILTER_TOP_IONS: usize = 10;
const PREFILTER_BASE_PEAK_PERCENTAGE: f64 = 0.1;
const SIMILARITY_BASE: usize = 10;
const SIMILARITY_LENGTH: usize = 10;
const SIMILARITY_BIN_SIZE: f64 = 100.0;
const INTENSITY_MAP: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
const HEX: &[u8; 16] = b"0123456789abcdef";

/// Error returned when generating a SPLASH code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SplashError {
    /// SPLASH cannot be computed for an empty peak list.
    #[error("SPLASH cannot be computed for an empty spectrum")]
    EmptySpectrum,
    /// At least one peak must have positive intensity.
    #[error("SPLASH cannot be computed when all intensities are zero")]
    AllZeroIntensities,
    /// Peak m/z values must be finite.
    #[error("mz values must be finite")]
    NonFiniteMz,
    /// Peak intensities must be finite.
    #[error("intensity values must be finite")]
    NonFiniteIntensity,
    /// Peak m/z values must be non-negative.
    #[error("mz values must be >= 0")]
    NegativeMz,
    /// Peak intensities must be non-negative.
    #[error("intensity values must be >= 0")]
    NegativeIntensity,
    /// A value exceeded the integer range supported by this implementation.
    #[error("value `{0}` is too large to encode as SPLASH")]
    ValueTooLarge(&'static str),
}

/// Spectrum type encoded in the first SPLASH block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SplashSpectrumType {
    /// Mass spectrum.
    #[default]
    MassSpectrum,
    /// Nuclear magnetic resonance spectrum.
    Nmr,
    /// Ultraviolet spectrum.
    Uv,
    /// Infrared spectrum.
    Ir,
    /// Raman spectrum.
    Raman,
}

impl SplashSpectrumType {
    /// Returns the numeric SPLASH spectrum-type code used in the initial
    /// `splashX0` block.
    #[inline]
    const fn code(self) -> u8 {
        match self {
            Self::MassSpectrum => 1,
            Self::Nmr => 2,
            Self::Uv => 3,
            Self::Ir => 4,
            Self::Raman => 5,
        }
    }
}

/// Converts values into a raw peak accepted by SPLASH generation.
///
/// This trait is implemented for `(P, P)` and `&(P, P)` where `P` is one of
/// the crate-supported spectrum floating-point precisions.
pub trait IntoSplashPeak {
    /// Returns `(m/z, intensity)` as `f64`.
    fn into_splash_peak(self) -> (f64, f64);
}

impl<P: SpectrumFloat> IntoSplashPeak for (P, P) {
    #[inline]
    fn into_splash_peak(self) -> (f64, f64) {
        (self.0.to_f64(), self.1.to_f64())
    }
}

impl<P: SpectrumFloat> IntoSplashPeak for &(P, P) {
    #[inline]
    fn into_splash_peak(self) -> (f64, f64) {
        (self.0.to_f64(), self.1.to_f64())
    }
}

#[derive(Debug, Clone, Copy)]
struct SplashPeak {
    /// Raw m/z value, kept unrounded until the exact hash block is encoded.
    mz: f64,
    /// Relative intensity after normalization to `RELATIVE_INTENSITY_SCALE`.
    intensity: f64,
}

/// Convenience extension trait for generating SPLASH codes from [`Spectrum`]
/// values.
pub trait SpectrumSplash: Spectrum {
    /// Returns the default mass-spectrum SPLASH code for this spectrum.
    ///
    /// # Errors
    ///
    /// Returns [`SplashError`] when the spectrum is empty, all intensities are
    /// zero, or any peak value is invalid for SPLASH.
    ///
    /// # Example
    ///
    /// ```
    /// use mass_spectrometry::prelude::*;
    ///
    /// let mut spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(250.0, 2).unwrap();
    /// spectrum.add_peak(100.0, 10.0).unwrap();
    /// spectrum.add_peak(200.0, 20.0).unwrap();
    ///
    /// assert_eq!(
    ///     spectrum.splash().unwrap(),
    ///     "splash10-0udi-0490000000-4425acda10ed7d4709bd"
    /// );
    /// ```
    fn splash(&self) -> Result<String, SplashError> {
        self.splash_with_type(SplashSpectrumType::MassSpectrum)
    }

    /// Returns a SPLASH code for this spectrum with the requested spectrum
    /// type encoded in the first block.
    ///
    /// # Errors
    ///
    /// Returns [`SplashError`] when the spectrum is empty, all intensities are
    /// zero, or any peak value is invalid for SPLASH.
    fn splash_with_type(&self, spectrum_type: SplashSpectrumType) -> Result<String, SplashError> {
        splash_from_peaks_with_type(self.peaks(), spectrum_type)
    }
}

impl<S: Spectrum + ?Sized> SpectrumSplash for S {}

/// Returns the default mass-spectrum SPLASH code for the provided peaks.
///
/// Unlike [`GenericSpectrum`](crate::structs::GenericSpectrum), the SPLASH raw
/// API accepts duplicate m/z values and zero-intensity peaks, matching the
/// reference implementations.
///
/// # Errors
///
/// Returns [`SplashError`] when the peak list is empty, all intensities are
/// zero, or any peak value is invalid for SPLASH.
///
/// # Example
///
/// ```
/// use mass_spectrometry::prelude::splash_from_peaks;
///
/// let splash = splash_from_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();
/// assert_eq!(splash, "splash10-0udi-0490000000-4425acda10ed7d4709bd");
/// ```
pub fn splash_from_peaks<I>(peaks: I) -> Result<String, SplashError>
where
    I: IntoIterator,
    I::Item: IntoSplashPeak,
{
    splash_from_peaks_with_type(peaks, SplashSpectrumType::MassSpectrum)
}

/// Returns a SPLASH code for the provided peaks and spectrum type.
///
/// # Errors
///
/// Returns [`SplashError`] when the peak list is empty, all intensities are
/// zero, or any peak value is invalid for SPLASH.
pub fn splash_from_peaks_with_type<I>(
    peaks: I,
    spectrum_type: SplashSpectrumType,
) -> Result<String, SplashError>
where
    I: IntoIterator,
    I::Item: IntoSplashPeak,
{
    let peaks = normalize_peaks(peaks)?;
    let prefilter = prefilter_block(&peaks)?;
    let similarity =
        calculate_histogram::<SIMILARITY_LENGTH>(&peaks, SIMILARITY_BASE, SIMILARITY_BIN_SIZE)?;
    let exact = encode_spectrum(&peaks)?;

    let mut splash = String::with_capacity(44);
    splash.push_str("splash");
    splash.push((b'0' + spectrum_type.code()) as char);
    splash.push('0');
    splash.push('-');
    splash.push_str(&prefilter);
    splash.push('-');
    push_histogram_digits(&mut splash, &similarity);
    splash.push('-');
    splash.push_str(&exact);
    Ok(splash)
}

/// Validates raw input peaks and normalizes intensities to the SPLASH relative
/// intensity scale.
///
/// The reference implementations normalize every intensity by the base peak
/// and multiply by 100 before any prefilter, histogram, or exact-hash work.
/// Duplicate m/z values and zero-intensity peaks are deliberately preserved;
/// they are part of the SPLASH input model even though `GenericSpectrum` is
/// stricter.
fn normalize_peaks<I>(peaks: I) -> Result<Vec<SplashPeak>, SplashError>
where
    I: IntoIterator,
    I::Item: IntoSplashPeak,
{
    let mut normalized = Vec::new();
    let mut max_intensity = 0.0_f64;

    for peak in peaks {
        let (mz, intensity) = peak.into_splash_peak();
        if !mz.is_finite() {
            return Err(SplashError::NonFiniteMz);
        }
        if !intensity.is_finite() {
            return Err(SplashError::NonFiniteIntensity);
        }
        if mz < 0.0 {
            return Err(SplashError::NegativeMz);
        }
        if intensity < 0.0 {
            return Err(SplashError::NegativeIntensity);
        }
        max_intensity = max_intensity.max(intensity);
        normalized.push(SplashPeak { mz, intensity });
    }

    if normalized.is_empty() {
        return Err(SplashError::EmptySpectrum);
    }
    if max_intensity == 0.0 {
        return Err(SplashError::AllZeroIntensities);
    }

    for peak in &mut normalized {
        peak.intensity = peak.intensity / max_intensity * RELATIVE_INTENSITY_SCALE;
    }

    Ok(normalized)
}

/// Builds the second SPLASH block from the highest-signal ions.
///
/// SPLASH first keeps ions whose relative intensity is at least 10% of the
/// base peak, then keeps the top ten by descending intensity and ascending
/// m/z. The resulting ten-bin base-3 wrapped histogram is translated to a
/// four-character base-36 block.
fn prefilter_block(peaks: &[SplashPeak]) -> Result<String, SplashError> {
    let base_peak_intensity = peaks
        .iter()
        .map(|peak| peak.intensity)
        .fold(0.0_f64, f64::max);
    let threshold = PREFILTER_BASE_PEAK_PERCENTAGE * base_peak_intensity;
    let mut filtered = Vec::with_capacity(PREFILTER_TOP_IONS);

    for &peak in peaks {
        if peak.intensity + EPS_CORRECTION < threshold {
            continue;
        }
        if filtered.len() < PREFILTER_TOP_IONS {
            filtered.push(peak);
            continue;
        }

        let (worst_index, worst_peak) = filtered
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| prefilter_order(left, right))
            .expect("top-10 buffer is not empty");
        if prefilter_order(&peak, worst_peak) == Ordering::Less {
            filtered[worst_index] = peak;
        }
    }

    filtered.sort_by(prefilter_order);

    let histogram =
        calculate_histogram::<PREFILTER_LENGTH>(&filtered, PREFILTER_BASE, PREFILTER_BIN_SIZE)?;
    Ok(translate_base_digits(&histogram, PREFILTER_BASE, 36, 4))
}

/// Orders peaks for the prefilter top-ion selection.
///
/// `Ordering::Less` means `left` is better and should appear earlier: higher
/// intensity wins, with lower m/z used as the tie-breaker.
#[inline]
fn prefilter_order(left: &SplashPeak, right: &SplashPeak) -> Ordering {
    right
        .intensity
        .total_cmp(&left.intensity)
        .then_with(|| left.mz.total_cmp(&right.mz))
}

/// Calculates a wrapped SPLASH histogram and returns its digit values.
///
/// Each ion is assigned to `(mz / bin_size) as usize % LENGTH`, matching the
/// reference wrapping strategy. The histogram is then normalized by its maximum
/// bin and scaled to digits in `0..base`, with the reference epsilon applied
/// before truncation.
fn calculate_histogram<const LENGTH: usize>(
    peaks: &[SplashPeak],
    base: usize,
    bin_size: f64,
) -> Result<[u8; LENGTH], SplashError> {
    let mut histogram = [0.0_f64; LENGTH];

    for peak in peaks {
        let scaled_mz = peak.mz / bin_size;
        if scaled_mz > usize::MAX as f64 {
            return Err(SplashError::ValueTooLarge("mz"));
        }
        let index = scaled_mz as usize % LENGTH;
        histogram[index] += peak.intensity;
    }

    let max_intensity = histogram.iter().copied().fold(0.0_f64, f64::max);
    if max_intensity == 0.0 {
        return Err(SplashError::AllZeroIntensities);
    }

    let mut encoded = [0u8; LENGTH];
    for (encoded_digit, value) in encoded.iter_mut().zip(histogram) {
        let digit = (EPS_CORRECTION + (base - 1) as f64 * value / max_intensity) as usize;
        *encoded_digit = digit as u8;
    }
    Ok(encoded)
}

/// Translates fixed-width histogram digits from one base to another.
///
/// The prefilter block is computed as ten base-3 digits and then represented
/// as a zero-padded base-36 string of length four.
fn translate_base_digits<const LENGTH: usize>(
    value: &[u8; LENGTH],
    initial_base: usize,
    final_base: usize,
    fill_length: usize,
) -> String {
    let mut n = 0usize;
    for &digit in value {
        n = n * initial_base + usize::from(digit);
    }

    let mut encoded = Vec::with_capacity(fill_length);
    while n > 0 {
        let digit = n % final_base;
        encoded.push(INTENSITY_MAP[digit]);
        n /= final_base;
    }

    let missing_zeroes = fill_length.saturating_sub(encoded.len());
    let mut padded = "0".repeat(missing_zeroes);
    padded.extend(encoded.iter().rev().map(|&byte| byte as char));
    padded
}

/// Appends histogram digits to the final SPLASH string using the SPLASH digit
/// alphabet.
fn push_histogram_digits<const LENGTH: usize>(target: &mut String, digits: &[u8; LENGTH]) {
    for &digit in digits {
        target.push(INTENSITY_MAP[usize::from(digit)] as char);
    }
}

/// Builds the exact-hash block from the canonical text representation.
///
/// The canonical representation is a space-separated list of
/// `formatted_mz:formatted_intensity` pairs. It is hashed with SHA-256 and
/// truncated to the first 20 hexadecimal characters.
fn encode_spectrum(peaks: &[SplashPeak]) -> Result<String, SplashError> {
    let mut sorted = peaks.to_vec();
    // The Java reference sorts by raw normalized m/z before formatting to
    // six-decimal integer m/z. This matters when two raw m/z values collapse
    // to the same formatted integer.
    sorted.sort_by(|left, right| {
        left.mz
            .total_cmp(&right.mz)
            .then_with(|| right.intensity.total_cmp(&left.intensity))
    });

    let mut encoded = String::with_capacity(sorted.len().saturating_mul(24));
    let mut mz_buffer = itoa::Buffer::new();
    let mut intensity_buffer = itoa::Buffer::new();
    for (index, peak) in sorted.iter().enumerate() {
        if index > 0 {
            encoded.push(' ');
        }
        encoded.push_str(mz_buffer.format(format_mz(peak.mz)?));
        encoded.push(':');
        encoded.push_str(intensity_buffer.format(format_intensity(peak.intensity)?));
    }

    let digest = Sha256::digest(encoded.as_bytes());
    let mut hash = String::with_capacity(MAX_HASH_CHARACTERS);
    for byte in digest.iter().take(MAX_HASH_CHARACTERS / 2) {
        hash.push(HEX[(byte >> 4) as usize] as char);
        hash.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(hash)
}

/// Formats m/z as the integer value used by the exact hash block.
///
/// SPLASH stores six decimal places by multiplying by `10^6`, applying the
/// reference epsilon first, and truncating to an integer.
#[inline]
fn format_mz(mz: f64) -> Result<u64, SplashError> {
    format_scaled("mz", mz, MZ_PRECISION_FACTOR)
}

/// Formats relative intensity as the integer value used by the exact hash
/// block.
///
/// The SPLASH10 MS reference uses zero decimal places for intensity, again
/// applying the reference epsilon before truncation.
#[inline]
fn format_intensity(intensity: f64) -> Result<u64, SplashError> {
    format_scaled("intensity", intensity, INTENSITY_PRECISION_FACTOR)
}

/// Applies the reference epsilon, scales a finite non-negative value, and
/// truncates it to the integer representation used by SPLASH.
fn format_scaled(label: &'static str, value: f64, factor: f64) -> Result<u64, SplashError> {
    let scaled = (value + EPS_CORRECTION) * factor;
    if scaled > u64::MAX as f64 {
        return Err(SplashError::ValueTooLarge(label));
    }
    Ok(scaled as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_base_pads_to_requested_width() {
        assert_eq!(translate_base_digits(&[0; 10], 3, 36, 4), "0000");
        assert_eq!(translate_base_digits(&[2; 10], 3, 36, 4), "19k8");
    }

    #[test]
    fn format_scaled_uses_reference_epsilon_before_truncation() {
        assert_eq!(format_intensity(1.999999947).unwrap(), 2);
        assert_eq!(format_mz(100.0000004).unwrap(), 100_000_000);
    }

    #[test]
    fn spectrum_type_codes_cover_all_splash_variants() {
        assert_eq!(SplashSpectrumType::MassSpectrum.code(), 1);
        assert_eq!(SplashSpectrumType::Nmr.code(), 2);
        assert_eq!(SplashSpectrumType::Uv.code(), 3);
        assert_eq!(SplashSpectrumType::Ir.code(), 4);
        assert_eq!(SplashSpectrumType::Raman.code(), 5);
    }

    #[test]
    fn prefilter_keeps_only_the_best_top_ions_after_buffer_fills() {
        let peaks = normalize_peaks([
            (0.0, 10.0),
            (5.0, 10.0),
            (10.0, 10.0),
            (15.0, 10.0),
            (20.0, 10.0),
            (25.0, 10.0),
            (30.0, 10.0),
            (35.0, 10.0),
            (40.0, 10.0),
            (45.0, 10.0),
            (50.0, 20.0),
        ])
        .unwrap();

        assert_eq!(prefilter_block(&peaks).unwrap(), "0udi");
    }

    #[test]
    fn histogram_rejects_mz_values_that_exceed_indexable_bins() {
        let peaks = [SplashPeak {
            mz: f64::MAX,
            intensity: 1.0,
        }];

        assert_eq!(
            calculate_histogram::<PREFILTER_LENGTH>(&peaks, PREFILTER_BASE, PREFILTER_BIN_SIZE)
                .unwrap_err(),
            SplashError::ValueTooLarge("mz")
        );
    }

    #[test]
    fn histogram_rejects_inputs_with_zero_total_signal() {
        let peaks = [SplashPeak {
            mz: 100.0,
            intensity: 0.0,
        }];

        assert_eq!(
            calculate_histogram::<PREFILTER_LENGTH>(&peaks, PREFILTER_BASE, PREFILTER_BIN_SIZE)
                .unwrap_err(),
            SplashError::AllZeroIntensities
        );
    }

    #[test]
    fn format_scaled_rejects_values_that_exceed_u64_range() {
        assert_eq!(
            format_scaled("probe", f64::MAX, 1.0).unwrap_err(),
            SplashError::ValueTooLarge("probe")
        );
    }

    #[test]
    fn exact_hash_sorts_by_raw_mz_before_integer_formatting() {
        let peaks = normalize_peaks([(100.0000004, 20.0), (100.0000001, 10.0)]).unwrap();
        assert_eq!(encode_spectrum(&peaks).unwrap(), "6dd39e8737923831fd73");
    }
}
