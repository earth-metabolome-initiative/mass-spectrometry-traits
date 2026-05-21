//! Submodule providing a trait for a mutable Spectrum.

use alloc::vec::Vec;

use super::{Spectrum, SpectrumFloat};

/// Parameters for generating a random spectrum with [`SpectrumAlloc::random`].
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomSpectrumConfig {
    /// Precursor m/z for the generated spectrum.
    pub precursor_mz: f64,
    /// Number of peaks to generate.
    pub n_peaks: usize,
    /// Minimum generated peak m/z.
    pub mz_min: f64,
    /// Maximum generated peak m/z.
    pub mz_max: f64,
    /// Minimum spacing between consecutive m/z values.
    pub min_peak_gap: f64,
    /// Minimum generated intensity.
    pub intensity_min: f64,
    /// Maximum generated intensity.
    pub intensity_max: f64,
}

/// Error returned by [`SpectrumAlloc::random`].
#[derive(Debug, thiserror::Error)]
pub enum RandomSpectrumGenerationError<E>
where
    E: core::error::Error,
{
    /// Input parameter set is invalid.
    #[error("invalid random spectrum config: {0}")]
    InvalidConfig(&'static str),
    /// Input value is non-finite.
    #[error("value must be finite: {0}")]
    NonFiniteValue(&'static str),
    /// Error while constructing or mutating the spectrum.
    #[error(transparent)]
    Mutation(E),
}

#[inline]
fn nonzero_seed(seed: u64) -> u64 {
    if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    }
}

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn next_unit_f64(state: &mut u64) -> f64 {
    // 53 random bits scaled to [0, 1).
    const INV_2POW53: f64 = 1.0 / ((1u64 << 53) as f64);
    ((next_u64(state) >> 11) as f64) * INV_2POW53
}

#[inline]
fn ensure_finite<E: core::error::Error>(
    value: f64,
    name: &'static str,
) -> Result<f64, RandomSpectrumGenerationError<E>> {
    if !value.is_finite() {
        return Err(RandomSpectrumGenerationError::NonFiniteValue(name));
    }
    Ok(value)
}

/// Trait for a mutable Spectrum.
pub trait SpectrumMut: Spectrum {
    /// The type of error that can occur when mutating the Spectrum.
    type MutationError: core::error::Error;

    /// Add a peak to the Spectrum.
    ///
    /// Implementations are expected to reject non-finite values, enforce
    /// strictly increasing `mz` ordering (rejecting duplicates), reject
    /// non-positive intensity values, and validate mz within
    /// `[ELECTRON_MASS, MAX_MZ]`.
    fn add_peak(
        &mut self,
        mz: Self::Precision,
        intensity: Self::Precision,
    ) -> Result<&mut Self, Self::MutationError>;

    /// Add several peaks to the Spectrum.
    ///
    /// The peaks are added in iteration order, so implementations that require
    /// sorted input keep the same invariant as [`Self::add_peak`].
    fn add_peaks<I>(&mut self, peaks: I) -> Result<&mut Self, Self::MutationError>
    where
        I: IntoIterator<Item = (Self::Precision, Self::Precision)>,
    {
        for (mz, intensity) in peaks {
            self.add_peak(mz, intensity)?;
        }
        Ok(self)
    }
}

/// Trait for an allocable Spectrum.
pub trait SpectrumAlloc: SpectrumMut + Sized {
    /// Create a new Spectrum with a given capacity.
    ///
    /// # Arguments
    ///
    /// * `precursor_mz`: The precursor mass over charge.
    /// * `capacity`: The capacity of the Spectrum.
    ///
    /// Implementations are expected to enforce constructor-time invariants for
    /// `precursor_mz`, returning an error when the value is invalid.
    fn with_capacity(precursor_mz: f64, capacity: usize) -> Result<Self, Self::MutationError>;

    /// Returns a new spectrum containing only the `k` most intense peaks.
    ///
    /// The selected peaks are stored in m/z order, preserving the normal
    /// [`Spectrum`] ordering invariant. When more peaks have the same
    /// intensity than can be kept, the lowest m/z values are retained first.
    ///
    /// # Errors
    ///
    /// Returns [`SpectrumMut::MutationError`] if constructing the new spectrum or
    /// adding one of the retained peaks fails.
    ///
    /// # Example
    ///
    /// ```
    /// use mass_spectrometry::prelude::*;
    ///
    /// let mut spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(250.0, 4).unwrap();
    /// spectrum
    ///     .add_peaks([(50.0, 1.0), (75.0, 5.0), (100.0, 3.0), (125.0, 5.0)])
    ///     .unwrap();
    ///
    /// let top: GenericSpectrum = spectrum.top_k_peaks(2).unwrap();
    /// assert_eq!(top.peaks().collect::<Vec<_>>(), vec![(75.0, 5.0), (125.0, 5.0)]);
    /// ```
    fn top_k_peaks(&self, k: usize) -> Result<Self, Self::MutationError> {
        let mut peaks: Vec<(usize, Self::Precision, Self::Precision)> = self
            .peaks()
            .enumerate()
            .map(|(index, (mz, intensity))| (index, mz, intensity))
            .collect();

        if k == 0 {
            peaks.clear();
        } else if k < peaks.len() {
            peaks.sort_unstable_by(|left, right| {
                right
                    .2
                    .to_f64()
                    .total_cmp(&left.2.to_f64())
                    .then_with(|| left.1.to_f64().total_cmp(&right.1.to_f64()))
                    .then_with(|| left.0.cmp(&right.0))
            });
            peaks.truncate(k);
        }

        peaks.sort_unstable_by(|left, right| {
            left.1
                .to_f64()
                .total_cmp(&right.1.to_f64())
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut spectrum = Self::with_capacity(self.precursor_mz().to_f64(), peaks.len())?;
        spectrum.add_peaks(peaks.into_iter().map(|(_, mz, intensity)| (mz, intensity)))?;
        Ok(spectrum)
    }

    /// Returns a new spectrum whose intensities are rescaled so the maximum
    /// intensity equals `1.0` (base-peak / L∞ normalization).
    ///
    /// The rescaled intensities are computed in `f64` and converted back to
    /// [`Spectrum::Precision`]. Peaks whose normalized intensity rounds to a
    /// non-positive value at the target precision are dropped, mirroring the
    /// underflow handling used elsewhere in the crate (e.g. `MsEntropyCleanSpectrum`).
    ///
    /// An empty spectrum and a spectrum whose maximum intensity is not
    /// positive both yield an empty result with the original `precursor_mz`.
    ///
    /// # Errors
    ///
    /// Returns [`SpectrumMut::MutationError`] if constructing the new spectrum or
    /// adding one of the rescaled peaks fails.
    ///
    /// # Example
    ///
    /// ```
    /// use mass_spectrometry::prelude::*;
    ///
    /// let mut spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(250.0, 3).unwrap();
    /// spectrum
    ///     .add_peaks([(50.0, 2.0), (75.0, 5.0), (100.0, 4.0)])
    ///     .unwrap();
    ///
    /// let normalized: GenericSpectrum = spectrum.intensity_normalized().unwrap();
    /// let peaks: Vec<(f64, f64)> = normalized.peaks().collect();
    /// assert_eq!(peaks, vec![(50.0, 0.4), (75.0, 1.0), (100.0, 0.8)]);
    /// ```
    fn intensity_normalized(&self) -> Result<Self, Self::MutationError> {
        let max_intensity = self
            .intensities()
            .map(SpectrumFloat::to_f64)
            .fold(0.0_f64, f64::max);

        let mut spectrum = Self::with_capacity(self.precursor_mz().to_f64(), self.len())?;

        if !(max_intensity > 0.0 && max_intensity.is_finite()) {
            return Ok(spectrum);
        }

        for (mz, intensity) in self.peaks() {
            let rescaled = Self::Precision::from_f64_lossy(intensity.to_f64() / max_intensity);
            if rescaled.to_f64() > 0.0 {
                spectrum.add_peak(mz, rescaled)?;
            }
        }

        Ok(spectrum)
    }

    /// Generate a random spectrum from a parameterized configuration.
    ///
    /// Generation is deterministic for a fixed `seed` and `config`.
    ///
    /// The generated peaks are strictly sorted by m/z and satisfy
    /// `mz[i + 1] - mz[i] >= min_peak_gap`.
    ///
    /// This default implementation is intended for benchmarks and synthetic
    /// tests where reproducible random spectra are useful.
    fn random(
        config: RandomSpectrumConfig,
        seed: u64,
    ) -> Result<Self, RandomSpectrumGenerationError<Self::MutationError>> {
        let mz_min = ensure_finite(config.mz_min, "mz_min")?;
        let mz_max = ensure_finite(config.mz_max, "mz_max")?;
        let min_peak_gap = ensure_finite(config.min_peak_gap, "min_peak_gap")?;
        let intensity_min = ensure_finite(config.intensity_min, "intensity_min")?;
        let intensity_max = ensure_finite(config.intensity_max, "intensity_max")?;

        if mz_max < mz_min {
            return Err(RandomSpectrumGenerationError::InvalidConfig(
                "mz_max must be >= mz_min",
            ));
        }
        if intensity_max < intensity_min {
            return Err(RandomSpectrumGenerationError::InvalidConfig(
                "intensity_max must be >= intensity_min",
            ));
        }
        if min_peak_gap <= 0.0 {
            return Err(RandomSpectrumGenerationError::InvalidConfig(
                "min_peak_gap must be > 0",
            ));
        }

        if config.n_peaks == 0 {
            return Self::with_capacity(config.precursor_mz, 0)
                .map_err(RandomSpectrumGenerationError::Mutation);
        }

        let mz_span = mz_max - mz_min;
        let required_span = min_peak_gap * ((config.n_peaks - 1) as f64);
        if required_span > mz_span {
            return Err(RandomSpectrumGenerationError::InvalidConfig(
                "n_peaks and min_peak_gap exceed [mz_min, mz_max] span",
            ));
        }

        let mut spectrum = Self::with_capacity(config.precursor_mz, config.n_peaks)
            .map_err(RandomSpectrumGenerationError::Mutation)?;

        let mut state = nonzero_seed(seed);
        let free_span = mz_span - required_span;
        let mut offsets = Vec::with_capacity(config.n_peaks);
        for _ in 0..config.n_peaks {
            offsets.push(next_unit_f64(&mut state) * free_span);
        }
        offsets.sort_unstable_by(f64::total_cmp);

        let intensity_span = intensity_max - intensity_min;
        for (i, offset) in offsets.into_iter().enumerate() {
            let mz = mz_min + ((i as f64) * min_peak_gap) + offset;
            let intensity = if intensity_span == 0.0 {
                intensity_min
            } else {
                intensity_min + (next_unit_f64(&mut state) * intensity_span)
            };
            let mz = Self::Precision::from_f64_lossy(mz);
            let intensity = Self::Precision::from_f64_lossy(intensity);

            spectrum
                .add_peak(mz, intensity)
                .map_err(RandomSpectrumGenerationError::Mutation)?;
        }

        Ok(spectrum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::GenericSpectrum;

    fn base_config() -> RandomSpectrumConfig {
        RandomSpectrumConfig {
            precursor_mz: 250.0,
            n_peaks: 4,
            mz_min: 100.0,
            mz_max: 130.0,
            min_peak_gap: 1.0,
            intensity_min: 1.0,
            intensity_max: 3.0,
        }
    }

    #[test]
    fn nonzero_seed_rewrites_zero_seed() {
        assert_ne!(nonzero_seed(0), 0);
        assert_eq!(nonzero_seed(7), 7);
    }

    #[test]
    fn random_rejects_invalid_config_ranges() {
        let error = GenericSpectrum::<f64>::random(
            RandomSpectrumConfig {
                mz_max: 99.0,
                ..base_config()
            },
            1,
        )
        .expect_err("mz range should be validated");
        assert!(matches!(
            error,
            RandomSpectrumGenerationError::InvalidConfig("mz_max must be >= mz_min")
        ));

        let error = GenericSpectrum::<f64>::random(
            RandomSpectrumConfig {
                intensity_max: 0.5,
                ..base_config()
            },
            1,
        )
        .expect_err("intensity range should be validated");
        assert!(matches!(
            error,
            RandomSpectrumGenerationError::InvalidConfig("intensity_max must be >= intensity_min")
        ));

        let error = GenericSpectrum::<f64>::random(
            RandomSpectrumConfig {
                min_peak_gap: 0.0,
                ..base_config()
            },
            1,
        )
        .expect_err("peak gap should be validated");
        assert!(matches!(
            error,
            RandomSpectrumGenerationError::InvalidConfig("min_peak_gap must be > 0")
        ));

        let error = GenericSpectrum::<f64>::random(
            RandomSpectrumConfig {
                mz_max: 102.0,
                min_peak_gap: 2.0,
                ..base_config()
            },
            1,
        )
        .expect_err("required span should be validated");
        assert!(matches!(
            error,
            RandomSpectrumGenerationError::InvalidConfig(
                "n_peaks and min_peak_gap exceed [mz_min, mz_max] span"
            )
        ));
    }

    #[test]
    fn intensity_normalized_rescales_to_base_peak() {
        let mut spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(250.0, 3).unwrap();
        spectrum
            .add_peaks([(50.0, 2.0), (75.0, 5.0), (100.0, 4.0)])
            .unwrap();

        let normalized: GenericSpectrum = spectrum
            .intensity_normalized()
            .expect("base-peak normalization should succeed");

        let peaks: Vec<(f64, f64)> = normalized.peaks().collect();
        assert_eq!(peaks, alloc::vec![(50.0, 0.4), (75.0, 1.0), (100.0, 0.8)]);
        assert_eq!(normalized.precursor_mz(), 250.0);
    }

    #[test]
    fn intensity_normalized_empty_spectrum_returns_empty() {
        let spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(150.0, 0).unwrap();
        let normalized: GenericSpectrum = spectrum
            .intensity_normalized()
            .expect("empty spectrum should normalize trivially");
        assert!(normalized.is_empty());
        assert_eq!(normalized.precursor_mz(), 150.0);
    }

    #[test]
    fn intensity_normalized_is_idempotent_under_proportional_scaling() {
        let mut a: GenericSpectrum = GenericSpectrum::try_with_capacity(200.0, 2).unwrap();
        a.add_peaks([(10.0, 1.0), (20.0, 4.0)]).unwrap();
        let mut b: GenericSpectrum = GenericSpectrum::try_with_capacity(200.0, 2).unwrap();
        b.add_peaks([(10.0, 100.0), (20.0, 400.0)]).unwrap();

        let normalized_a: GenericSpectrum = a.intensity_normalized().unwrap();
        let normalized_b: GenericSpectrum = b.intensity_normalized().unwrap();

        let peaks_a: Vec<(f64, f64)> = normalized_a.peaks().collect();
        let peaks_b: Vec<(f64, f64)> = normalized_b.peaks().collect();
        assert_eq!(peaks_a, peaks_b);
    }

    #[test]
    fn random_zero_peaks_and_constant_intensity_paths_work() {
        let empty: GenericSpectrum = GenericSpectrum::random(
            RandomSpectrumConfig {
                n_peaks: 0,
                ..base_config()
            },
            0,
        )
        .expect("zero-peak spectrum should build");
        assert!(empty.is_empty());

        let constant: GenericSpectrum = GenericSpectrum::random(
            RandomSpectrumConfig {
                intensity_max: 2.5,
                intensity_min: 2.5,
                ..base_config()
            },
            0,
        )
        .expect("constant-intensity spectrum should build");
        assert!(constant.intensities().all(|intensity| intensity == 2.5));
    }
}
