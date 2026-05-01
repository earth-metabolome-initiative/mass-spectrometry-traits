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
