//! Submodule providing a trait for a mutable Spectrum.

use alloc::vec::Vec;

use num_traits::{Float, NumCast, ToPrimitive};

use super::Spectrum;
use crate::numeric_validation::{NumericValidationError, checked_to_f64};

/// Parameters for generating a random spectrum with [`SpectrumAlloc::random`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomSpectrumConfig<Mz, Intensity> {
    /// Precursor m/z for the generated spectrum.
    pub precursor_mz: Mz,
    /// Number of peaks to generate.
    pub n_peaks: usize,
    /// Minimum generated peak m/z.
    pub mz_min: Mz,
    /// Maximum generated peak m/z.
    pub mz_max: Mz,
    /// Minimum spacing between consecutive m/z values.
    pub min_peak_gap: Mz,
    /// Minimum generated intensity.
    pub intensity_min: Intensity,
    /// Maximum generated intensity.
    pub intensity_max: Intensity,
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
    /// Input value cannot be represented as `f64`.
    #[error("value is not representable as f64: {0}")]
    ValueNotRepresentable(&'static str),
    /// Input value is non-finite.
    #[error("value must be finite: {0}")]
    NonFiniteValue(&'static str),
    /// Generated numeric value cannot be represented in target type.
    #[error("generated value is not representable in target type: {0}")]
    GeneratedValueNotRepresentable(&'static str),
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
fn to_f64_checked<E, T>(
    value: T,
    name: &'static str,
) -> Result<f64, RandomSpectrumGenerationError<E>>
where
    E: core::error::Error,
    T: ToPrimitive,
{
    checked_to_f64(value, name).map_err(|error| match error {
        NumericValidationError::NonRepresentable(name) => {
            RandomSpectrumGenerationError::ValueNotRepresentable(name)
        }
        NumericValidationError::NonFinite(name) => {
            RandomSpectrumGenerationError::NonFiniteValue(name)
        }
    })
}

/// Trait for a mutable Spectrum.
pub trait SpectrumMut: Spectrum {
    /// The type of error that can occur when mutating the Spectrum.
    type MutationError: core::error::Error;

    /// Add a peak to the Spectrum.
    ///
    /// Implementations are expected to reject non-finite values, enforce
    /// strictly increasing `mz` ordering (rejecting duplicates), and reject
    /// negative intensity values.
    fn add_peak(
        &mut self,
        mz: Self::Mz,
        intensity: Self::Intensity,
    ) -> Result<(), Self::MutationError>;
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
    fn with_capacity(precursor_mz: Self::Mz, capacity: usize) -> Result<Self, Self::MutationError>;

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
        config: RandomSpectrumConfig<Self::Mz, Self::Intensity>,
        seed: u64,
    ) -> Result<Self, RandomSpectrumGenerationError<Self::MutationError>>
    where
        Self::Mz: Float + NumCast + ToPrimitive,
        Self::Intensity: Float + NumCast + ToPrimitive,
    {
        let mz_min = to_f64_checked::<Self::MutationError, _>(config.mz_min, "mz_min")?;
        let mz_max = to_f64_checked::<Self::MutationError, _>(config.mz_max, "mz_max")?;
        let min_peak_gap =
            to_f64_checked::<Self::MutationError, _>(config.min_peak_gap, "min_peak_gap")?;
        let intensity_min =
            to_f64_checked::<Self::MutationError, _>(config.intensity_min, "intensity_min")?;
        let intensity_max =
            to_f64_checked::<Self::MutationError, _>(config.intensity_max, "intensity_max")?;

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
            let mz_f64 = mz_min + ((i as f64) * min_peak_gap) + offset;
            let intensity_f64 = if intensity_span == 0.0 {
                intensity_min
            } else {
                intensity_min + (next_unit_f64(&mut state) * intensity_span)
            };

            let mz = num_traits::cast::<f64, Self::Mz>(mz_f64)
                .ok_or(RandomSpectrumGenerationError::GeneratedValueNotRepresentable("mz"))?;
            let intensity = num_traits::cast::<f64, Self::Intensity>(intensity_f64).ok_or(
                RandomSpectrumGenerationError::GeneratedValueNotRepresentable("intensity"),
            )?;

            spectrum
                .add_peak(mz, intensity)
                .map_err(RandomSpectrumGenerationError::Mutation)?;
        }

        Ok(spectrum)
    }
}
