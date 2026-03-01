//! Implementation of spectral entropy similarity for mass spectra.
//!
//! Based on Li et al., "Spectral entropy outperforms MS/MS dot product
//! similarity for small-molecule compound identification",
//! Nature Methods 18, 1524–1531 (2021).

use alloc::vec::Vec;

use geometric_traits::prelude::{Number, ScalarSimilarity};
use num_traits::{Float, NumCast, ToPrimitive, Zero};

use super::cosine_common::validate_non_negative_tolerance;
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{ScalarSpectralSimilarity, Spectrum};

/// Spectral entropy similarity (Li et al., Nature Methods 2021).
///
/// Uses Jensen-Shannon divergence on greedy-matched peaks. Two variants
/// are available:
///
/// - **Weighted** (default): intensities are raised to a data-driven power
///   based on the spectral Shannon entropy, then re-normalized.
/// - **Unweighted**: raw intensities are normalized to sum to 1 directly.
pub struct EntropySimilarity<MZ> {
    mz_tolerance: MZ,
    weighted: bool,
}

impl<MZ: Number> EntropySimilarity<MZ> {
    /// Returns the m/z tolerance.
    #[inline]
    pub fn mz_tolerance(&self) -> MZ {
        self.mz_tolerance
    }

    /// Returns whether entropy-based intensity weighting is enabled.
    #[inline]
    pub fn is_weighted(&self) -> bool {
        self.weighted
    }
}

impl<MZ> EntropySimilarity<MZ>
where
    MZ: Number + ToPrimitive + PartialOrd,
{
    /// Creates a new `EntropySimilarity` with the given m/z tolerance and
    /// weighting mode.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `mz_tolerance` is not finite,
    /// not representable as `f64`, or negative.
    #[inline]
    pub fn new(mz_tolerance: MZ, weighted: bool) -> Result<Self, SimilarityConfigError> {
        validate_non_negative_tolerance(mz_tolerance)?;
        Ok(Self {
            mz_tolerance,
            weighted,
        })
    }

    /// Creates a new weighted `EntropySimilarity`.
    #[inline]
    pub fn weighted(mz_tolerance: MZ) -> Result<Self, SimilarityConfigError> {
        Self::new(mz_tolerance, true)
    }

    /// Creates a new unweighted `EntropySimilarity`.
    #[inline]
    pub fn unweighted(mz_tolerance: MZ) -> Result<Self, SimilarityConfigError> {
        Self::new(mz_tolerance, false)
    }
}

/// Shannon entropy `H = -sum(p * ln(p))`, skipping zero entries.
fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Apply entropy-based weighting: if H < 3, raise each intensity to
/// `0.25 * (1 + H)` and re-normalize to sum to 1.
fn apply_entropy_weight(intensities: &mut [f64]) {
    let entropy = shannon_entropy(intensities);
    if entropy < 3.0 {
        let power = 0.25 * (1.0 + entropy);
        for v in intensities.iter_mut() {
            *v = v.powf(power);
        }
        let sum: f64 = intensities.iter().sum();
        if sum > 0.0 {
            for v in intensities.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Entropy contribution for a single pair `(a, b)`:
/// `(a + b) * log2(a + b) - a * log2(a) - b * log2(b)`
///
/// Handles zeros: `0 * log2(0) = 0`.
#[inline]
fn entropy_pair(a: f64, b: f64) -> f64 {
    let mut result = 0.0;
    let ab = a + b;
    if ab > 0.0 {
        result += ab * ab.log2();
    }
    if a > 0.0 {
        result -= a * a.log2();
    }
    if b > 0.0 {
        result -= b * b.log2();
    }
    result
}

impl<S1, S2> ScalarSimilarity<S1, S2> for EntropySimilarity<S1::Mz>
where
    S1::Mz: Float + Number,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
    type Similarity = Result<(S1::Mz, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        // Collect peaks into f64 arrays for precision.
        let mut left_mz: Vec<f64> = Vec::with_capacity(left.len());
        let mut left_int: Vec<f64> = Vec::with_capacity(left.len());
        for (mz, intensity) in left.peaks() {
            let Some(mz_f64) = mz.to_f64() else {
                return Err(SimilarityComputationError::ValueNotRepresentable("left_mz"));
            };
            let Some(int_f64) = intensity.to_f64() else {
                return Err(SimilarityComputationError::ValueNotRepresentable(
                    "left_intensity",
                ));
            };
            left_mz.push(mz_f64);
            left_int.push(int_f64);
        }

        let mut right_mz: Vec<f64> = Vec::with_capacity(right.len());
        let mut right_int: Vec<f64> = Vec::with_capacity(right.len());
        for (mz, intensity) in right.peaks() {
            let Some(mz_f64) = mz.to_f64() else {
                return Err(SimilarityComputationError::ValueNotRepresentable(
                    "right_mz",
                ));
            };
            let Some(int_f64) = intensity.to_f64() else {
                return Err(SimilarityComputationError::ValueNotRepresentable(
                    "right_intensity",
                ));
            };
            right_mz.push(mz_f64);
            right_int.push(int_f64);
        }

        // Normalize intensities to sum to 1.
        let left_sum: f64 = left_int.iter().sum();
        let right_sum: f64 = right_int.iter().sum();

        if left_sum == 0.0 || right_sum == 0.0 {
            return Ok((S1::Mz::zero(), 0));
        }

        for v in left_int.iter_mut() {
            *v /= left_sum;
        }
        for v in right_int.iter_mut() {
            *v /= right_sum;
        }

        // Apply entropy-based weighting if enabled.
        if self.weighted {
            apply_entropy_weight(&mut left_int);
            apply_entropy_weight(&mut right_int);
        }

        // Two-pointer greedy matching within m/z tolerance.
        let Some(tolerance) = self.mz_tolerance.to_f64() else {
            return Err(SimilarityComputationError::ValueNotRepresentable(
                "mz_tolerance",
            ));
        };
        let mut i = 0usize;
        let mut j = 0usize;
        let mut score = 0.0f64;
        let mut n_matches: usize = 0;

        while i < left_mz.len() && j < right_mz.len() {
            let diff = left_mz[i] - right_mz[j];
            if diff.abs() <= tolerance {
                score += entropy_pair(left_int[i], right_int[j]);
                n_matches += 1;
                i += 1;
                j += 1;
            } else if diff < 0.0 {
                i += 1;
            } else {
                j += 1;
            }
        }

        // Divide by 2 to normalize JSD to [0, 1].
        let similarity = (score / 2.0).clamp(0.0, 1.0);

        let sim: S1::Mz = NumCast::from(similarity).ok_or(
            SimilarityComputationError::ValueNotRepresentable("similarity_score"),
        )?;
        Ok((sim, n_matches))
    }
}

impl<S1, S2> ScalarSpectralSimilarity<S1, S2> for EntropySimilarity<S1::Mz>
where
    S1::Mz: Float + Number,
    S1: Spectrum<Intensity = <S1 as Spectrum>::Mz>,
    S2: Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
{
}
