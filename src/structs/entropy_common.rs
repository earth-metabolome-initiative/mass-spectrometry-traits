//! Shared building blocks for the entropy similarity family.
//!
//! All items are `pub(crate)` — individual variant modules re-export only their
//! public struct via the parent `structs` module.

use alloc::vec::Vec;

use super::cosine_common::ensure_finite;
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

// ---------------------------------------------------------------------------
// Shannon entropy helpers
// ---------------------------------------------------------------------------

/// Shannon entropy `H = -sum(p * ln(p))`, skipping zero entries.
#[inline]
pub(crate) fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .fold(0.0, |acc, &p| acc - p * p.ln())
}

/// Apply entropy-based weighting: if H < 3, raise each intensity to
/// `0.25 * (1 + H)` and re-normalize to sum to 1.
pub(crate) fn apply_entropy_weight(intensities: &mut [f64]) {
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
pub(crate) fn entropy_pair(a: f64, b: f64) -> f64 {
    let mut result = 0.0_f64;
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

// ---------------------------------------------------------------------------
// Config macro
// ---------------------------------------------------------------------------

macro_rules! impl_entropy_config_api {
    ($type_name:ident, $doc:expr) => {
        impl $type_name {
            /// Returns the m/z tolerance.
            #[inline]
            pub fn mz_tolerance(&self) -> f64 {
                self.mz_tolerance
            }

            /// Returns whether entropy-based intensity weighting is enabled.
            #[inline]
            pub fn is_weighted(&self) -> bool {
                self.weighted
            }

            /// Returns the power to which the mass/charge ratio is raised.
            #[inline]
            pub fn mz_power(&self) -> f64 {
                self.mz_power
            }

            /// Returns the power to which the intensity is raised.
            #[inline]
            pub fn intensity_power(&self) -> f64 {
                self.intensity_power
            }

            #[doc = concat!("Creates a new `", stringify!($type_name), "` with the given powers, m/z tolerance and weighting mode.")]
            ///
            /// # Errors
            ///
            /// Returns [`crate::structs::SimilarityConfigError`] if any numeric
            /// parameter is not finite or if `mz_tolerance` is negative.
            #[inline]
            pub fn new(
                mz_power: f64,
                intensity_power: f64,
                mz_tolerance: f64,
                weighted: bool,
            ) -> Result<Self, super::similarity_errors::SimilarityConfigError> {
                super::cosine_common::validate_numeric_parameter(mz_power, "mz_power")?;
                super::cosine_common::validate_numeric_parameter(
                    intensity_power,
                    "intensity_power",
                )?;
                super::cosine_common::validate_non_negative_tolerance(mz_tolerance)?;
                Ok(Self {
                    mz_power,
                    intensity_power,
                    mz_tolerance,
                    weighted,
                })
            }

            #[doc = concat!("Creates a new weighted `", stringify!($type_name), "` with default powers (mz_power=0, intensity_power=1).")]
            #[inline]
            pub fn weighted(
                mz_tolerance: f64,
            ) -> Result<Self, super::similarity_errors::SimilarityConfigError> {
                Self::new(0.0, 1.0, mz_tolerance, true)
            }

            #[doc = concat!("Creates a new unweighted `", stringify!($type_name), "` with default powers (mz_power=0, intensity_power=1).")]
            #[inline]
            pub fn unweighted(
                mz_tolerance: f64,
            ) -> Result<Self, super::similarity_errors::SimilarityConfigError> {
                Self::new(0.0, 1.0, mz_tolerance, false)
            }
        }
    };
}

pub(crate) use impl_entropy_config_api;

// ---------------------------------------------------------------------------
// ScalarSpectralSimilarity blanket macro
// ---------------------------------------------------------------------------

macro_rules! impl_entropy_spectral_similarity {
    ($type_name:ident) => {
        impl<S1, S2> crate::traits::ScalarSpectralSimilarity<S1, S2> for $type_name
        where
            S1: crate::traits::Spectrum,
            S2: crate::traits::Spectrum,
        {
        }
    };
}

pub(crate) use impl_entropy_spectral_similarity;

// ---------------------------------------------------------------------------
// Peak preparation
// ---------------------------------------------------------------------------

/// Intermediate representation: m/z and intensities both f64.
pub(crate) struct PreparedEntropyPeaks {
    pub(crate) mz: Vec<f64>,
    pub(crate) int: Vec<f64>,
}

/// Collect peaks, compute `mz^p * intensity^q` products, normalize to sum=1,
/// optionally apply entropy weighting. Returns empty vecs when the product
/// sum is zero.
pub(crate) fn prepare_entropy_peaks<S: Spectrum>(
    spectrum: &S,
    weighted: bool,
    mz_power_f64: f64,
    intensity_power_f64: f64,
) -> Result<PreparedEntropyPeaks, SimilarityComputationError> {
    let mut mz = Vec::with_capacity(spectrum.len());
    let mut int = Vec::with_capacity(spectrum.len());

    for (m, i) in spectrum.peaks() {
        ensure_finite(i, "intensity")?;
        mz.push(m);

        let product = if mz_power_f64 == 0.0 {
            i.powf(intensity_power_f64)
        } else {
            m.powf(mz_power_f64) * i.powf(intensity_power_f64)
        };
        ensure_finite(product, "peak_product")?;
        int.push(product);
    }

    let sum: f64 = int.iter().sum();
    if !sum.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue("product_sum"));
    }

    if sum == 0.0 {
        return Ok(PreparedEntropyPeaks {
            mz: Vec::new(),
            int: Vec::new(),
        });
    }

    for v in int.iter_mut() {
        *v /= sum;
    }

    if weighted {
        apply_entropy_weight(&mut int);
        let wsum: f64 = int.iter().sum();
        if !wsum.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue(
                "weighted_product_sum",
            ));
        }
    }

    Ok(PreparedEntropyPeaks { mz, int })
}

// ---------------------------------------------------------------------------
// Finalization
// ---------------------------------------------------------------------------

/// Divide by 2, clamp to [0,1], validate.
#[inline]
pub(crate) fn finalize_entropy_score(
    raw_score: f64,
    n_matches: usize,
) -> Result<(f64, usize), SimilarityComputationError> {
    let similarity = (raw_score / 2.0).clamp(0.0, 1.0);
    if !similarity.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue(
            "similarity_score",
        ));
    }
    Ok((similarity, n_matches))
}

// ---------------------------------------------------------------------------
// Pair scoring (linear + inline greedy)
// ---------------------------------------------------------------------------

/// Sum `entropy_pair(left_int[i], right_int[j])` for each index pair.
pub(crate) fn entropy_score_pairs(
    pairs: &[(usize, usize)],
    left_int: &[f64],
    right_int: &[f64],
) -> Result<(f64, usize), SimilarityComputationError> {
    let mut score = 0.0_f64;
    let mut n_matches = 0usize;
    for &(i, j) in pairs {
        let pair = entropy_pair(left_int[i], right_int[j]);
        if !pair.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue(
                "entropy_pair_score",
            ));
        }
        score += pair;
        n_matches += 1;
    }
    Ok((score, n_matches))
}
