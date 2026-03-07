//! Shared building blocks for the entropy similarity family.
//!
//! All items are `pub(crate)` — individual variant modules re-export only their
//! public struct via the parent `structs` module.
//!
//! All entropy arithmetic is generic over `F: Float`, so callers choose the
//! computation precision (f32 or f64) while m/z matching always uses f64.

use alloc::vec::Vec;

use num_traits::{Float, ToPrimitive};

use super::cosine_common::to_f64_checked_for_computation;
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

// ---------------------------------------------------------------------------
// Shannon entropy helpers
// ---------------------------------------------------------------------------

/// Shannon entropy `H = -sum(p * ln(p))`, skipping zero entries.
#[inline]
pub(crate) fn shannon_entropy<F: Float>(probs: &[F]) -> F {
    probs
        .iter()
        .filter(|&&p| p > F::zero())
        .fold(F::zero(), |acc, &p| acc - p * p.ln())
}

/// Apply entropy-based weighting: if H < 3, raise each intensity to
/// `0.25 * (1 + H)` and re-normalize to sum to 1.
pub(crate) fn apply_entropy_weight<F: Float>(intensities: &mut [F]) {
    let entropy = shannon_entropy(intensities);
    let three = F::from(3.0).expect("3.0 is representable in any Float");
    if entropy < three {
        let quarter = F::from(0.25).expect("0.25 is representable in any Float");
        let power = quarter * (F::one() + entropy);
        for v in intensities.iter_mut() {
            *v = v.powf(power);
        }
        let sum = intensities.iter().fold(F::zero(), |acc, &v| acc + v);
        if sum > F::zero() {
            for v in intensities.iter_mut() {
                *v = *v / sum;
            }
        }
    }
}

/// Entropy contribution for a single pair `(a, b)`:
/// `(a + b) * log2(a + b) - a * log2(a) - b * log2(b)`
///
/// Handles zeros: `0 * log2(0) = 0`.
#[inline]
pub(crate) fn entropy_pair<F: Float>(a: F, b: F) -> F {
    let mut result = F::zero();
    let ab = a + b;
    if ab > F::zero() {
        result = result + ab * ab.log2();
    }
    if a > F::zero() {
        result = result - a * a.log2();
    }
    if b > F::zero() {
        result = result - b * b.log2();
    }
    result
}

// ---------------------------------------------------------------------------
// Config macro
// ---------------------------------------------------------------------------

macro_rules! impl_entropy_config_api {
    ($type_name:ident, $doc:expr) => {
        impl<EXP: geometric_traits::prelude::Number, MZ: geometric_traits::prelude::Number>
            $type_name<EXP, MZ>
        {
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

            /// Returns the power to which the mass/charge ratio is raised.
            #[inline]
            pub fn mz_power(&self) -> EXP {
                self.mz_power
            }

            /// Returns the power to which the intensity is raised.
            #[inline]
            pub fn intensity_power(&self) -> EXP {
                self.intensity_power
            }
        }

        impl<EXP, MZ> $type_name<EXP, MZ>
        where
            EXP: geometric_traits::prelude::Number + num_traits::ToPrimitive,
            MZ: geometric_traits::prelude::Number + num_traits::ToPrimitive + PartialOrd,
        {
            #[doc = concat!("Creates a new `", stringify!($type_name), "` with the given powers, m/z tolerance and weighting mode.")]
            ///
            /// # Errors
            ///
            /// Returns [`crate::structs::SimilarityConfigError`] if any numeric
            /// parameter is not
            /// finite/representable or if `mz_tolerance` is negative.
            #[inline]
            pub fn new(
                mz_power: EXP,
                intensity_power: EXP,
                mz_tolerance: MZ,
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

        }

        /// Convenience constructors with default powers (`mz_power=0`,
        /// `intensity_power=1`) where `EXP = MZ`.
        impl<MZ> $type_name<MZ, MZ>
        where
            MZ: geometric_traits::prelude::Number
                + num_traits::ToPrimitive
                + PartialOrd
                + num_traits::Zero
                + num_traits::One,
        {
            #[doc = concat!("Creates a new weighted `", stringify!($type_name), "` with default powers (mz_power=0, intensity_power=1).")]
            #[inline]
            pub fn weighted(
                mz_tolerance: MZ,
            ) -> Result<Self, super::similarity_errors::SimilarityConfigError> {
                Self::new(MZ::zero(), MZ::one(), mz_tolerance, true)
            }

            #[doc = concat!("Creates a new unweighted `", stringify!($type_name), "` with default powers (mz_power=0, intensity_power=1).")]
            #[inline]
            pub fn unweighted(
                mz_tolerance: MZ,
            ) -> Result<Self, super::similarity_errors::SimilarityConfigError> {
                Self::new(MZ::zero(), MZ::one(), mz_tolerance, false)
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
        impl<EXP, S1, S2> crate::traits::ScalarSpectralSimilarity<S1, S2>
            for $type_name<EXP, S1::Mz>
        where
            EXP: geometric_traits::prelude::Number + num_traits::ToPrimitive,
            S1::Mz: num_traits::Float + geometric_traits::prelude::Number,
            S1: crate::traits::Spectrum<Intensity = <S1 as crate::traits::Spectrum>::Mz>,
            S2: crate::traits::Spectrum<Intensity = S1::Mz, Mz = S1::Mz>,
        {
        }
    };
}

pub(crate) use impl_entropy_spectral_similarity;

// ---------------------------------------------------------------------------
// Peak preparation
// ---------------------------------------------------------------------------

/// Intermediate representation: m/z stays f64 for matching; intensities
/// use the caller-chosen computation precision `F`.
pub(crate) struct PreparedEntropyPeaks<F> {
    pub(crate) mz: Vec<f64>,
    pub(crate) int: Vec<F>,
}

/// Collect peaks, compute `mz^p * intensity^q` products in `F` precision,
/// normalize to sum=1, optionally apply entropy weighting.  Returns empty
/// vecs when the product sum is zero.
///
/// m/z values are always stored as f64 (for matching); intensity products
/// are computed and stored in `F`.
pub(crate) fn prepare_entropy_peaks<F, S>(
    spectrum: &S,
    weighted: bool,
    mz_power_f64: f64,
    intensity_power_f64: f64,
) -> Result<PreparedEntropyPeaks<F>, SimilarityComputationError>
where
    F: Float,
    S: Spectrum,
    S::Mz: ToPrimitive,
    S::Intensity: ToPrimitive,
{
    let mut mz = Vec::with_capacity(spectrum.len());
    let mut int = Vec::with_capacity(spectrum.len());

    for (m, i) in spectrum.peaks() {
        let mz_f64 = to_f64_checked_for_computation(m, "mz")?;
        let int_f64 = to_f64_checked_for_computation(i, "intensity")?;
        mz.push(mz_f64);

        let product_f64 = if mz_power_f64 == 0.0 {
            int_f64.powf(intensity_power_f64)
        } else {
            mz_f64.powf(mz_power_f64) * int_f64.powf(intensity_power_f64)
        };
        let product = F::from(product_f64).ok_or(
            SimilarityComputationError::ValueNotRepresentable("peak_product"),
        )?;
        int.push(product);
    }

    let sum = int.iter().fold(F::zero(), |acc, &v| acc + v);
    if !sum.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue("product_sum"));
    }

    if sum == F::zero() {
        return Ok(PreparedEntropyPeaks {
            mz: Vec::new(),
            int: Vec::new(),
        });
    }

    for v in int.iter_mut() {
        *v = *v / sum;
    }

    if weighted {
        apply_entropy_weight(&mut int);
        let wsum = int.iter().fold(F::zero(), |acc, &v| acc + v);
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
pub(crate) fn finalize_entropy_score<F: Float>(
    raw_score: F,
    n_matches: usize,
) -> Result<(F, usize), SimilarityComputationError> {
    let two = F::from(2.0).expect("2.0 is representable in any Float");
    let similarity = (raw_score / two).max(F::zero()).min(F::one());
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
pub(crate) fn entropy_score_pairs<F: Float>(
    pairs: &[(usize, usize)],
    left_int: &[F],
    right_int: &[F],
) -> Result<(F, usize), SimilarityComputationError> {
    let mut score = F::zero();
    let mut n_matches = 0usize;
    for &(i, j) in pairs {
        let pair = entropy_pair(left_int[i], right_int[j]);
        if !pair.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue(
                "entropy_pair_score",
            ));
        }
        score = score + pair;
        n_matches += 1;
    }
    Ok((score, n_matches))
}
