//! Implementation of modified linear-time cosine similarity for mass spectra.
//!
//! Extends [`super::LinearCosine`] by also matching fragment peaks shifted by
//! the precursor mass difference when that shift exceeds the configured
//! tolerance. Direct and shifted match candidates are merged and resolved via
//! optimal DP-based assignment on the conflict graph's path components,
//! producing the same score as [`super::ModifiedHungarianCosine`] in linear
//! time for well-separated spectra (match counts may differ on near-zero
//! edges due to f64 tie-breaking).

use alloc::vec::Vec;

use geometric_traits::prelude::ScalarSimilarity;

use super::cosine_common::{
    CosineConfig, ensure_finite, finalize_similarity_score, impl_cosine_wrapper_config_api,
    optimal_modified_linear_matches, prepare_peak_products, validate_well_separated,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::{ScalarSpectralSimilarity, Spectrum, SpectrumFloat};

/// Modified linear-time cosine similarity for mass spectra.
///
/// Combines direct and precursor-shifted peak matches from two linear sweeps
/// when `|precursor_delta| > mz_tolerance`, then resolves conflicts via
/// optimal DP-based assignment on the conflict graph's path components.
/// For well-separated spectra this produces the same score as
/// [`super::ModifiedHungarianCosine`] in linear time (match counts may
/// differ on near-zero edges due to f64 tie-breaking).
/// Requires the same strict well-separated precondition as
/// [`super::LinearCosine`] (consecutive peaks > `2 * mz_tolerance`).
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct ModifiedLinearCosine {
    config: CosineConfig,
}

impl_cosine_wrapper_config_api!(
    ModifiedLinearCosine,
    "the modified linear cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl<S1, S2> ScalarSimilarity<S1, S2> for ModifiedLinearCosine
where
    S1: Spectrum,
    S2: Spectrum,
{
    type Similarity = Result<(f64, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks =
            prepare_peak_products(left, self.config.mz_power(), self.config.intensity_power())?;
        let right_peaks =
            prepare_peak_products(right, self.config.mz_power(), self.config.intensity_power())?;

        if left_peaks.max == 0.0 || right_peaks.max == 0.0 {
            return Ok((0.0, 0));
        }

        let tolerance = ensure_finite(self.config.mz_tolerance(), "mz_tolerance")?;

        // Collect mz as f64.
        let left_mz: Vec<f64> = left.mz().map(SpectrumFloat::to_f64).collect();
        let right_mz: Vec<f64> = right.mz().map(SpectrumFloat::to_f64).collect();

        validate_well_separated(&left_mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_mz, tolerance, "right spectrum")?;

        let left_prec = ensure_finite(left.precursor_mz().to_f64(), "left_precursor_mz")?;
        let right_prec = ensure_finite(right.precursor_mz().to_f64(), "right_precursor_mz")?;

        let max_left = left_peaks.max;
        let max_right = right_peaks.max;
        let selected = optimal_modified_linear_matches(
            &left_mz,
            &right_mz,
            tolerance,
            left_prec,
            right_prec,
            |i, j| {
                if (left_peaks.products[i] * right_peaks.products[j]) == 0.0 {
                    return 0.0;
                }
                let normalized =
                    (left_peaks.products[i] / max_left) * (right_peaks.products[j] / max_right);
                normalized.max(f64::EPSILON)
            },
        );

        let mut score_sum = 0.0_f64;
        let mut n_matches = 0usize;
        for (i, j) in selected {
            let product = left_peaks.products[i] * right_peaks.products[j];
            if product != 0.0 {
                score_sum += product;
                n_matches += 1;
            }
        }

        finalize_similarity_score(score_sum, n_matches, left_peaks.norm, right_peaks.norm)
    }
}

impl<S1, S2> ScalarSpectralSimilarity<S1, S2> for ModifiedLinearCosine
where
    S1: Spectrum,
    S2: Spectrum,
{
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use geometric_traits::prelude::ScalarSimilarity;

    use super::*;

    #[derive(Clone)]
    struct RawSpectrum {
        precursor_mz: f64,
        peaks: Vec<(f64, f64)>,
    }

    impl Spectrum for RawSpectrum {
        type Precision = f64;

        type SortedIntensitiesIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedMzIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedPeaksIter<'a>
            = core::iter::Copied<core::slice::Iter<'a, (f64, f64)>>
        where
            Self: 'a;

        fn len(&self) -> usize {
            self.peaks.len()
        }

        fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
            self.peaks.iter().map(|peak| peak.1)
        }

        fn intensity_nth(&self, n: usize) -> f64 {
            self.peaks[n].1
        }

        fn mz(&self) -> Self::SortedMzIter<'_> {
            self.peaks.iter().map(|peak| peak.0)
        }

        fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
            self.peaks[index..].iter().map(|peak| peak.0)
        }

        fn mz_nth(&self, n: usize) -> f64 {
            self.peaks[n].0
        }

        fn peaks(&self) -> Self::SortedPeaksIter<'_> {
            self.peaks.iter().copied()
        }

        fn peak_nth(&self, n: usize) -> (f64, f64) {
            self.peaks[n]
        }

        fn precursor_mz(&self) -> f64 {
            self.precursor_mz
        }
    }

    #[test]
    fn accessors_and_zero_product_short_circuit_work() {
        let scorer = ModifiedLinearCosine::new(0.5, 2.0, 0.1).expect("config should build");
        assert_eq!(scorer.mz_power(), 0.5);
        assert_eq!(scorer.intensity_power(), 2.0);
        assert_eq!(scorer.mz_tolerance(), 0.1);

        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: vec![(50.0, 0.0)],
        };
        assert_eq!(
            scorer
                .similarity(&spectrum, &spectrum)
                .expect("zero-product similarity should succeed"),
            (0.0, 0)
        );
    }

    #[test]
    fn zero_product_candidates_are_ignored_inside_selected_matches() {
        let scorer = ModifiedLinearCosine::new(0.0, 1.0, 0.1).expect("config should build");
        let left = RawSpectrum {
            precursor_mz: 200.0,
            peaks: vec![(100.0, 0.0), (200.0, 2.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 200.0,
            peaks: vec![(100.0, 3.0), (200.0, 4.0)],
        };

        let (score, matches) = scorer
            .similarity(&left, &right)
            .expect("similarity should succeed");
        assert_eq!(matches, 1);
        assert!(score.is_finite() && score > 0.0);
    }
}
