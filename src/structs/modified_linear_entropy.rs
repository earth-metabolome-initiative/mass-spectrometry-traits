//! Implementation of modified linear-time spectral entropy similarity.
//!
//! Extends [`super::LinearEntropy`] by also matching fragment peaks shifted by
//! the precursor mass difference when that shift exceeds the configured
//! tolerance. Direct and shifted match candidates are merged and resolved via
//! optimal DP-based assignment on the conflict graph's path components.

use geometric_traits::prelude::ScalarSimilarity;

use super::cosine_common::{
    ensure_finite, optimal_modified_linear_matches, validate_well_separated,
};
use super::entropy_common::{
    entropy_pair, entropy_score_pairs, finalize_entropy_score, impl_entropy_config_api,
    impl_entropy_spectral_similarity, prepare_entropy_peaks,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Modified linear-time spectral entropy similarity.
///
/// Combines direct and precursor-shifted peak matches from two linear sweeps
/// when `|precursor_delta| > mz_tolerance`, then resolves conflicts via
/// optimal DP-based assignment. Requires the same strict well-separated
/// precondition as [`super::LinearEntropy`]
/// (consecutive peaks > `2 * mz_tolerance`).
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct ModifiedLinearEntropy {
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    weighted: bool,
}

impl_entropy_config_api!(ModifiedLinearEntropy, "modified linear entropy similarity");
impl_entropy_spectral_similarity!(ModifiedLinearEntropy);

impl<S1, S2> ScalarSimilarity<S1, S2> for ModifiedLinearEntropy
where
    S1: Spectrum,
    S2: Spectrum,
{
    type Similarity = Result<(f64, usize), SimilarityComputationError>;

    fn similarity(&self, left: &S1, right: &S2) -> Self::Similarity {
        let left_peaks =
            prepare_entropy_peaks(left, self.weighted, self.mz_power, self.intensity_power)?;
        let right_peaks =
            prepare_entropy_peaks(right, self.weighted, self.mz_power, self.intensity_power)?;

        if left_peaks.int.is_empty() || right_peaks.int.is_empty() {
            return Ok((0.0, 0));
        }

        let tolerance = ensure_finite(self.mz_tolerance, "mz_tolerance")?;

        validate_well_separated(&left_peaks.mz, tolerance, "left spectrum")?;
        validate_well_separated(&right_peaks.mz, tolerance, "right spectrum")?;

        let left_prec = ensure_finite(left.precursor_mz(), "left_precursor_mz")?;
        let right_prec = ensure_finite(right.precursor_mz(), "right_precursor_mz")?;

        let selected = optimal_modified_linear_matches(
            &left_peaks.mz,
            &right_peaks.mz,
            tolerance,
            left_prec,
            right_prec,
            |i, j| entropy_pair(left_peaks.int[i], right_peaks.int[j]),
        );

        let (raw_score, n_matches) =
            entropy_score_pairs(&selected, &left_peaks.int, &right_peaks.int)?;

        finalize_entropy_score(raw_score, n_matches)
    }
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
    fn zero_product_inputs_return_zero_similarity() {
        let scorer = ModifiedLinearEntropy::new(0.0, 1.0, 0.1, false).expect("config should build");
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
    fn modified_entropy_matches_on_shifted_precursor_pairs() {
        let scorer = ModifiedLinearEntropy::weighted(0.1).expect("config should build");
        let left = RawSpectrum {
            precursor_mz: 210.0,
            peaks: vec![(100.0, 3.0), (210.0, 2.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 200.0,
            peaks: vec![(100.0, 3.0), (200.0, 2.0)],
        };

        let (score, matches) = scorer
            .similarity(&left, &right)
            .expect("similarity should succeed");
        assert!(score.is_finite());
        assert!(matches > 0);
    }
}
