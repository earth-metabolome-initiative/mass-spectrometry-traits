//! Implementation of linear-time spectral entropy similarity.
//!
//! When spectra are *well-separated* (consecutive peaks > `2 * mz_tolerance`),
//! the two-pointer sweep is provably optimal.

use geometric_traits::prelude::ScalarSimilarity;

use super::cosine_common::{collect_linear_matches, ensure_finite, validate_well_separated};
use super::entropy_common::{
    entropy_score_pairs, finalize_entropy_score, impl_entropy_config_api,
    impl_entropy_spectral_similarity, prepare_entropy_peaks,
};
use super::similarity_errors::SimilarityComputationError;
use crate::traits::Spectrum;

/// Linear-time spectral entropy similarity.
///
/// Requires spectra to be *well-separated*: consecutive peaks within each
/// spectrum must be greater than `2 * mz_tolerance`. Under this invariant the
/// two-pointer sweep is provably optimal.
///
/// Returns an error when the strict peak-spacing precondition is violated.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct LinearEntropy {
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    weighted: bool,
}

impl_entropy_config_api!(LinearEntropy, "linear entropy similarity");
impl_entropy_spectral_similarity!(LinearEntropy);

impl<S1, S2> ScalarSimilarity<S1, S2> for LinearEntropy
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

        let pairs = collect_linear_matches(&left_peaks.mz, &right_peaks.mz, tolerance, 0.0);
        let (raw_score, n_matches) =
            entropy_score_pairs(&pairs, &left_peaks.int, &right_peaks.int)?;

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
    fn zero_product_inputs_return_zero_similarity() {
        let scorer = LinearEntropy::new(0.0, 1.0, 0.1, false).expect("config should build");
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
}
