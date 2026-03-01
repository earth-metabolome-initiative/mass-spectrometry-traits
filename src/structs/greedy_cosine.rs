//! Implementation of the greedy cosine similarity for mass spectra.
//!
//! This matches the matchms `CosineGreedy` algorithm: candidate peak pairs
//! are sorted by descending product score and greedily assigned (each peak
//! used at most once).

use super::cosine_common::{impl_cosine_wrapper_config_api, impl_cosine_wrapper_similarity};

/// Greedy cosine similarity for mass spectra.
///
/// Matches the matchms `CosineGreedy` algorithm: candidate peak pairs within
/// `mz_tolerance` are sorted by descending product weight and greedily
/// assigned so that each peak is used at most once.
pub struct GreedyCosine<EXP, MZ> {
    config: super::cosine_common::CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    GreedyCosine,
    "the greedy cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl_cosine_wrapper_similarity!(
    GreedyCosine,
    super::cosine_common::compute_cosine_similarity_greedy,
    mz_tolerance,
    row,
    col,
    crate::traits::Spectrum::matching_peaks(row, col, mz_tolerance),
    crate::traits::Spectrum::matching_peaks(row, col, mz_tolerance)
);
