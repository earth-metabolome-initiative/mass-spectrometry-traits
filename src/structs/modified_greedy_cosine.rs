//! Implementation of the modified greedy cosine similarity for mass spectra.
//!
//! Modified Greedy Cosine extends Greedy Cosine by also matching fragment peaks
//! shifted by the precursor mass difference. This matches the matchms
//! `ModifiedCosine` algorithm.

use super::cosine_common::{impl_cosine_wrapper_config_api, impl_cosine_wrapper_similarity};

/// Modified greedy cosine similarity for mass spectra.
///
/// Extends [`super::GreedyCosine`] by also matching fragment peaks shifted by
/// the precursor mass difference, using greedy assignment. This matches the
/// matchms `ModifiedCosine` algorithm.
pub struct ModifiedGreedyCosine<EXP, MZ> {
    config: super::cosine_common::CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    ModifiedGreedyCosine,
    "the modified greedy cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl_cosine_wrapper_similarity!(
    ModifiedGreedyCosine,
    super::cosine_common::compute_cosine_similarity_greedy,
    mz_tolerance,
    row,
    col,
    crate::traits::Spectrum::modified_matching_peaks(
        row,
        col,
        mz_tolerance,
        crate::traits::Spectrum::precursor_mz(row) - crate::traits::Spectrum::precursor_mz(col)
    ),
    crate::traits::Spectrum::modified_matching_peaks(
        row,
        col,
        mz_tolerance,
        crate::traits::Spectrum::precursor_mz(row) - crate::traits::Spectrum::precursor_mz(col)
    )
);
