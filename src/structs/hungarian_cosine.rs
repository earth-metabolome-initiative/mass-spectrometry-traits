//! Implementation of the Hungarian cosine similarity for mass spectra.

use super::cosine_common::{impl_cosine_wrapper_config_api, impl_cosine_wrapper_similarity};

/// Implementation of the Hungarian cosine similarity for mass spectra.
pub struct HungarianCosine<EXP, MZ> {
    config: super::cosine_common::CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    HungarianCosine,
    "the Hungarian cosine similarity",
    "Returns the tolerance for the mass/charge ratio."
);

impl_cosine_wrapper_similarity!(
    HungarianCosine,
    super::cosine_common::compute_cosine_similarity,
    mz_tolerance,
    row,
    col,
    crate::traits::Spectrum::matching_peaks(row, col, mz_tolerance),
    crate::traits::Spectrum::matching_peaks(row, col, mz_tolerance)
);
