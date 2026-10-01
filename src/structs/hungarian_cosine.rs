//! Implementation of the Hungarian cosine similarity for mass spectra.

use super::cosine_common::{impl_cosine_wrapper_config_api, impl_cosine_wrapper_similarity};

/// Implementation of the Hungarian cosine similarity for mass spectra.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct HungarianCosine {
    config: super::cosine_common::CosineConfig,
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
