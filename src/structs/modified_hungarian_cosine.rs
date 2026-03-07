//! Implementation of the modified Hungarian cosine similarity for mass spectra.
//!
//! Modified Hungarian Cosine extends Hungarian Cosine by also matching fragment
//! peaks shifted by the precursor mass difference when that shift exceeds the
//! configured tolerance. This captures neutral-loss-related peak
//! correspondences between spectra with different precursor masses.
//!
//! Unlike matchms `ModifiedCosine` (greedy assignment), this uses Crouse
//! rectangular LAPJV for optimal assignment.

use super::cosine_common::{impl_cosine_wrapper_config_api, impl_cosine_wrapper_similarity};

/// Modified cosine similarity for mass spectra.
///
/// Extends [`super::HungarianCosine`] by also matching fragment peaks shifted
/// by the precursor mass difference when that shift exceeds the configured
/// tolerance, using optimal (Crouse LAPJV) assignment.
pub struct ModifiedHungarianCosine<EXP, MZ> {
    config: super::cosine_common::CosineConfig<EXP, MZ>,
}

impl_cosine_wrapper_config_api!(
    ModifiedHungarianCosine,
    "the modified Hungarian cosine similarity",
    "Returns the tolerance for the mass-shift of the mass/charge ratio."
);

impl_cosine_wrapper_similarity!(
    ModifiedHungarianCosine,
    super::cosine_common::compute_cosine_similarity,
    mz_tolerance,
    row,
    col,
    crate::traits::Spectrum::modified_matching_peaks(
        row,
        col,
        mz_tolerance,
        crate::traits::Spectrum::precursor_mz(row),
        crate::traits::Spectrum::precursor_mz(col)
    ),
    crate::traits::Spectrum::modified_matching_peaks(
        row,
        col,
        mz_tolerance,
        crate::traits::Spectrum::precursor_mz(row),
        crate::traits::Spectrum::precursor_mz(col)
    )
);
