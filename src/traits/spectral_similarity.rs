//! Submodule defining a spectral similarity trait.

use geometric_traits::prelude::ScalarSimilarity;

use crate::prelude::Spectrum;

/// Trait for calculating the similarity between two [`Spectrum`]s.
pub trait ScalarSpectralSimilarity<Left: Spectrum, Right: Spectrum>:
    ScalarSimilarity<Left, Right>
{
}
