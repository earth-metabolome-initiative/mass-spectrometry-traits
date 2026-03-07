//! Submodule providing data for arachidic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of arachidic acid.
pub trait ArachidicAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of arachidic acid.
    fn arachidic_acid() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for arachidic acid.
pub const ARACHIDIC_ACID_PRECURSOR_MZ: f32 = 311.313;

/// The mass over charge values for arachidic acid.
pub const ARACHIDIC_ACID_MZ: [f32; 14] = [
    50.299179, 57.300552, 59.769974, 70.039909, 77.463814, 78.139076, 90.732834, 94.812172,
    96.737442, 108.702026, 146.965622, 174.965546, 311.317657, 312.325165,
];
/// The intensities for arachidic acid.
pub const ARACHIDIC_ACID_INTENSITIES: [f32; 14] = [
    3439.615234,
    3587.85376,
    3063.953369,
    3177.591309,
    3341.716064,
    3331.392334,
    3709.186523,
    3306.816895,
    7007.583496,
    3708.455811,
    4947.753418,
    55672.625,
    711829.625,
    62529.160156,
];

super::impl_reference_spectrum!(
    ArachidicAcidSpectrum,
    arachidic_acid,
    ARACHIDIC_ACID_PRECURSOR_MZ,
    ARACHIDIC_ACID_MZ,
    ARACHIDIC_ACID_INTENSITIES
);
