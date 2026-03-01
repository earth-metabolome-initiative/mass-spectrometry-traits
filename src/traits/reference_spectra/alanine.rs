//! Submodule providing data for alanine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of alanine.
pub trait AlanineSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of alanine.
    fn alanine() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for alanine.
pub const ALANINE_PRECURSOR_MZ: f32 = 90.0555;

/// The mass over charge values for alanine.
pub const ALANINE_MZ: [f32; 10] = [
    55.055328, 56.014194, 57.058414, 70.066986, 71.160225, 72.050529, 73.049271, 89.075317,
    90.075943, 104.713135,
];
/// The intensities for alanine.
pub const ALANINE_INTENSITIES: [f32; 10] = [
    1669.299316,
    2542.530273,
    2042.29834,
    2254.094482,
    5241.987793,
    850956.625,
    227528.765625,
    101509.6875,
    160398.0,
    1510.326904,
];

super::impl_reference_spectrum!(
    AlanineSpectrum,
    alanine,
    ALANINE_PRECURSOR_MZ,
    ALANINE_MZ,
    ALANINE_INTENSITIES
);
