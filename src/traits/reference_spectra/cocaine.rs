//! Submodule providing data for the cocaine molecule.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of cocaine.
pub trait CocaineSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of cocaine.
    fn cocaine() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for cocaine.
pub const COCAINE_PRECURSOR_MZ: f32 = 304.153_14;

/// The mass over charge values for cocaine.
pub const COCAINE_MZ: [f32; 9] = [
    82.064_79, 105.033_25, 109.213745, 119.04921, 150.0914, 182.117_68, 185.804_69, 226.579_07,
    304.153_14,
];
/// The intensities for cocaine.
pub const COCAINE_INTENSITIES: [f32; 9] = [
    13_342.493,
    3_264.133_5,
    1_584.274_8,
    2_382.931,
    3_257.366_2,
    133_504.3,
    1_849.140_1,
    1_391.734_5,
    86052.375,
];

super::impl_reference_spectrum!(
    CocaineSpectrum,
    cocaine,
    COCAINE_PRECURSOR_MZ,
    COCAINE_MZ,
    COCAINE_INTENSITIES
);
