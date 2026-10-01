//! Submodule providing data for adenine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of adenine.
pub trait AdenineSpectrum: SpectrumAlloc {
    /// Create a new spectrum of adenine.
    fn adenine() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for adenine.
pub const ADENINE_PRECURSOR_MZ: f32 = 136.062;

/// The mass over charge values for adenine.
pub const ADENINE_MZ: [f32; 10] = [
    65.869789, 83.021255, 91.772469, 94.043243, 109.054817, 119.040543, 120.041893, 136.068161,
    137.07045, 138.055847,
];
/// The intensities for adenine.
pub const ADENINE_INTENSITIES: [f32; 10] = [
    1141476.625,
    1235087.375,
    979065.5625,
    3044282.75,
    1286043.0,
    14678569.0,
    3967049.75,
    974787008.0,
    253094176.0,
    2822116.25,
];

super::impl_reference_spectrum!(
    AdenineSpectrum,
    adenine,
    ADENINE_PRECURSOR_MZ,
    ADENINE_MZ,
    ADENINE_INTENSITIES
);
