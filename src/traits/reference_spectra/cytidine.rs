//! Submodule providing data for cytidine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of cytidine.
pub trait CytidineSpectrum: SpectrumAlloc {
    /// Create a new spectrum of cytidine.
    fn cytidine() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for cytidine.
pub const CYTIDINE_PRECURSOR_MZ: f32 = 244.09;

/// The mass over charge values for cytidine.
pub const CYTIDINE_MZ: [f32; 20] = [
    52.019474, 53.027031, 55.018566, 57.034271, 67.030914, 68.038086, 69.043175, 73.029907,
    85.030792, 87.046883, 94.04261, 95.027168, 97.031456, 103.041801, 112.05471, 113.057205,
    115.041733, 133.055267, 157.117874, 244.115021,
];
/// The intensities for cytidine.
pub const CYTIDINE_INTENSITIES: [f32; 20] = [
    122090.726562,
    12754.80957,
    59303.785156,
    164545.375,
    39722.273438,
    45352.679688,
    287177.0625,
    120080.898438,
    129825.390625,
    33839.511719,
    78339.539062,
    530138.375,
    56686.054688,
    30518.804688,
    257622784.0,
    1043890.75,
    211091.6875,
    303797.71875,
    26490.792969,
    820996.125,
];

super::impl_reference_spectrum!(
    CytidineSpectrum,
    cytidine,
    CYTIDINE_PRECURSOR_MZ,
    CYTIDINE_MZ,
    CYTIDINE_INTENSITIES
);
