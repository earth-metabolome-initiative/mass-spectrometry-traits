//! Submodule providing data for the glucose molecule.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of glucose.
pub trait GlucoseSpectrum: SpectrumAlloc {
    /// Create a new spectrum of glucose.
    fn glucose() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for glucose.
pub const GLUCOSE_PRECURSOR_MZ: f32 = 203.05;

/// The mass over charge values for glucose.
pub const GLUCOSE_MZ: [f32; 14] = [
    82.952_15, 105.270_45, 112.789_4, 121.208, 129.116_7, 131.104_1, 131.990_69, 135.007_93,
    142.50119, 143.102_9, 158.092_25, 160.235_29, 173.100_46, 185.152_68,
];
/// The intensities for glucose.
pub const GLUCOSE_INTENSITIES: [f32; 14] = [
    798.858_9,
    1_257.253_4,
    3_923.249,
    1_952.965_5,
    169.587_34,
    412.055_18,
    309.939_5,
    520.569_15,
    555.742_43,
    13_786.814,
    1_758.816_4,
    408.889_77,
    892.346_9,
    9_220.535,
];

super::impl_reference_spectrum!(
    GlucoseSpectrum,
    glucose,
    GLUCOSE_PRECURSOR_MZ,
    GLUCOSE_MZ,
    GLUCOSE_INTENSITIES
);
