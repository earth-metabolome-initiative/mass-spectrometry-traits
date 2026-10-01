//! Submodule providing data for citric acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of citric acid.
pub trait CitricAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of citric acid.
    fn citric_acid() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for citric acid.
pub const CITRIC_ACID_PRECURSOR_MZ: f32 = 191.035;

/// The mass over charge values for citric acid.
pub const CITRIC_ACID_MZ: [f32; 24] = [
    55.309017, 57.033176, 57.775112, 59.013027, 67.018593, 72.744606, 75.912361, 85.030861,
    86.033173, 87.009438, 88.012619, 101.025459, 103.041756, 111.011368, 112.014816, 116.05545,
    129.024292, 130.027008, 131.003357, 147.034683, 155.004623, 173.018311, 191.03299, 192.03421,
];
/// The intensities for citric acid.
pub const CITRIC_ACID_INTENSITIES: [f32; 24] = [
    26988.396484,
    1072515.625,
    24416.537109,
    23209.166016,
    84438.867188,
    45378.042969,
    29132.867188,
    7550744.5,
    159962.09375,
    11575775.0,
    185860.828125,
    131822.875,
    115444.835938,
    32460302.0,
    912500.875,
    26533.630859,
    2809614.5,
    49458.449219,
    280171.03125,
    268974.84375,
    187163.109375,
    693458.125,
    5119567.5,
    185818.765625,
];

super::impl_reference_spectrum!(
    CitricAcidSpectrum,
    citric_acid,
    CITRIC_ACID_PRECURSOR_MZ,
    CITRIC_ACID_MZ,
    CITRIC_ACID_INTENSITIES
);
