//! Submodule providing data for ascorbic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of ascorbic acid.
pub trait AscorbicAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of ascorbic acid.
    fn ascorbic_acid() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for ascorbic acid.
pub const ASCORBIC_ACID_PRECURSOR_MZ: f32 = 175.024;

/// The mass over charge values for ascorbic acid.
pub const ASCORBIC_ACID_MZ: [f32; 42] = [
    53.001652, 55.017174, 57.014565, 58.004681, 59.012596, 60.014759, 60.992226, 67.018181,
    69.015503, 71.013306, 72.016899, 73.004959, 74.019714, 75.008507, 83.014206, 84.017136,
    85.028069, 86.020142, 87.009476, 88.012764, 95.014046, 96.994072, 99.010262, 100.012817,
    101.02639, 103.005859, 105.021652, 111.011009, 113.011147, 114.030151, 115.007126, 116.009918,
    117.021637, 127.007904, 129.022736, 131.039505, 139.007797, 147.036224, 157.021301, 161.031235,
    175.032471, 176.037323,
];
/// The intensities for ascorbic acid.
pub const ASCORBIC_ACID_INTENSITIES: [f32; 42] = [
    4819.629395,
    97560.703125,
    567121.875,
    238461.4375,
    26324304.0,
    288406.53125,
    10762.765625,
    261879.03125,
    128148.054688,
    24254448.0,
    369049.28125,
    3914393.25,
    34280.472656,
    293282.625,
    924209.125,
    18528.539062,
    2249360.75,
    75019.328125,
    54513956.0,
    732946.6875,
    95593.734375,
    53625.683594,
    908034.1875,
    9818.677734,
    385683.625,
    221342.984375,
    38340.539062,
    169007.546875,
    1950227.75,
    19225.458984,
    23456662.0,
    457756.375,
    9479.566406,
    280419.625,
    154423.65625,
    28395.685547,
    79431.828125,
    16254.768555,
    51913.722656,
    139088.75,
    27175972.0,
    808394.6875,
];

super::impl_reference_spectrum!(
    AscorbicAcidSpectrum,
    ascorbic_acid,
    ASCORBIC_ACID_PRECURSOR_MZ,
    ASCORBIC_ACID_MZ,
    ASCORBIC_ACID_INTENSITIES
);
