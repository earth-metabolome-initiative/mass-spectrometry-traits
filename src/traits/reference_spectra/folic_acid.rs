//! Submodule providing data for folic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of folic acid.
pub trait FolicAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of folic acid.
    fn folic_acid() -> Self;
}

/// The precursor mass over charge value for folic acid.
pub const FOLIC_ACID_PRECURSOR_MZ: f32 = 442.16;

/// The mass over charge values for folic acid.
pub const FOLIC_ACID_MZ: [f32; 25] = [
    58.880062, 71.700081, 81.046005, 82.769615, 106.044563, 108.060692, 120.049431, 121.051521,
    134.038895, 148.068359, 149.05098, 176.067825, 177.070648, 182.094254, 194.077957, 250.090591,
    267.120911, 269.137848, 295.124634, 296.127686, 305.124786, 313.137024, 314.134674, 327.149841,
    442.19516,
];
/// The intensities for folic acid.
pub const FOLIC_ACID_INTENSITIES: [f32; 25] = [
    48356.238281,
    295197.40625,
    123308.601562,
    41704.882812,
    874843.75,
    300754.59375,
    20700240.0,
    67938.40625,
    279719.84375,
    53475.984375,
    50057.117188,
    16705402.0,
    215187.171875,
    48479.125,
    61921.226562,
    74274.101562,
    484785.71875,
    945684.125,
    306769024.0,
    3412429.25,
    386006.0625,
    33077454.0,
    374116.0,
    67593.265625,
    462587.34375,
];

super::impl_reference_spectrum!(
    FolicAcidSpectrum,
    folic_acid,
    FOLIC_ACID_PRECURSOR_MZ,
    FOLIC_ACID_MZ,
    FOLIC_ACID_INTENSITIES
);
