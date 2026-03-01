//! Submodule providing data for n2 5 dihydroxybenzoic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of n2 5 dihydroxybenzoic acid.
pub trait N25DihydroxybenzoicAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of n2 5 dihydroxybenzoic acid.
    fn n2_5_dihydroxybenzoic_acid() -> Self;
}

/// The precursor mass over charge value for n2 5 dihydroxybenzoic acid.
pub const N2_5_DIHYDROXYBENZOIC_ACID_PRECURSOR_MZ: f32 = 153.019;

/// The mass over charge values for n2 5 dihydroxybenzoic acid.
pub const N2_5_DIHYDROXYBENZOIC_ACID_MZ: [f32; 26] = [
    57.745575, 66.162102, 69.016235, 71.05275, 72.331573, 79.01915, 81.027649, 81.603699,
    85.029915, 95.006508, 96.186348, 97.030617, 99.010056, 104.37291, 105.440338, 108.024094,
    109.032532, 110.03569, 113.026794, 120.501625, 122.041016, 123.013351, 137.024582, 152.031998,
    153.027039, 154.029434,
];
/// The intensities for n2 5 dihydroxybenzoic acid.
pub const N2_5_DIHYDROXYBENZOIC_ACID_INTENSITIES: [f32; 26] = [
    2997.222168,
    3343.909668,
    30642.314453,
    19795.748047,
    9998.125,
    7102.51123,
    45549.53125,
    3546.741943,
    114824.90625,
    140172.21875,
    3408.001465,
    3058.425293,
    6869.054688,
    3438.462891,
    3170.231445,
    474899.84375,
    2965866.0,
    95776.25,
    6849.272461,
    6527.620605,
    7757.842773,
    113434.804688,
    7016.287598,
    88642.71875,
    1870490.375,
    72867.859375,
];

super::impl_reference_spectrum!(
    N25DihydroxybenzoicAcidSpectrum,
    n2_5_dihydroxybenzoic_acid,
    N2_5_DIHYDROXYBENZOIC_ACID_PRECURSOR_MZ,
    N2_5_DIHYDROXYBENZOIC_ACID_MZ,
    N2_5_DIHYDROXYBENZOIC_ACID_INTENSITIES
);
