//! Submodule providing data for n4 aminobenzoic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of n4 aminobenzoic acid.
pub trait N4AminobenzoicAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of n4 aminobenzoic acid.
    fn n4_aminobenzoic_acid() -> Self;
}

/// The precursor mass over charge value for n4 aminobenzoic acid.
pub const N4_AMINOBENZOIC_ACID_PRECURSOR_MZ: f32 = 138.055;

/// The mass over charge values for n4 aminobenzoic acid.
pub const N4_AMINOBENZOIC_ACID_MZ: [f32; 27] = [
    50.016323, 51.024139, 53.039322, 53.703335, 60.52816, 65.041016, 67.056046, 71.260994,
    75.02533, 76.032616, 77.040817, 79.056847, 81.035889, 92.052719, 93.060478, 94.06855,
    95.052765, 96.047089, 105.049187, 110.063156, 120.049873, 121.034462, 132.7099, 138.061478,
    139.06308, 149.041687, 156.072052,
];
/// The intensities for n4 aminobenzoic acid.
pub const N4_AMINOBENZOIC_ACID_INTENSITIES: [f32; 27] = [
    68476.070312,
    55785.453125,
    752443.6875,
    33525.296875,
    28843.992188,
    697745.1875,
    314985.3125,
    239734.9375,
    47529.8125,
    27754.806641,
    861075.0,
    31853.894531,
    54610.789062,
    184124.859375,
    2033454.75,
    19333656.0,
    4399703.0,
    42647.984375,
    1297030.0,
    120648.554688,
    3671001.75,
    312304.71875,
    29649.253906,
    135688320.0,
    1622096.5,
    564485.9375,
    172943.5,
];

super::impl_reference_spectrum!(
    N4AminobenzoicAcidSpectrum,
    n4_aminobenzoic_acid,
    N4_AMINOBENZOIC_ACID_PRECURSOR_MZ,
    N4_AMINOBENZOIC_ACID_MZ,
    N4_AMINOBENZOIC_ACID_INTENSITIES
);
