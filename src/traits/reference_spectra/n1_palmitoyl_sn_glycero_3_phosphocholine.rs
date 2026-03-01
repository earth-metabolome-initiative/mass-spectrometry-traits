//! Submodule providing data for n1 palmitoyl sn glycero 3 phosphocholine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of n1 palmitoyl sn glycero 3 phosphocholine.
pub trait N1PalmitoylSnGlycero3PhosphocholineSpectrum: SpectrumAlloc {
    /// Create a new spectrum of n1 palmitoyl sn glycero 3 phosphocholine.
    fn n1_palmitoyl_sn_glycero_3_phosphocholine() -> Self;
}

/// The precursor mass over charge value for n1 palmitoyl sn glycero 3 phosphocholine.
pub const N1_PALMITOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_PRECURSOR_MZ: f32 = 496.34;

/// The mass over charge values for n1 palmitoyl sn glycero 3 phosphocholine.
pub const N1_PALMITOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_MZ: [f32; 36] = [
    51.36277, 56.965591, 58.066555, 59.0746, 60.081715, 60.854828, 62.256985, 65.978775, 69.60791,
    71.080765, 81.072037, 85.103256, 86.098892, 95.088104, 98.98748, 104.110771, 125.006012,
    163.023254, 181.035599, 184.08432, 193.512405, 199.050293, 240.115219, 250.612717, 258.13324,
    283.284912, 313.307526, 370.64444, 377.59964, 382.784271, 409.699921, 419.300598, 423.371582,
    424.023407, 478.392975, 496.398193,
];
/// The intensities for n1 palmitoyl sn glycero 3 phosphocholine.
pub const N1_PALMITOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_INTENSITIES: [f32; 36] = [
    2014644.25,
    7101339.0,
    19658010.0,
    2106540.0,
    65814812.0,
    1877933.0,
    1740470.5,
    3891545.5,
    2146809.5,
    16124084.0,
    3707660.5,
    2068801.625,
    268932224.0,
    2035273.875,
    1854370.375,
    2290994432.0,
    53993288.0,
    11685794.0,
    3609697.0,
    2913305856.0,
    2514100.0,
    4470432.0,
    2100184.25,
    2603946.75,
    59326564.0,
    1992020.75,
    97899016.0,
    2168631.0,
    2923469.5,
    1902304.25,
    2106601.25,
    27471138.0,
    2398145.5,
    2181241.25,
    164633648.0,
    125002984.0,
];

super::impl_reference_spectrum!(
    N1PalmitoylSnGlycero3PhosphocholineSpectrum,
    n1_palmitoyl_sn_glycero_3_phosphocholine,
    N1_PALMITOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_PRECURSOR_MZ,
    N1_PALMITOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_MZ,
    N1_PALMITOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_INTENSITIES
);
