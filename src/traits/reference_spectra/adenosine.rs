//! Submodule providing data for adenosine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of adenosine.
pub trait AdenosineSpectrum: SpectrumAlloc {
    /// Create a new spectrum of adenosine.
    fn adenosine() -> Self;
}

/// The precursor mass over charge value for adenosine.
pub const ADENOSINE_PRECURSOR_MZ: f32 = 268.105;

/// The mass over charge values for adenosine.
pub const ADENOSINE_MZ: [f32; 35] = [
    51.011391, 52.01926, 53.017235, 54.022449, 55.020733, 57.034298, 59.050282, 61.029453,
    65.014641, 66.023109, 67.026573, 69.034981, 71.014297, 73.030754, 74.097778, 75.04615,
    77.025414, 82.042183, 85.031548, 87.046661, 92.027817, 93.03524, 94.042786, 97.031265,
    108.045868, 115.043411, 119.040314, 120.047226, 135.058197, 136.068146, 137.071655, 178.084503,
    250.110535, 268.127045, 269.127563,
];
/// The intensities for adenosine.
pub const ADENOSINE_INTENSITIES: [f32; 35] = [
    12575.009766,
    12244.470703,
    157375.59375,
    96299.914062,
    2317442.0,
    3178846.75,
    253846.171875,
    538790.4375,
    119906.546875,
    408811.9375,
    8363.267578,
    2486099.0,
    568842.25,
    2615089.75,
    129135.617188,
    36518.726562,
    37041.964844,
    26264.203125,
    5306120.5,
    705608.25,
    121187.296875,
    53959.460938,
    135012.328125,
    578719.125,
    98197.179688,
    888532.375,
    752642.1875,
    25805.783203,
    80733.054688,
    508259936.0,
    862069.75,
    168991.171875,
    75951.609375,
    17627020.0,
    42165.738281,
];

super::impl_reference_spectrum!(
    AdenosineSpectrum,
    adenosine,
    ADENOSINE_PRECURSOR_MZ,
    ADENOSINE_MZ,
    ADENOSINE_INTENSITIES
);
