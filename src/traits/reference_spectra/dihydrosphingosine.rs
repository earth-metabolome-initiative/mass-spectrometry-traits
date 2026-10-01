//! Submodule providing data for dihydrosphingosine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of dihydrosphingosine.
pub trait DihydrosphingosineSpectrum: SpectrumAlloc {
    /// Create a new spectrum of dihydrosphingosine.
    fn dihydrosphingosine() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for dihydrosphingosine.
pub const DIHYDROSPHINGOSINE_PRECURSOR_MZ: f32 = 302.31;

/// The mass over charge values for dihydrosphingosine.
pub const DIHYDROSPHINGOSINE_MZ: [f32; 33] = [
    55.054996, 57.070847, 60.045559, 61.037689, 67.053925, 69.07164, 70.066589, 71.050674,
    71.785172, 74.062134, 79.055984, 81.072159, 83.087936, 85.099113, 95.088875, 97.104568,
    99.98259, 109.105804, 111.121216, 123.122093, 125.136955, 137.138062, 151.154663, 205.604324,
    240.290863, 249.2733, 254.306625, 255.304108, 266.308258, 267.28775, 284.322296, 285.322632,
    302.336945,
];
/// The intensities for dihydrosphingosine.
pub const DIHYDROSPHINGOSINE_INTENSITIES: [f32; 33] = [
    943072.1875,
    1335143.0,
    55192548.0,
    226691.4375,
    3798174.0,
    2830709.25,
    592200.875,
    577995.875,
    381769.3125,
    260135.484375,
    176402.234375,
    6944893.5,
    4423350.5,
    1005971.25,
    10111571.0,
    4426984.0,
    93259.867188,
    5433791.5,
    1911117.25,
    2206172.5,
    273595.8125,
    654352.875,
    119006.148438,
    321319.4375,
    3070377.0,
    118263.648438,
    10303927.0,
    176650.625,
    9577478.0,
    182076.015625,
    60919528.0,
    2085753.5,
    4255582.5,
];

super::impl_reference_spectrum!(
    DihydrosphingosineSpectrum,
    dihydrosphingosine,
    DIHYDROSPHINGOSINE_PRECURSOR_MZ,
    DIHYDROSPHINGOSINE_MZ,
    DIHYDROSPHINGOSINE_INTENSITIES
);
