//! Submodule providing data for arginine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of arginine.
pub trait ArginineSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of arginine.
    fn arginine() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for arginine.
pub const ARGININE_PRECURSOR_MZ: f32 = 175.119;

/// The mass over charge values for arginine.
pub const ARGININE_MZ: [f32; 25] = [
    60.056824, 70.066811, 71.062859, 72.082695, 72.967789, 81.936661, 88.078316, 97.078789,
    98.062721, 110.461555, 112.090759, 113.074547, 114.106842, 115.090767, 116.075928, 117.077805,
    130.103699, 133.165359, 140.08812, 141.071899, 157.116974, 158.100647, 159.102417, 175.131241,
    176.131012,
];
/// The intensities for arginine.
pub const ARGININE_INTENSITIES: [f32; 25] = [
    20566794.0,
    17220388.0,
    129677.507812,
    1394306.0,
    64787.1875,
    32840.96875,
    63569.90625,
    185473.484375,
    152986.984375,
    33604.675781,
    1915127.0,
    223662.546875,
    717526.9375,
    592988.875,
    21967044.0,
    164672.046875,
    7933185.5,
    201222.609375,
    185517.296875,
    154714.6875,
    1577862.125,
    8488992.0,
    53228.253906,
    46103468.0,
    249996.796875,
];

super::impl_reference_spectrum!(
    ArginineSpectrum,
    arginine,
    ARGININE_PRECURSOR_MZ,
    ARGININE_MZ,
    ARGININE_INTENSITIES
);
