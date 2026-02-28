//! Submodule providing data for diniconazole.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of diniconazole.
pub trait DiniconazoleSpectrum: SpectrumAlloc {
    /// Create a new spectrum of diniconazole.
    fn diniconazole() -> Self;
}

/// The precursor mass over charge value for diniconazole.
pub const DINICONAZOLE_PRECURSOR_MZ: f32 = 370.073;

/// The mass over charge values for diniconazole.
pub const DINICONAZOLE_MZ: [f32; 30] = [
    71.509712, 75.734535, 78.284218, 81.917122, 87.434189, 88.290558, 91.11673, 92.44825,
    92.876999, 98.838715, 102.538101, 103.417053, 115.073303, 122.155457, 125.018257, 127.487625,
    131.096695, 148.284576, 177.838348, 179.272049, 183.972763, 191.779327, 208.119919, 223.643356,
    224.699524, 237.99501, 296.771454, 297.930206, 334.23468, 358.375488,
];
/// The intensities for diniconazole.
pub const DINICONAZOLE_INTENSITIES: [f32; 30] = [
    5724.83252,
    5299.964844,
    5699.005371,
    5858.859375,
    6089.695801,
    5526.692383,
    5547.290039,
    5810.729004,
    5978.911133,
    6258.443359,
    6125.699219,
    5841.128906,
    6023.207031,
    7037.744629,
    7178.09082,
    6299.458496,
    6535.404785,
    6097.58252,
    6710.890137,
    6818.189941,
    80886.570312,
    6108.046387,
    7749.924316,
    6566.936035,
    6282.483887,
    33629.90625,
    6773.456543,
    6737.882812,
    7306.113281,
    6645.259277,
];

impl<S: SpectrumAlloc> DiniconazoleSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn diniconazole() -> Self {
        let mut spectrum =
            Self::with_capacity(DINICONAZOLE_PRECURSOR_MZ.into(), DINICONAZOLE_MZ.len());
        for (&mz, &intensity) in DINICONAZOLE_MZ.iter().zip(DINICONAZOLE_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add diniconazole peak to spectrum");
        }
        spectrum
    }
}
