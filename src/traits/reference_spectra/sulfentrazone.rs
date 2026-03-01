//! Submodule providing data for sulfentrazone.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of sulfentrazone.
pub trait SulfentrazoneSpectrum: SpectrumAlloc {
    /// Create a new spectrum of sulfentrazone.
    fn sulfentrazone() -> Self;
}

/// The precursor mass over charge value for sulfentrazone.
pub const SULFENTRAZONE_PRECURSOR_MZ: f32 = 384.975;

/// The mass over charge values for sulfentrazone.
pub const SULFENTRAZONE_MZ: [f32; 21] = [
    150.961197, 174.005707, 198.948196, 199.949173, 201.951874, 206.544968, 225.957642, 231.004288,
    239.97377, 244.964264, 251.97789, 256.003418, 278.9776, 305.985931, 306.997101, 308.996948,
    334.975586, 364.968811, 384.975494, 385.966583, 386.970795,
];
/// The intensities for sulfentrazone.
pub const SULFENTRAZONE_INTENSITIES: [f32; 21] = [
    191.0, 182.0, 722.0, 192.0, 206.0, 95.0, 294.0, 243.0, 407.0, 183.0, 184.0, 107.0, 499.0,
    549.0, 2325.0, 218.0, 438.0, 153.0, 10655.0, 94.0, 90.0,
];

super::impl_reference_spectrum!(
    SulfentrazoneSpectrum,
    sulfentrazone,
    SULFENTRAZONE_PRECURSOR_MZ,
    SULFENTRAZONE_MZ,
    SULFENTRAZONE_INTENSITIES
);
