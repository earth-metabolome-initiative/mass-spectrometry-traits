//! Submodule providing data for cysteine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of cysteine.
pub trait CysteineSpectrum: SpectrumAlloc {
    /// Create a new spectrum of cysteine.
    fn cysteine() -> Self;
}

/// The precursor mass over charge value for cysteine.
pub const CYSTEINE_PRECURSOR_MZ: f32 = 122.03;

/// The mass over charge values for cysteine.
pub const CYSTEINE_MZ: [f32; 18] = [
    56.050392, 57.034729, 58.996159, 73.221016, 74.007462, 76.022491, 80.0513, 86.992828,
    88.042358, 93.066216, 94.0681, 104.074631, 105.004654, 107.076355, 107.622299, 122.079407,
    123.059181, 127.828667,
];
/// The intensities for cysteine.
pub const CYSTEINE_INTENSITIES: [f32; 18] = [
    42345.585938,
    8275.46875,
    386116.8125,
    8805.25,
    7222.203613,
    1004393.6875,
    3623.723633,
    243361.84375,
    5243.276367,
    11583.741211,
    10610.408203,
    64400.964844,
    289717.34375,
    18734.919922,
    4155.836914,
    520521.84375,
    17090.435547,
    4254.670898,
];

impl<S: SpectrumAlloc> CysteineSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn cysteine() -> Self {
        let mut spectrum = Self::with_capacity(CYSTEINE_PRECURSOR_MZ.into(), CYSTEINE_MZ.len());
        for (&mz, &intensity) in CYSTEINE_MZ.iter().zip(CYSTEINE_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add cysteine peak to spectrum");
        }
        spectrum
    }
}
