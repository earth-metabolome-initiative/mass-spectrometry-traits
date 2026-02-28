//! Submodule providing data for neburon.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of neburon.
pub trait NeburonSpectrum: SpectrumAlloc {
    /// Create a new spectrum of neburon.
    fn neburon() -> Self;
}

/// The precursor mass over charge value for neburon.
pub const NEBURON_PRECURSOR_MZ: f32 = 273.057;

/// The mass over charge values for neburon.
pub const NEBURON_MZ: [f32; 30] = [
    78.565781, 80.555191, 85.259598, 85.964752, 93.619781, 96.819359, 103.186424, 104.954704,
    112.697258, 118.455429, 121.980431, 123.995911, 143.397552, 149.975479, 149.978531, 152.465088,
    153.226852, 159.920059, 159.972839, 160.002106, 160.027054, 166.598648, 185.95195, 187.967514,
    216.994354, 227.856873, 231.525848, 243.581482, 273.057068, 273.177979,
];
/// The intensities for neburon.
pub const NEBURON_INTENSITIES: [f32; 30] = [
    6452.944824,
    6958.725586,
    7028.074707,
    7633.516113,
    6547.099121,
    6931.334473,
    6531.01709,
    26946.664062,
    6974.170898,
    18902.011719,
    12895.924805,
    39489.359375,
    7781.064941,
    614264.375,
    12004.375,
    7163.355957,
    7300.218262,
    9245.451172,
    7964397.5,
    7543.385742,
    29414.501953,
    7256.82373,
    2804349.5,
    12190.744141,
    184885.84375,
    10997.932617,
    7929.888184,
    7564.797852,
    6887249.5,
    11251.418945,
];

impl<S: SpectrumAlloc> NeburonSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn neburon() -> Self {
        let mut spectrum = Self::with_capacity(NEBURON_PRECURSOR_MZ.into(), NEBURON_MZ.len());
        for (&mz, &intensity) in NEBURON_MZ.iter().zip(NEBURON_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add neburon peak to spectrum");
        }
        spectrum
    }
}
