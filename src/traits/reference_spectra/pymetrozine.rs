//! Submodule providing data for pymetrozine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of pymetrozine.
pub trait PymetrozineSpectrum: SpectrumAlloc {
    /// Create a new spectrum of pymetrozine.
    fn pymetrozine() -> Self;
}

/// The precursor mass over charge value for pymetrozine.
pub const PYMETROZINE_PRECURSOR_MZ: f32 = 254.045;

/// The mass over charge values for pymetrozine.
pub const PYMETROZINE_MZ: [f32; 41] = [
    70.003845, 74.725365, 77.65255, 85.287811, 85.999191, 86.000511, 86.484047, 92.885658,
    93.432137, 95.712517, 96.275879, 96.709457, 98.925407, 100.503563, 101.941376, 107.684593,
    109.037773, 109.100983, 110.905235, 111.995857, 117.045731, 120.074455, 124.074234, 124.787483,
    126.011551, 128.034103, 145.574936, 153.022446, 164.909042, 175.273209, 179.997025, 195.328613,
    196.101913, 201.050308, 207.044556, 210.044022, 230.065887, 233.988083, 236.035049, 242.080521,
    254.045197,
];
/// The intensities for pymetrozine.
pub const PYMETROZINE_INTENSITIES: [f32; 41] = [
    246.715271,
    252.824051,
    270.032776,
    266.715057,
    24820.123047,
    561.685608,
    246.809708,
    281.303497,
    289.714325,
    215.064163,
    264.006531,
    281.458893,
    261.101562,
    225.727478,
    251.957291,
    241.729324,
    225.902481,
    233.304367,
    241.248505,
    996.720215,
    413.455872,
    295.724823,
    245.455338,
    237.767181,
    1445.849854,
    652.296082,
    273.771881,
    75320.53125,
    252.21843,
    275.631897,
    542.628906,
    295.754272,
    255.563812,
    275.496063,
    4027.952148,
    6919.461914,
    256.506836,
    622.810852,
    279.074432,
    246.973282,
    158700.890625,
];

impl<S: SpectrumAlloc> PymetrozineSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn pymetrozine() -> Self {
        let mut spectrum =
            Self::with_capacity(PYMETROZINE_PRECURSOR_MZ.into(), PYMETROZINE_MZ.len());
        for (&mz, &intensity) in PYMETROZINE_MZ.iter().zip(PYMETROZINE_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add pymetrozine peak to spectrum");
        }
        spectrum
    }
}
