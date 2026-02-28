//! Submodule providing data for desmosterol.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of desmosterol.
pub trait DesmosterolSpectrum: SpectrumAlloc {
    /// Create a new spectrum of desmosterol.
    fn desmosterol() -> Self;
}

/// The precursor mass over charge value for desmosterol.
pub const DESMOSTEROL_PRECURSOR_MZ: f32 = 385.439;

/// The mass over charge values for desmosterol.
pub const DESMOSTEROL_MZ: [f32; 50] = [
    57.070545, 69.071159, 71.086914, 81.072067, 83.062744, 93.073021, 95.088928, 97.069519,
    107.042076, 109.0709, 111.120369, 119.090279, 121.09819, 123.094826, 125.136765, 133.106857,
    135.121628, 137.125427, 145.107224, 147.12352, 149.138779, 151.142899, 159.124283, 161.139755,
    163.137482, 173.141571, 175.15683, 177.141968, 179.188797, 185.141342, 187.159119, 189.172379,
    191.159576, 193.205811, 199.160629, 201.176117, 203.166351, 205.194641, 213.177536, 215.193619,
    217.17717, 219.202393, 227.19455, 231.196533, 241.212875, 255.233688, 259.237762, 273.251007,
    367.370056, 385.387756,
];
/// The intensities for desmosterol.
pub const DESMOSTEROL_INTENSITIES: [f32; 50] = [
    11004014.0,
    10135773.0,
    12771678.0,
    29456328.0,
    59386244.0,
    5465900.5,
    39435492.0,
    496477664.0,
    13328427.0,
    510063360.0,
    14257538.0,
    7493996.0,
    14445735.0,
    68101408.0,
    7219818.5,
    11233749.0,
    10042070.0,
    15306982.0,
    9567521.0,
    11445974.0,
    18910480.0,
    8332197.5,
    13272444.0,
    12480081.0,
    25533666.0,
    10521535.0,
    7246767.0,
    18020994.0,
    9010143.0,
    5516387.5,
    8660372.0,
    5419574.0,
    11738866.0,
    8352818.0,
    8585796.0,
    11776112.0,
    10283285.0,
    9683710.0,
    11170467.0,
    8547426.0,
    9122228.0,
    8671463.0,
    10896007.0,
    9950591.0,
    16124764.0,
    24344224.0,
    6493542.0,
    6973478.0,
    81755248.0,
    255675520.0,
];

impl<S: SpectrumAlloc> DesmosterolSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn desmosterol() -> Self {
        let mut spectrum =
            Self::with_capacity(DESMOSTEROL_PRECURSOR_MZ.into(), DESMOSTEROL_MZ.len());
        for (&mz, &intensity) in DESMOSTEROL_MZ.iter().zip(DESMOSTEROL_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add desmosterol peak to spectrum");
        }
        spectrum
    }
}
