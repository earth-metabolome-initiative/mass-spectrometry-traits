//! Submodule providing data for triadimefon.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of triadimefon.
pub trait TriadimefonSpectrum: SpectrumAlloc {
    /// Create a new spectrum of triadimefon.
    fn triadimefon() -> Self;
}

/// The precursor mass over charge value for triadimefon.
pub const TRIADIMEFON_PRECURSOR_MZ: f32 = 300.057;

/// The mass over charge values for triadimefon.
pub const TRIADIMEFON_MZ: [f32; 50] = [
    70.302773, 78.33577, 84.750015, 89.369507, 98.191048, 108.990799, 118.232361, 138.983185,
    155.986084, 166.978302, 167.986237, 174.959946, 180.99411, 182.973282, 185.084991, 187.94249,
    188.227539, 191.996048, 194.001831, 195.009872, 200.108551, 201.947021, 202.950455, 212.002289,
    212.071854, 213.080231, 218.782883, 219.991135, 220.017441, 221.025421, 223.119904, 228.103424,
    233.988068, 234.203796, 239.996872, 248.04866, 249.056717, 256.097961, 260.00293, 263.964691,
    264.068451, 264.079651, 264.192657, 266.899567, 279.990234, 280.010223, 299.916656, 299.980713,
    300.057068, 300.192749,
];
/// The intensities for triadimefon.
pub const TRIADIMEFON_INTENSITIES: [f32; 50] = [
    4157.065918,
    344.904449,
    382.645325,
    353.683868,
    404.364594,
    377.570404,
    368.724701,
    648.020386,
    554.081482,
    3149.577637,
    5330.525391,
    3519.610352,
    603.236267,
    1948.418091,
    4187.558105,
    434.055511,
    353.618225,
    759.776611,
    795.567322,
    4326.125977,
    406.334076,
    11376.140625,
    12839.874023,
    379.256866,
    1749.997559,
    4430.435547,
    363.18808,
    599.544922,
    684.107239,
    17009.701172,
    368.366943,
    1327.268311,
    494.837982,
    1611.129639,
    532.868958,
    6927.643066,
    72495.773438,
    355.05014,
    614.82605,
    653.574707,
    2513.075684,
    456142.6875,
    526.600342,
    444.593781,
    834.63147,
    576.51001,
    561.995605,
    363.804688,
    390160.6875,
    481.738037,
];

impl<S: SpectrumAlloc> TriadimefonSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn triadimefon() -> Self {
        let mut spectrum =
            Self::with_capacity(TRIADIMEFON_PRECURSOR_MZ.into(), TRIADIMEFON_MZ.len());
        for (&mz, &intensity) in TRIADIMEFON_MZ.iter().zip(TRIADIMEFON_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add triadimefon peak to spectrum");
        }
        spectrum
    }
}
