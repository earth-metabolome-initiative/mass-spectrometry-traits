//! Submodule providing data for avermectin.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of avermectin.
pub trait AvermectinSpectrum: SpectrumAlloc {
    /// Create a new spectrum of avermectin.
    fn avermectin() -> Self;
}

/// The precursor mass over charge value for avermectin.
pub const AVERMECTIN_PRECURSOR_MZ: f32 = 917.49;

/// The mass over charge values for avermectin.
pub const AVERMECTIN_MZ: [f32; 50] = [
    83.0495, 84.02314, 85.029549, 96.017845, 119.051979, 121.028343, 137.054504, 159.038528,
    171.078308, 179.070099, 179.263229, 188.696671, 197.102463, 217.122101, 223.077896, 227.116226,
    229.108063, 230.11586, 243.14035, 255.128128, 266.982697, 271.135803, 271.226715, 317.203918,
    319.170013, 342.971558, 345.15274, 399.23175, 443.224274, 444.229187, 455.285706, 497.020172,
    511.285522, 529.296509, 540.984558, 544.996399, 547.289551, 562.986694, 565.314514, 575.266602,
    759.411133, 761.447632, 761.480957, 827.496216, 835.466858, 836.47522, 853.022217, 853.476135,
    854.469482, 871.493652,
];
/// The intensities for avermectin.
pub const AVERMECTIN_INTENSITIES: [f32; 50] = [
    242.0, 671.0, 1088.0, 274.0, 303.0, 196.0, 210.0, 240.0, 211.0, 207.0, 403.0, 257.0, 405.0,
    234.0, 822.0, 203.0, 17308.0, 509.0, 212.0, 574.0, 219.0, 223.0, 306.0, 307.0, 224.0, 267.0,
    410.0, 294.0, 2327.0, 289.0, 210.0, 218.0, 637.0, 2679.0, 279.0, 241.0, 527.0, 219.0, 858.0,
    370.0, 451.0, 223.0, 379.0, 811.0, 3268.0, 1692.0, 209.0, 3958.0, 594.0, 1400.0,
];

impl<S: SpectrumAlloc> AvermectinSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn avermectin() -> Self {
        let mut spectrum = Self::with_capacity(AVERMECTIN_PRECURSOR_MZ.into(), AVERMECTIN_MZ.len());
        for (&mz, &intensity) in AVERMECTIN_MZ.iter().zip(AVERMECTIN_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add avermectin peak to spectrum");
        }
        spectrum
    }
}
