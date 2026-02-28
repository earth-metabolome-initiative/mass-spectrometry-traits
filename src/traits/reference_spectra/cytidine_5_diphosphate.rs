//! Submodule providing data for cytidine 5 diphosphate.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of cytidine 5 diphosphate.
pub trait Cytidine5DiphosphateSpectrum: SpectrumAlloc {
    /// Create a new spectrum of cytidine 5 diphosphate.
    fn cytidine_5_diphosphate() -> Self;
}

/// The precursor mass over charge value for cytidine 5 diphosphate.
pub const CYTIDINE_5_DIPHOSPHATE_PRECURSOR_MZ: f32 = 404.02;

/// The mass over charge values for cytidine 5 diphosphate.
pub const CYTIDINE_5_DIPHOSPHATE_MZ: [f32; 15] = [
    60.055748, 69.042969, 75.107704, 83.050262, 95.027405, 97.031845, 112.054977, 113.056587,
    192.026703, 208.085556, 214.521286, 226.098831, 288.063721, 306.075653, 404.07074,
];
/// The intensities for cytidine 5 diphosphate.
pub const CYTIDINE_5_DIPHOSPHATE_INTENSITIES: [f32; 15] = [
    7449.517578,
    41024.453125,
    7897.96875,
    8260.068359,
    102957.789062,
    1933484.625,
    36699864.0,
    178665.34375,
    51846.46875,
    386645.4375,
    8433.279297,
    42665.246094,
    69798.84375,
    68617.414062,
    212609.28125,
];

impl<S: SpectrumAlloc> Cytidine5DiphosphateSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn cytidine_5_diphosphate() -> Self {
        let mut spectrum = Self::with_capacity(
            CYTIDINE_5_DIPHOSPHATE_PRECURSOR_MZ.into(),
            CYTIDINE_5_DIPHOSPHATE_MZ.len(),
        );
        for (&mz, &intensity) in CYTIDINE_5_DIPHOSPHATE_MZ
            .iter()
            .zip(CYTIDINE_5_DIPHOSPHATE_INTENSITIES.iter())
        {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add cytidine 5 diphosphate peak to spectrum");
        }
        spectrum
    }
}
