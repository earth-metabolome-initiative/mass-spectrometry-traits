//! Submodule providing data for adenosine 5 monophosphate.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of adenosine 5 monophosphate.
pub trait Adenosine5MonophosphateSpectrum: SpectrumAlloc {
    /// Create a new spectrum of adenosine 5 monophosphate.
    fn adenosine_5_monophosphate() -> Self;
}

/// The precursor mass over charge value for adenosine 5 monophosphate.
pub const ADENOSINE_5_MONOPHOSPHATE_PRECURSOR_MZ: f32 = 348.071;

/// The mass over charge values for adenosine 5 monophosphate.
pub const ADENOSINE_5_MONOPHOSPHATE_MZ: [f32; 16] = [
    53.16864, 54.022614, 55.029789, 56.351715, 67.030807, 72.564934, 78.628639, 97.031715,
    119.039558, 136.067947, 137.071198, 178.083191, 232.09903, 250.113251, 348.10614, 349.103241,
];
/// The intensities for adenosine 5 monophosphate.
pub const ADENOSINE_5_MONOPHOSPHATE_INTENSITIES: [f32; 16] = [
    183522.75,
    77341.0,
    327396.3125,
    94952.304688,
    185159.515625,
    167656.46875,
    108692.148438,
    17950362.0,
    148895.328125,
    363874208.0,
    1917089.5,
    111951.851562,
    147844.703125,
    1312412.25,
    29370612.0,
    537890.4375,
];

impl<S: SpectrumAlloc> Adenosine5MonophosphateSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn adenosine_5_monophosphate() -> Self {
        let mut spectrum = Self::with_capacity(
            ADENOSINE_5_MONOPHOSPHATE_PRECURSOR_MZ.into(),
            ADENOSINE_5_MONOPHOSPHATE_MZ.len(),
        );
        for (&mz, &intensity) in ADENOSINE_5_MONOPHOSPHATE_MZ
            .iter()
            .zip(ADENOSINE_5_MONOPHOSPHATE_INTENSITIES.iter())
        {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add adenosine 5 monophosphate peak to spectrum");
        }
        spectrum
    }
}
