//! Submodule providing data for aspartic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of aspartic acid.
pub trait AsparticAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of aspartic acid.
    fn aspartic_acid() -> Self;
}

/// The precursor mass over charge value for aspartic acid.
pub const ASPARTIC_ACID_PRECURSOR_MZ: f32 = 134.045;

/// The mass over charge values for aspartic acid.
pub const ASPARTIC_ACID_MZ: [f32; 14] = [
    56.367771, 61.029579, 70.030296, 73.229965, 74.025749, 75.028831, 79.023346, 88.042374,
    89.045036, 109.839478, 116.039391, 134.05101, 135.053543, 149.681931,
];
/// The intensities for aspartic acid.
pub const ASPARTIC_ACID_INTENSITIES: [f32; 14] = [
    52688.234375,
    54141.765625,
    561100.75,
    41380.050781,
    32275294.0,
    249212.140625,
    46146.296875,
    24659428.0,
    273908.6875,
    25265.826172,
    3457162.75,
    2983189.75,
    41962.378906,
    27315.875,
];

impl<S: SpectrumAlloc> AsparticAcidSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn aspartic_acid() -> Self {
        let mut spectrum =
            Self::with_capacity(ASPARTIC_ACID_PRECURSOR_MZ.into(), ASPARTIC_ACID_MZ.len());
        for (&mz, &intensity) in ASPARTIC_ACID_MZ
            .iter()
            .zip(ASPARTIC_ACID_INTENSITIES.iter())
        {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add aspartic acid peak to spectrum");
        }
        spectrum
    }
}
