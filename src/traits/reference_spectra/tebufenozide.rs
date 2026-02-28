//! Submodule providing data for tebufenozide.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of tebufenozide.
pub trait TebufenozideSpectrum: SpectrumAlloc {
    /// Create a new spectrum of tebufenozide.
    fn tebufenozide() -> Self;
}

/// The precursor mass over charge value for tebufenozide.
pub const TEBUFENOZIDE_PRECURSOR_MZ: f32 = 397.213;

/// The mass over charge values for tebufenozide.
pub const TEBUFENOZIDE_MZ: [f32; 16] = [
    75.623543, 105.074944, 129.142441, 130.926895, 148.275497, 148.784241, 148.905792, 149.059143,
    150.060501, 204.139206, 294.136505, 349.465271, 349.623169, 351.016052, 351.207153, 352.167999,
];
/// The intensities for tebufenozide.
pub const TEBUFENOZIDE_INTENSITIES: [f32; 16] = [
    120.0, 127.0, 99.0, 119.0, 234.0, 309.0, 119.0, 10118.0, 127.0, 199.0, 183.0, 232.0, 266.0,
    155.0, 10642.0, 178.0,
];

impl<S: SpectrumAlloc> TebufenozideSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn tebufenozide() -> Self {
        let mut spectrum =
            Self::with_capacity(TEBUFENOZIDE_PRECURSOR_MZ.into(), TEBUFENOZIDE_MZ.len());
        for (&mz, &intensity) in TEBUFENOZIDE_MZ.iter().zip(TEBUFENOZIDE_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add tebufenozide peak to spectrum");
        }
        spectrum
    }
}
