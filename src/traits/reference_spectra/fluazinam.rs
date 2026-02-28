//! Submodule providing data for fluazinam.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of fluazinam.
pub trait FluazinamSpectrum: SpectrumAlloc {
    /// Create a new spectrum of fluazinam.
    fn fluazinam() -> Self;
}

/// The precursor mass over charge value for fluazinam.
pub const FLUAZINAM_PRECURSOR_MZ: f32 = 462.944;

/// The mass over charge values for fluazinam.
pub const FLUAZINAM_MZ: [f32; 37] = [
    123.047623, 130.992355, 155.768143, 184.581833, 201.031723, 202.817169, 204.347488, 206.870407,
    222.141113, 232.208725, 237.440399, 274.009949, 304.974701, 306.973755, 311.170074, 334.97818,
    337.96521, 339.973846, 353.936432, 353.98761, 365.937592, 366.956696, 368.967621, 369.950836,
    373.958344, 377.986816, 379.966064, 381.987488, 385.940369, 386.958038, 392.99173, 397.97818,
    415.942413, 416.955811, 426.578033, 426.960419, 462.945862,
];
/// The intensities for fluazinam.
pub const FLUAZINAM_INTENSITIES: [f32; 37] = [
    123.0, 175.0, 70.0, 127.0, 160.0, 103.0, 63.0, 217.0, 93.0, 292.0, 171.0, 183.0, 160.0, 82.0,
    98.0, 526.0, 160.0, 247.0, 164.0, 146.0, 236.0, 83.0, 133.0, 712.0, 97.0, 116.0, 731.0, 247.0,
    250.0, 690.0, 230.0, 6796.0, 18042.0, 146.0, 173.0, 195.0, 5805.0,
];

impl<S: SpectrumAlloc> FluazinamSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn fluazinam() -> Self {
        let mut spectrum = Self::with_capacity(FLUAZINAM_PRECURSOR_MZ.into(), FLUAZINAM_MZ.len());
        for (&mz, &intensity) in FLUAZINAM_MZ.iter().zip(FLUAZINAM_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add fluazinam peak to spectrum");
        }
        spectrum
    }
}
