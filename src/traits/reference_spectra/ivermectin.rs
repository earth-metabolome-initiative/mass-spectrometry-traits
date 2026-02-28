//! Submodule providing data for ivermectin.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of ivermectin.
pub trait IvermectinSpectrum: SpectrumAlloc {
    /// Create a new spectrum of ivermectin.
    fn ivermectin() -> Self;
}

/// The precursor mass over charge value for ivermectin.
pub const IVERMECTIN_PRECURSOR_MZ: f32 = 919.506;

/// The mass over charge values for ivermectin.
pub const IVERMECTIN_MZ: [f32; 50] = [
    100.933121, 101.938957, 109.029472, 110.040298, 113.098145, 115.049286, 135.039154, 179.066971,
    185.157364, 197.094849, 219.186584, 221.141083, 224.444107, 228.723587, 229.108307, 230.112747,
    276.282532, 282.975677, 307.004669, 325.252563, 334.976868, 369.997314, 376.996643, 425.242401,
    439.25589, 491.991364, 501.986694, 502.98291, 505.00647, 510.992706, 531.309998, 532.306335,
    549.317505, 560.981262, 575.26178, 634.970764, 675.952026, 693.000977, 693.423035, 711.004456,
    761.42511, 781.031616, 829.500305, 830.520752, 837.483337, 855.482666, 856.505432, 873.399414,
    873.508301, 919.033203,
];
/// The intensities for ivermectin.
pub const IVERMECTIN_INTENSITIES: [f32; 50] = [
    156.0, 179.0, 151.0, 402.0, 382.0, 137.0, 236.0, 201.0, 890.0, 233.0, 243.0, 314.0, 140.0,
    161.0, 6203.0, 529.0, 181.0, 194.0, 150.0, 166.0, 243.0, 217.0, 140.0, 353.0, 206.0, 138.0,
    159.0, 179.0, 283.0, 148.0, 1196.0, 220.0, 135.0, 215.0, 244.0, 233.0, 217.0, 147.0, 250.0,
    164.0, 162.0, 145.0, 180.0, 146.0, 1431.0, 1194.0, 416.0, 205.0, 766.0, 185.0,
];

impl<S: SpectrumAlloc> IvermectinSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn ivermectin() -> Self {
        let mut spectrum = Self::with_capacity(IVERMECTIN_PRECURSOR_MZ.into(), IVERMECTIN_MZ.len());
        for (&mz, &intensity) in IVERMECTIN_MZ.iter().zip(IVERMECTIN_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add ivermectin peak to spectrum");
        }
        spectrum
    }
}
