//! Submodule providing data for cyazofamid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of cyazofamid.
pub trait CyazofamidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of cyazofamid.
    fn cyazofamid() -> Self;
}

/// The precursor mass over charge value for cyazofamid.
pub const CYAZOFAMID_PRECURSOR_MZ: f32 = 681.016;

/// The mass over charge values for cyazofamid.
pub const CYAZOFAMID_MZ: [f32; 50] = [
    72.276527, 101.142952, 108.358002, 118.108566, 126.90435, 131.913498, 133.56546, 133.946671,
    136.97052, 137.026413, 137.630234, 141.273163, 148.700806, 149.90625, 152.0233, 161.006393,
    167.012115, 170.984268, 174.017944, 179.831253, 194.020645, 205.031494, 207.540955, 214.028442,
    232.039215, 234.03511, 251.367798, 252.045013, 253.620056, 253.677979, 253.964325, 254.041763,
    254.11232, 254.516266, 254.561096, 255.043594, 256.808594, 256.903259, 258.024353, 258.242188,
    271.921722, 274.048279, 275.047791, 278.046539, 375.958252, 387.940704, 405.964142, 497.024933,
    681.017212, 814.765747,
];
/// The intensities for cyazofamid.
pub const CYAZOFAMID_INTENSITIES: [f32; 50] = [
    191.0, 208.0, 179.0, 157.0, 1913.0, 671.0, 155.0, 137.0, 245.0, 212.0, 137.0, 139.0, 333.0,
    169.0, 169.0, 219.0, 190.0, 203.0, 347.0, 166.0, 3982.0, 584.0, 161.0, 12739.0, 1036.0, 5524.0,
    152.0, 250.0, 188.0, 255.0, 234.0, 156201.0, 150.0, 162.0, 164.0, 513.0, 223.0, 158.0, 371.0,
    263.0, 24420.0, 8897.0, 142.0, 185.0, 177.0, 212.0, 215.0, 173.0, 180.0, 192.0,
];

impl<S: SpectrumAlloc> CyazofamidSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn cyazofamid() -> Self {
        let mut spectrum = Self::with_capacity(CYAZOFAMID_PRECURSOR_MZ.into(), CYAZOFAMID_MZ.len());
        for (&mz, &intensity) in CYAZOFAMID_MZ.iter().zip(CYAZOFAMID_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add cyazofamid peak to spectrum");
        }
        spectrum
    }
}
