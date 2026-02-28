//! Submodule providing data for acephate.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of acephate.
pub trait AcephateSpectrum: SpectrumAlloc {
    /// Create a new spectrum of acephate.
    fn acephate() -> Self;
}

/// The precursor mass over charge value for acephate.
pub const ACEPHATE_PRECURSOR_MZ: f32 = 182.005;

/// The mass over charge values for acephate.
pub const ACEPHATE_MZ: [f32; 50] = [
    78.958755, 81.704353, 87.040894, 87.876038, 88.577194, 91.18782, 92.919777, 92.993736,
    93.920334, 93.997002, 94.916969, 94.918274, 94.99025, 100.973427, 101.975006, 103.12278,
    103.990677, 103.992668, 104.308235, 123.348885, 130.799835, 133.986374, 134.001511, 136.017151,
    136.910019, 136.983749, 137.910385, 137.913452, 137.987183, 138.019897, 138.056046, 138.871277,
    138.906982, 139.994141, 140.978134, 146.868744, 149.140945, 149.581161, 151.243408, 151.81311,
    151.957809, 155.882568, 159.451523, 175.893219, 181.837662, 181.977295, 182.004684, 182.009323,
    184.108841, 188.843765,
];
/// The intensities for acephate.
pub const ACEPHATE_INTENSITIES: [f32; 50] = [
    59449.738281,
    219.397141,
    214.359955,
    216.880508,
    226.319199,
    245.073746,
    1104.349243,
    446.473267,
    965.867432,
    526.465698,
    3014.461182,
    300.671417,
    10583.603516,
    217.857315,
    5877.550293,
    297.563965,
    48730.992188,
    567.668152,
    222.421265,
    230.701828,
    228.800705,
    249.415314,
    3393.503418,
    33200.535156,
    411.487793,
    365.019684,
    265.832611,
    289.602356,
    2151.223877,
    361.831451,
    297.771484,
    270.085266,
    539.579224,
    476.601257,
    107802.96875,
    876.013489,
    240.698868,
    247.65918,
    236.902542,
    250.818054,
    380.134338,
    297.188293,
    228.095322,
    294.967224,
    379.33847,
    1082.812622,
    30183.724609,
    524.393066,
    284.966248,
    277.326111,
];

impl<S: SpectrumAlloc> AcephateSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn acephate() -> Self {
        let mut spectrum = Self::with_capacity(ACEPHATE_PRECURSOR_MZ.into(), ACEPHATE_MZ.len());
        for (&mz, &intensity) in ACEPHATE_MZ.iter().zip(ACEPHATE_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add acephate peak to spectrum");
        }
        spectrum
    }
}
