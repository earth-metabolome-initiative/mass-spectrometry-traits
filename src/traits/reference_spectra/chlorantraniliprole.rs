//! Submodule providing data for chlorantraniliprole.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of chlorantraniliprole.
pub trait ChlorantraniliproleSpectrum: SpectrumAlloc {
    /// Create a new spectrum of chlorantraniliprole.
    fn chlorantraniliprole() -> Self;
}

/// The precursor mass over charge value for chlorantraniliprole.
pub const CHLORANTRANILIPROLE_PRECURSOR_MZ: f32 = 479.964;

/// The mass over charge values for chlorantraniliprole.
pub const CHLORANTRANILIPROLE_MZ: [f32; 50] = [
    70.171921, 70.367195, 72.251167, 73.754585, 78.899437, 78.918396, 78.937012, 80.831085,
    87.608498, 91.537224, 94.503052, 95.038841, 96.783012, 100.258865, 119.476387, 127.990814,
    127.993217, 143.30072, 144.940994, 148.051834, 157.930893, 158.319427, 165.940292, 166.00676,
    188.931305, 201.884369, 201.962631, 202.038193, 202.859238, 215.03862, 222.12822, 223.02832,
    223.033005, 232.027954, 236.3992, 251.669174, 272.059418, 293.02359, 296.358307, 298.909241,
    300.671875, 305.109375, 308.037323, 312.210968, 337.933838, 350.757385, 350.9664, 386.965302,
    446.247223, 483.708984,
];
/// The intensities for chlorantraniliprole.
pub const CHLORANTRANILIPROLE_INTENSITIES: [f32; 50] = [
    2795.068115,
    2720.832764,
    2728.957764,
    2833.638672,
    4008.813965,
    1859386.0,
    4990.959961,
    3899.562012,
    2871.988281,
    2999.245361,
    3009.175293,
    2977.292236,
    3055.749023,
    6326.728516,
    2941.19043,
    17978.957031,
    2811.181152,
    2827.105225,
    1605849.75,
    5811.536133,
    3731.634033,
    3188.862061,
    3564.800049,
    56729.847656,
    5378.251953,
    5350.80127,
    3025822.75,
    5411.795898,
    3321.483643,
    3268.66626,
    3031.274658,
    271538.375,
    4053.812012,
    4947.661133,
    3416.256104,
    2913.366699,
    4357.493652,
    4236.173828,
    2944.219482,
    2980.716064,
    3468.368164,
    3124.975342,
    19580.460938,
    3038.205322,
    5392.540039,
    3083.024902,
    105626.820312,
    5039.891602,
    3211.460693,
    3283.13916,
];

impl<S: SpectrumAlloc> ChlorantraniliproleSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn chlorantraniliprole() -> Self {
        let mut spectrum = Self::with_capacity(
            CHLORANTRANILIPROLE_PRECURSOR_MZ.into(),
            CHLORANTRANILIPROLE_MZ.len(),
        );
        for (&mz, &intensity) in CHLORANTRANILIPROLE_MZ
            .iter()
            .zip(CHLORANTRANILIPROLE_INTENSITIES.iter())
        {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add chlorantraniliprole peak to spectrum");
        }
        spectrum
    }
}
