//! Submodule providing data for diflubenzuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of diflubenzuron.
pub trait DiflubenzuronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of diflubenzuron.
    fn diflubenzuron() -> Self;
}

/// The precursor mass over charge value for diflubenzuron.
pub const DIFLUBENZURON_PRECURSOR_MZ: f32 = 309.025;

/// The mass over charge values for diflubenzuron.
pub const DIFLUBENZURON_MZ: [f32; 34] = [
    70.499237, 75.340157, 78.74511, 81.121902, 82.531464, 83.066193, 93.014458, 104.146309,
    112.22921, 113.021111, 113.023102, 126.011887, 131.103745, 137.098724, 142.201431, 143.305588,
    151.006912, 156.026749, 160.75975, 188.774979, 196.305222, 210.064194, 211.907166, 221.498367,
    223.56076, 246.012863, 251.127548, 265.281525, 267.611328, 274.779297, 288.99176, 289.018494,
    309.025085, 309.150269,
];
/// The intensities for diflubenzuron.
pub const DIFLUBENZURON_INTENSITIES: [f32; 34] = [
    11121.253906,
    11368.794922,
    11375.603516,
    12181.450195,
    11538.429688,
    11395.354492,
    371775.28125,
    11710.199219,
    13030.732422,
    1364730.875,
    29283.671875,
    579879.0,
    13576.740234,
    13550.517578,
    12900.200195,
    12049.810547,
    1268025.375,
    741181.5625,
    12715.168945,
    12298.321289,
    13737.850586,
    13710.392578,
    12777.15332,
    12650.871094,
    12443.246094,
    58054.746094,
    11989.155273,
    12279.414062,
    12599.55957,
    16009.441406,
    15209.018555,
    988983.75,
    114719.546875,
    13360.837891,
];

impl<S: SpectrumAlloc> DiflubenzuronSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn diflubenzuron() -> Self {
        let mut spectrum =
            Self::with_capacity(DIFLUBENZURON_PRECURSOR_MZ.into(), DIFLUBENZURON_MZ.len());
        for (&mz, &intensity) in DIFLUBENZURON_MZ
            .iter()
            .zip(DIFLUBENZURON_INTENSITIES.iter())
        {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add diflubenzuron peak to spectrum");
        }
        spectrum
    }
}
