//! Submodule providing data for flufenoxuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of flufenoxuron.
pub trait FlufenoxuronSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of flufenoxuron.
    fn flufenoxuron() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for flufenoxuron.
pub const FLUFENOXURON_PRECURSOR_MZ: f32 = 487.029;

/// The mass over charge values for flufenoxuron.
pub const FLUFENOXURON_MZ: [f32; 50] = [
    76.018776, 91.927444, 93.014389, 95.013611, 104.014107, 113.02108, 113.023102, 119.372086,
    122.024864, 124.020393, 137.359085, 150.023804, 156.027161, 156.030457, 156.306168, 163.707993,
    165.774155, 178.988159, 185.008514, 194.983231, 219.599365, 220.038345, 222.9785, 238.048416,
    243.622101, 248.033371, 249.017593, 256.015381, 264.004364, 265.824371, 269.036743, 278.159454,
    282.625061, 283.002625, 284.010132, 289.044098, 289.121124, 303.503143, 304.009064, 304.016418,
    327.017181, 329.01178, 330.08728, 356.18454, 367.049469, 411.040894, 447.017273, 467.024139,
    481.574249, 487.030762,
];
/// The intensities for flufenoxuron.
pub const FLUFENOXURON_INTENSITIES: [f32; 50] = [
    41127.238281,
    4461.829102,
    65007.410156,
    36654.273438,
    209602.578125,
    257369.1875,
    6772.396973,
    3626.891357,
    4512.128418,
    52496.152344,
    4091.438232,
    85253.367188,
    308835.34375,
    3973.61084,
    3559.456543,
    3732.536133,
    4461.603516,
    181538.59375,
    3598.356934,
    203374.625,
    4120.1875,
    4937.722656,
    13033.578125,
    28174.111328,
    4103.203125,
    25966.625,
    21813.578125,
    19856.234375,
    3919.273193,
    3649.607178,
    38259.664062,
    3739.652588,
    4016.87915,
    46484.15625,
    83609.070312,
    53056.960938,
    3595.767578,
    5062.706055,
    3955.464355,
    625107.5625,
    4542.858887,
    434703.84375,
    4405.793945,
    3932.313721,
    6917.507324,
    104147.460938,
    90002.015625,
    28816.701172,
    3803.714844,
    8503.919922,
];

super::impl_reference_spectrum!(
    FlufenoxuronSpectrum,
    flufenoxuron,
    FLUFENOXURON_PRECURSOR_MZ,
    FLUFENOXURON_MZ,
    FLUFENOXURON_INTENSITIES
);
