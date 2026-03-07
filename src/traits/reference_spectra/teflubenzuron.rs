//! Submodule providing data for teflubenzuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of teflubenzuron.
pub trait TeflubenzuronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of teflubenzuron.
    fn teflubenzuron() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for teflubenzuron.
pub const TEFLUBENZURON_PRECURSOR_MZ: f32 = 378.967;

/// The mass over charge values for teflubenzuron.
pub const TEFLUBENZURON_MZ: [f32; 50] = [
    73.007629, 86.825081, 88.434479, 93.014229, 102.833138, 103.805115, 104.114792, 106.080719,
    113.020836, 113.022507, 127.510056, 132.034164, 139.963959, 140.341675, 146.961319, 150.896942,
    156.026733, 174.956116, 194.589447, 195.879333, 195.883224, 195.954086, 196.026337, 218.954071,
    219.360291, 220.94902, 221.933517, 235.545593, 242.943573, 258.987915, 262.157654, 262.958679,
    265.524475, 267.954254, 278.99408, 279.979065, 280.969299, 286.400848, 290.954193, 290.987152,
    294.9646, 302.978394, 315.955353, 318.949158, 320.988678, 322.985504, 335.962555, 338.955078,
    358.961243, 378.968323,
];
/// The intensities for teflubenzuron.
pub const TEFLUBENZURON_INTENSITIES: [f32; 50] = [
    3645.952393,
    3713.89624,
    3410.864014,
    84562.882812,
    3215.754395,
    3257.21875,
    3405.375488,
    3481.137451,
    327996.53125,
    4145.049805,
    14238.692383,
    3398.15918,
    3563.853027,
    3511.967285,
    28558.986328,
    3541.074463,
    239480.78125,
    76029.515625,
    3500.026855,
    10929.040039,
    4094.213623,
    5531751.5,
    12884.327148,
    4437.442383,
    3440.193115,
    589867.25,
    93692.570312,
    3740.106445,
    6672.341797,
    39836.71875,
    3476.93457,
    23881.501953,
    3548.19458,
    37580.046875,
    22573.322266,
    5874.236328,
    19073.46875,
    3931.129639,
    13386.59375,
    5530.916992,
    586467.4375,
    5435.673828,
    391211.46875,
    81778.539062,
    53026.9375,
    32962.582031,
    15988.009766,
    1741611.625,
    29980.107422,
    17804.919922,
];

super::impl_reference_spectrum!(
    TeflubenzuronSpectrum,
    teflubenzuron,
    TEFLUBENZURON_PRECURSOR_MZ,
    TEFLUBENZURON_MZ,
    TEFLUBENZURON_INTENSITIES
);
