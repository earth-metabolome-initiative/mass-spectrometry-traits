//! Submodule providing data for triflumuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of triflumuron.
pub trait TriflumuronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of triflumuron.
    fn triflumuron() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for triflumuron.
pub const TRIFLUMURON_PRECURSOR_MZ: f32 = 357.026;

/// The mass over charge values for triflumuron.
pub const TRIFLUMURON_MZ: [f32; 47] = [
    72.815125, 73.359489, 76.595413, 79.027885, 81.838531, 84.990295, 90.248634, 93.969177,
    97.547714, 106.029968, 107.03775, 107.199173, 108.57618, 110.041199, 116.264999, 119.208633,
    121.417252, 135.039307, 141.214264, 147.499725, 148.209137, 152.356674, 153.954041, 153.956757,
    154.006607, 154.058273, 158.906799, 161.916306, 163.944672, 176.032806, 179.881943, 186.695587,
    190.733627, 201.028641, 214.737564, 251.343781, 269.077972, 269.81424, 284.895416, 300.026276,
    314.020233, 321.049011, 324.972351, 344.995422, 345.826599, 356.997284, 357.026062,
];
/// The intensities for triflumuron.
pub const TRIFLUMURON_INTENSITIES: [f32; 47] = [
    6362.656738,
    6575.768066,
    7062.664062,
    6684.683594,
    6576.907227,
    2102948.5,
    6320.805664,
    7144.137207,
    6882.875488,
    10862.045898,
    41890.042969,
    6163.004395,
    6668.574707,
    181349.265625,
    6284.269043,
    5944.505371,
    6119.970215,
    7162.72168,
    6824.635254,
    6818.415527,
    6701.348145,
    6905.377441,
    8198.143555,
    9424.683594,
    6386001.5,
    11903.78125,
    6439.624512,
    6136.927246,
    7397.185059,
    2007122.875,
    7245.501465,
    6082.802246,
    6918.004395,
    27225.001953,
    7512.961426,
    7538.822266,
    8391.760742,
    6015.51416,
    7475.662598,
    7593.951172,
    124276.195312,
    10235.746094,
    6372.055664,
    7290.296387,
    6791.981445,
    10512.073242,
    112693.210938,
];

super::impl_reference_spectrum!(
    TriflumuronSpectrum,
    triflumuron,
    TRIFLUMURON_PRECURSOR_MZ,
    TRIFLUMURON_MZ,
    TRIFLUMURON_INTENSITIES
);
