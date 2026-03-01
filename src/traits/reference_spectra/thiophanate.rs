//! Submodule providing data for thiophanate.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of thiophanate.
pub trait ThiophanateSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of thiophanate.
    fn thiophanate() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for thiophanate.
pub const THIOPHANATE_PRECURSOR_MZ: f32 = 341.039;

/// The mass over charge values for thiophanate.
pub const THIOPHANATE_MZ: [f32; 50] = [
    86.986443, 92.995667, 108.991028, 116.99614, 117.045967, 132.056808, 132.059692, 140.749786,
    148.97052, 149.013214, 149.017975, 149.067139, 157.052063, 158.036316, 166.992462, 168.729401,
    168.990601, 170.987701, 174.01329, 188.997131, 189.014999, 190.062637, 204.990021, 209.003235,
    212.734634, 212.997543, 216.989777, 224.050552, 228.98967, 230.986237, 232.984573, 236.995895,
    238.145477, 240.98996, 252.99025, 254.986771, 257.001984, 260.978882, 260.99588, 272.978882,
    272.996002, 277.007812, 280.984863, 300.991516, 320.97937, 320.997101, 340.987061, 341.003479,
    341.022888, 364.669678,
];
/// The intensities for thiophanate.
pub const THIOPHANATE_INTENSITIES: [f32; 50] = [
    447.82666,
    577.938171,
    432.700317,
    401.052795,
    15767.561523,
    38514.898438,
    563.988342,
    706.026367,
    726.748474,
    1148.890381,
    390290.40625,
    1285.395996,
    1396.764404,
    182097.234375,
    707.441711,
    388.054291,
    523.209412,
    602.9599,
    8989.220703,
    556.092468,
    374.637604,
    62244.261719,
    689.014343,
    400.270599,
    374.406433,
    439.754944,
    15063.50293,
    1882.760864,
    413.868439,
    669.006531,
    732.208191,
    11885.15332,
    387.26178,
    1576.243408,
    1491.927246,
    1955.838745,
    3320.415039,
    1926.110474,
    738.143494,
    583.591064,
    607.058899,
    485.955719,
    1423.80957,
    2371.622314,
    1380.622314,
    1603.530884,
    385.436859,
    1651.762085,
    449.770874,
    384.20224,
];

super::impl_reference_spectrum!(
    ThiophanateSpectrum,
    thiophanate,
    THIOPHANATE_PRECURSOR_MZ,
    THIOPHANATE_MZ,
    THIOPHANATE_INTENSITIES
);
