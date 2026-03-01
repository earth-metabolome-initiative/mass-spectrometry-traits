//! Submodule providing data for doramectin.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of doramectin.
pub trait DoramectinSpectrum: SpectrumAlloc {
    /// Create a new spectrum of doramectin.
    fn doramectin() -> Self;
}

/// The precursor mass over charge value for doramectin.
pub const DORAMECTIN_PRECURSOR_MZ: f32 = 943.506;

/// The mass over charge values for doramectin.
pub const DORAMECTIN_MZ: [f32; 50] = [
    84.022591, 85.029709, 97.065849, 107.052368, 109.029823, 111.045212, 116.92897, 123.080002,
    125.06662, 133.33197, 135.03952, 137.058182, 149.097961, 153.052826, 159.079727, 161.058533,
    197.097488, 211.115005, 214.061264, 216.120178, 223.075851, 225.092972, 229.108185, 230.112228,
    241.090408, 254.984344, 260.103455, 329.208252, 345.145752, 361.17746, 381.230774, 443.21991,
    555.323181, 556.308716, 573.316711, 575.268738, 591.337036, 592.328796, 785.421509, 796.950684,
    805.997986, 853.519104, 854.523315, 861.481567, 862.477539, 879.492065, 879.575623, 880.505493,
    897.505188, 898.514465,
];
/// The intensities for doramectin.
pub const DORAMECTIN_INTENSITIES: [f32; 50] = [
    1269.0, 955.0, 407.0, 426.0, 1058.0, 780.0, 1129.0, 542.0, 313.0, 326.0, 678.0, 460.0, 513.0,
    519.0, 313.0, 348.0, 506.0, 337.0, 309.0, 379.0, 506.0, 367.0, 12778.0, 580.0, 645.0, 315.0,
    356.0, 404.0, 788.0, 385.0, 306.0, 2329.0, 1209.0, 406.0, 580.0, 1175.0, 519.0, 311.0, 412.0,
    291.0, 338.0, 661.0, 325.0, 2866.0, 1302.0, 4955.0, 418.0, 846.0, 1725.0, 616.0,
];

super::impl_reference_spectrum!(
    DoramectinSpectrum,
    doramectin,
    DORAMECTIN_PRECURSOR_MZ,
    DORAMECTIN_MZ,
    DORAMECTIN_INTENSITIES
);
