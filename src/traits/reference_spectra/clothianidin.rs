//! Submodule providing data for clothianidin.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of clothianidin.
pub trait ClothianidinSpectrum: SpectrumAlloc {
    /// Create a new spectrum of clothianidin.
    fn clothianidin() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for clothianidin.
pub const CLOTHIANIDIN_PRECURSOR_MZ: f32 = 248.002;

/// The mass over charge values for clothianidin.
pub const CLOTHIANIDIN_MZ: [f32; 50] = [
    78.642494, 85.999237, 91.125984, 92.024948, 95.024803, 96.684891, 97.85347, 110.012215,
    111.002266, 111.230286, 113.017715, 117.949081, 117.952423, 119.036331, 122.036064, 122.98951,
    125.005295, 125.989098, 132.044296, 133.052109, 134.311462, 135.997314, 137.005539, 138.013092,
    139.02092, 139.026398, 141.012787, 146.851135, 148.993027, 150.000793, 150.00386, 151.00824,
    157.192734, 157.412628, 157.710434, 159.424927, 163.008438, 164.016418, 165.024124, 166.03212,
    167.17746, 168.023834, 178.615234, 182.853668, 195.022217, 202.129517, 212.025177, 214.537796,
    243.049667, 248.002258,
];
/// The intensities for clothianidin.
pub const CLOTHIANIDIN_INTENSITIES: [f32; 50] = [
    5003.942871,
    35022.433594,
    4930.152832,
    19078.367188,
    161607.078125,
    4993.806641,
    5664.941406,
    6654.39209,
    615117.875,
    5291.260742,
    6206.6875,
    5215.691406,
    164842.84375,
    17764.019531,
    104174.695312,
    11008.191406,
    9643.411133,
    8555.709961,
    21764.166016,
    20840.316406,
    4967.373535,
    25974.339844,
    10529.207031,
    30837.345703,
    8801.631836,
    132262.625,
    11421.364258,
    5391.549805,
    37200.746094,
    509912.09375,
    6075.168945,
    165163.65625,
    5009.054688,
    5255.186035,
    5143.043945,
    5072.960449,
    9085.204102,
    6428.61084,
    1191399.625,
    177904.640625,
    4906.07959,
    130685.304688,
    5293.277344,
    5088.202148,
    109545.125,
    5084.004395,
    84454.351562,
    5514.27002,
    6096.865234,
    315305.375,
];

super::impl_reference_spectrum!(
    ClothianidinSpectrum,
    clothianidin,
    CLOTHIANIDIN_PRECURSOR_MZ,
    CLOTHIANIDIN_MZ,
    CLOTHIANIDIN_INTENSITIES
);
