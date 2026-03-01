//! Submodule providing data for thidiazuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of thidiazuron.
pub trait ThidiazuronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of thidiazuron.
    fn thidiazuron() -> Self;
}

/// The precursor mass over charge value for thidiazuron.
pub const THIDIAZURON_PRECURSOR_MZ: f32 = 219.035;

/// The mass over charge values for thidiazuron.
pub const THIDIAZURON_MZ: [f32; 43] = [
    70.966415, 70.96727, 70.982864, 70.984001, 71.990715, 72.840889, 74.862152, 74.927628,
    76.529305, 77.040993, 79.420654, 81.686508, 81.870117, 82.306442, 82.470482, 82.64534,
    83.010712, 89.071709, 99.851089, 99.96949, 99.97139, 99.994621, 99.997467, 100.024132,
    123.171402, 148.448563, 150.592316, 151.253571, 151.256226, 159.746078, 163.01178, 174.124893,
    184.406021, 188.195007, 193.857025, 203.147064, 203.974197, 214.034058, 217.710876, 218.988602,
    227.456863, 229.885483, 231.067001,
];
/// The intensities for thidiazuron.
pub const THIDIAZURON_INTENSITIES: [f32; 43] = [
    9583.515625,
    10803.051758,
    3840831.0,
    18889.859375,
    896601.0625,
    8479.571289,
    7709.653809,
    10724.00293,
    7802.760254,
    8608.639648,
    8099.840332,
    8072.421387,
    8061.723145,
    8043.350586,
    8012.317383,
    8348.992188,
    8696.895508,
    8162.51123,
    8583.772461,
    19446.578125,
    35583.5,
    85536.90625,
    17898354.0,
    71216.554688,
    10322.253906,
    9032.558594,
    8965.47168,
    22934.09375,
    11531.216797,
    8277.481445,
    9922.274414,
    8296.21582,
    9348.827148,
    9792.426758,
    10912.022461,
    9821.317383,
    8868.973633,
    8466.057617,
    8333.925781,
    9902.670898,
    8584.798828,
    10503.183594,
    7742.956055,
];

super::impl_reference_spectrum!(
    ThidiazuronSpectrum,
    thidiazuron,
    THIDIAZURON_PRECURSOR_MZ,
    THIDIAZURON_MZ,
    THIDIAZURON_INTENSITIES
);
