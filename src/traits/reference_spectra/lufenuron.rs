//! Submodule providing data for lufenuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of lufenuron.
pub trait LufenuronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of lufenuron.
    fn lufenuron() -> Self;
}

/// The precursor mass over charge value for lufenuron.
pub const LUFENURON_PRECURSOR_MZ: f32 = 508.971;

/// The mass over charge values for lufenuron.
pub const LUFENURON_MZ: [f32; 50] = [
    83.094193, 93.014374, 108.926857, 113.020821, 121.875069, 130.992722, 145.891113, 146.987488,
    156.026733, 165.970261, 173.952332, 174.898483, 174.95993, 174.96431, 175.023056, 175.967987,
    185.95253, 199.955536, 201.94693, 203.983917, 219.953949, 222.68425, 231.997513, 249.969818,
    258.000366, 265.939362, 267.021545, 269.975067, 272.113617, 275.004059, 285.946167, 287.950714,
    294.96109, 301.990356, 302.998413, 305.952728, 325.95816, 331.932312, 338.974396, 350.953674,
    351.937958, 397.513672, 408.999451, 432.981384, 445.958893, 452.988678, 456.893433, 462.733948,
    468.958923, 488.965851,
];
/// The intensities for lufenuron.
pub const LUFENURON_INTENSITIES: [f32; 50] = [
    12338.117188,
    112268.96875,
    34422.703125,
    381861.3125,
    11047.966797,
    235026.453125,
    12405.919922,
    609210.375,
    173360.390625,
    193226.5625,
    41766.101562,
    21734.363281,
    11439488.0,
    13762.734375,
    17124.392578,
    18367.929688,
    42960.230469,
    412564.84375,
    3361856.75,
    34242.398438,
    40939.828125,
    11937.34375,
    66427.195312,
    22299.386719,
    236998.78125,
    134320.65625,
    43797.878906,
    12022.984375,
    11561.140625,
    49353.222656,
    338698.71875,
    11017.478516,
    100681.320312,
    61592.75,
    729664.5,
    283924.0,
    6500512.0,
    20639.556641,
    2801203.5,
    496474.46875,
    38889.890625,
    11328.057617,
    46830.757812,
    17416.542969,
    43903.425781,
    346271.59375,
    11212.665039,
    11331.806641,
    15838.735352,
    51068.511719,
];

impl<S: SpectrumAlloc> LufenuronSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn lufenuron() -> Self {
        let mut spectrum = Self::with_capacity(LUFENURON_PRECURSOR_MZ.into(), LUFENURON_MZ.len());
        for (&mz, &intensity) in LUFENURON_MZ.iter().zip(LUFENURON_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add lufenuron peak to spectrum");
        }
        spectrum
    }
}
