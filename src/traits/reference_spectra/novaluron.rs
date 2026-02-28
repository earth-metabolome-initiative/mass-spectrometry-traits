//! Submodule providing data for novaluron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of novaluron.
pub trait NovaluronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of novaluron.
    fn novaluron() -> Self;
}

/// The precursor mass over charge value for novaluron.
pub const NOVALURON_PRECURSOR_MZ: f32 = 491.005;

/// The mass over charge values for novaluron.
pub const NOVALURON_MZ: [f32; 50] = [
    84.969307, 84.990303, 85.010551, 93.014435, 103.555786, 110.041183, 113.021103, 113.023094,
    134.028748, 136.020599, 140.998825, 156.002716, 156.026733, 157.466446, 162.982712, 165.348785,
    165.994003, 167.986053, 182.98912, 216.046967, 216.253448, 216.757324, 236.971664, 251.026581,
    260.015839, 260.999695, 262.007751, 269.03714, 278.592285, 287.00293, 305.014526, 307.596436,
    307.983154, 307.992126, 332.987549, 364.995331, 372.703644, 385.001587, 427.993896, 431.086182,
    435.022552, 439.342621, 445.900879, 448.000427, 450.992493, 455.030304, 459.809204, 470.998352,
    490.729553, 491.005493,
];
/// The intensities for novaluron.
pub const NOVALURON_INTENSITIES: [f32; 50] = [
    13026.161133,
    5400955.0,
    10255.760742,
    178947.71875,
    9381.606445,
    12289.570312,
    817678.0625,
    15664.736328,
    9505.320312,
    17342.798828,
    497461.15625,
    12099.375,
    645178.9375,
    10575.383789,
    435713.28125,
    9041.619141,
    363986.21875,
    295360.53125,
    12822.868164,
    16251.597656,
    10552.486328,
    9307.140625,
    10221.396484,
    62881.023438,
    40595.433594,
    60514.617188,
    31311.748047,
    50265.117188,
    9827.049805,
    16629.833984,
    2174646.75,
    9664.56543,
    15560.444336,
    590298.3125,
    652807.0,
    9168.759766,
    10730.80957,
    16927.640625,
    97644.40625,
    9834.771484,
    15975.753906,
    11094.994141,
    10158.78125,
    33244.886719,
    38844.117188,
    15257.545898,
    9613.751953,
    1648197.5,
    9020.766602,
    9067.113281,
];

impl<S: SpectrumAlloc> NovaluronSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn novaluron() -> Self {
        let mut spectrum = Self::with_capacity(NOVALURON_PRECURSOR_MZ.into(), NOVALURON_MZ.len());
        for (&mz, &intensity) in NOVALURON_MZ.iter().zip(NOVALURON_INTENSITIES.iter()) {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add novaluron peak to spectrum");
        }
        spectrum
    }
}
