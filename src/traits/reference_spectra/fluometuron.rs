//! Submodule providing data for fluometuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of fluometuron.
pub trait FluometuronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of fluometuron.
    fn fluometuron() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for fluometuron.
pub const FLUOMETURON_PRECURSOR_MZ: f32 = 231.075;

/// The mass over charge values for fluometuron.
pub const FLUOMETURON_MZ: [f32; 50] = [
    92.995468, 94.994881, 103.477097, 113.998604, 116.995888, 121.980446, 126.208969, 130.992706,
    138.016098, 140.960709, 140.996033, 142.047501, 142.992828, 143.011475, 144.991028, 145.027466,
    146.002319, 146.004761, 147.006485, 149.975464, 154.047363, 154.992935, 155.763153, 156.955978,
    159.972824, 160.038086, 160.985657, 166.011261, 166.992477, 170.987732, 171.006287, 182.987839,
    184.021851, 184.876648, 184.950592, 185.951935, 186.017471, 187.967499, 190.993088, 191.012497,
    193.637054, 210.982483, 210.999313, 211.069107, 229.941727, 230.917175, 230.942978, 231.000412,
    231.010406, 231.075363,
];
/// The intensities for fluometuron.
pub const FLUOMETURON_INTENSITIES: [f32; 50] = [
    1070.881592,
    817.482178,
    650.098328,
    953.939087,
    6981.539062,
    5288.478027,
    745.524597,
    1182.514771,
    1228.614136,
    1162.585083,
    1386.337158,
    863.625305,
    1045.432129,
    679.081238,
    754.778748,
    679.229492,
    664.853699,
    5940.523438,
    721.856689,
    90712.640625,
    2206.841553,
    1448.434204,
    692.309204,
    1486.601074,
    205984.15625,
    29351.722656,
    780.362488,
    30585.292969,
    1255.964478,
    2300.630615,
    827.329834,
    7135.11084,
    829.690796,
    817.628113,
    3974.198486,
    425139.0,
    230604.3125,
    1206.440063,
    9418.154297,
    883.679077,
    683.425842,
    10409.8125,
    3955.724121,
    1054.624512,
    880.444214,
    2079.729492,
    731.365173,
    3336.276855,
    766147.5625,
    348723.125,
];

super::impl_reference_spectrum!(
    FluometuronSpectrum,
    fluometuron,
    FLUOMETURON_PRECURSOR_MZ,
    FLUOMETURON_MZ,
    FLUOMETURON_INTENSITIES
);
