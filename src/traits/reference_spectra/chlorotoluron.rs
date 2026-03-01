//! Submodule providing data for chlorotoluron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of chlorotoluron.
pub trait ChlorotoluronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of chlorotoluron.
    fn chlorotoluron() -> Self;
}

/// The precursor mass over charge value for chlorotoluron.
pub const CHLOROTOLURON_PRECURSOR_MZ: f32 = 211.065;

/// The mass over charge values for chlorotoluron.
pub const CHLOROTOLURON_MZ: [f32; 50] = [
    73.00779, 78.96534, 86.988396, 92.927948, 92.995529, 94.924965, 94.993591, 96.976158,
    99.005241, 104.99585, 107.017021, 114.581589, 116.995911, 120.99073, 123.005325, 127.000145,
    130.992752, 140.996048, 142.992905, 142.9953, 143.01152, 144.990662, 146.990036, 147.006516,
    150.988892, 151.00032, 154.992981, 163.017776, 163.025711, 166.996613, 167.03244, 167.107727,
    170.987686, 170.990952, 171.00679, 181.01619, 182.987457, 189.040955, 190.992996, 191.012939,
    191.056976, 193.052582, 209.067017, 210.982666, 210.998627, 211.003006, 211.018677, 211.063324,
    211.099167, 211.134628,
];
/// The intensities for chlorotoluron.
pub const CHLOROTOLURON_INTENSITIES: [f32; 50] = [
    365.122711,
    12880.634766,
    334.09729,
    13180.380859,
    3672.512207,
    13145.848633,
    388.206818,
    10412.742188,
    308.771912,
    6378.486816,
    418.227295,
    298.490082,
    5947.436523,
    315.910095,
    1005.42981,
    1039.387817,
    485.793915,
    3663.487549,
    5023.747559,
    957.985168,
    1594.42041,
    337.84668,
    348.773529,
    474.466461,
    2954.248535,
    613.890991,
    542.252563,
    1064.938599,
    4581.387207,
    502.425873,
    290.191162,
    396.726501,
    10027.28418,
    430.860687,
    3984.820801,
    2737.487061,
    2886.806152,
    1564.701172,
    7992.160645,
    2959.170166,
    85362.453125,
    2296.549561,
    4372.046387,
    1582.077881,
    4457.635742,
    1945.980347,
    2720.68042,
    53968.835938,
    2104.182861,
    958.409546,
];

super::impl_reference_spectrum!(
    ChlorotoluronSpectrum,
    chlorotoluron,
    CHLOROTOLURON_PRECURSOR_MZ,
    CHLOROTOLURON_MZ,
    CHLOROTOLURON_INTENSITIES
);
