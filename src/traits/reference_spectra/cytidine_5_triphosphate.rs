//! Submodule providing data for cytidine 5 triphosphate.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of cytidine 5 triphosphate.
pub trait Cytidine5TriphosphateSpectrum: SpectrumAlloc {
    /// Create a new spectrum of cytidine 5 triphosphate.
    fn cytidine_5_triphosphate() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for cytidine 5 triphosphate.
pub const CYTIDINE_5_TRIPHOSPHATE_PRECURSOR_MZ: f32 = 483.16;

/// The mass over charge values for cytidine 5 triphosphate.
pub const CYTIDINE_5_TRIPHOSPHATE_MZ: [f32; 31] = [
    53.56868, 69.04248, 95.026825, 97.031891, 98.987991, 104.417656, 109.779282, 112.055038,
    113.05764, 115.042686, 152.374496, 187.843979, 190.069534, 192.028702, 199.129211, 205.943848,
    208.086121, 217.851196, 226.097397, 252.936005, 258.937836, 272.002777, 288.062836, 306.073273,
    324.09137, 327.968628, 328.968597, 331.194214, 369.927277, 386.051056, 484.043457,
];
/// The intensities for cytidine 5 triphosphate.
pub const CYTIDINE_5_TRIPHOSPHATE_INTENSITIES: [f32; 31] = [
    1986.039062,
    16601.279297,
    27994.484375,
    618501.3125,
    5388.470703,
    2298.899658,
    2292.095459,
    7334189.5,
    3632.199219,
    6668.391602,
    2181.114258,
    2087.490723,
    2319.678223,
    12218.072266,
    2743.249023,
    2411.615234,
    203061.671875,
    2062.929688,
    11621.907227,
    2862.479492,
    17388.818359,
    2880.776123,
    18733.226562,
    18825.599609,
    59793.070312,
    3044.641357,
    2270.577393,
    2142.332031,
    9569.357422,
    9121.21875,
    56006.492188,
];

super::impl_reference_spectrum!(
    Cytidine5TriphosphateSpectrum,
    cytidine_5_triphosphate,
    CYTIDINE_5_TRIPHOSPHATE_PRECURSOR_MZ,
    CYTIDINE_5_TRIPHOSPHATE_MZ,
    CYTIDINE_5_TRIPHOSPHATE_INTENSITIES
);
