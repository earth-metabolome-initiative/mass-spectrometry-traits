//! Submodule providing data for halofenozide.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of halofenozide.
pub trait HalofenozideSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of halofenozide.
    fn halofenozide() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for halofenozide.
pub const HALOFENOZIDE_PRECURSOR_MZ: f32 = 329.107;

/// The mass over charge values for halofenozide.
pub const HALOFENOZIDE_MZ: [f32; 42] = [
    71.217468, 77.039093, 77.574699, 78.743614, 80.986748, 82.516167, 84.808685, 100.542259,
    105.248543, 109.415108, 109.797424, 110.387177, 111.000534, 117.007881, 117.046005, 120.045425,
    120.99324, 120.995453, 121.029549, 121.064598, 135.079132, 135.364288, 151.006912, 154.006592,
    154.990662, 161.035706, 176.108078, 181.512009, 193.186783, 193.628799, 194.996918, 201.789154,
    205.934998, 234.458969, 241.647858, 271.027985, 272.035919, 273.044281, 308.260193, 329.106781,
    336.834595, 340.572449,
];
/// The intensities for halofenozide.
pub const HALOFENOZIDE_INTENSITIES: [f32; 42] = [
    85261.578125,
    506643.09375,
    76159.90625,
    85028.742188,
    82321.328125,
    80079.625,
    88184.390625,
    77187.226562,
    81736.609375,
    76744.390625,
    74182.46875,
    81585.398438,
    622420.9375,
    93287.15625,
    94594.5625,
    187641.25,
    738083.5,
    198312.421875,
    263237584.0,
    823973.9375,
    92058.53125,
    105635.5,
    120882.453125,
    110706.15625,
    6731199.0,
    780950.125,
    692712.9375,
    87285.828125,
    89573.960938,
    82860.179688,
    532079.5,
    88561.851562,
    82505.8125,
    89468.875,
    90306.640625,
    159692.203125,
    3868541.0,
    2607540.75,
    98575.4375,
    38326284.0,
    95791.382812,
    87944.8125,
];

super::impl_reference_spectrum!(
    HalofenozideSpectrum,
    halofenozide,
    HALOFENOZIDE_PRECURSOR_MZ,
    HALOFENOZIDE_MZ,
    HALOFENOZIDE_INTENSITIES
);
