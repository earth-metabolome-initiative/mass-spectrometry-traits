//! Submodule providing data for metaflumizone.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of metaflumizone.
pub trait MetaflumizoneSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of metaflumizone.
    fn metaflumizone() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for metaflumizone.
pub const METAFLUMIZONE_PRECURSOR_MZ: f32 = 505.11;

/// The mass over charge values for metaflumizone.
pub const METAFLUMIZONE_MZ: [f32; 50] = [
    83.74292, 102.032654, 112.098534, 112.533958, 115.0317, 116.034981, 116.060135, 117.044144,
    137.017044, 141.043777, 145.858398, 146.022079, 148.485184, 149.258011, 153.889435, 156.72261,
    157.023895, 163.945999, 165.329651, 171.042923, 171.687454, 174.15062, 176.032547, 199.045044,
    214.064865, 218.055832, 232.062515, 234.066772, 242.071152, 253.069016, 272.086975, 274.090302,
    275.088989, 282.086853, 285.065704, 288.346375, 299.648041, 300.0755, 301.179138, 301.355377,
    301.846741, 301.900513, 302.09201, 302.739441, 302.950592, 303.084625, 304.971924, 305.46405,
    328.071594, 369.12738,
];
/// The intensities for metaflumizone.
pub const METAFLUMIZONE_INTENSITIES: [f32; 50] = [
    227.0, 597.0, 109.0, 185.0, 247.0, 169.0, 187.0, 795.0, 278.0, 1755.0, 89.0, 169.0, 159.0,
    203.0, 164.0, 85.0, 310.0, 139.0, 72.0, 230.0, 119.0, 66.0, 179.0, 466.0, 840.0, 196.0, 276.0,
    314.0, 124.0, 89.0, 769.0, 1000.0, 168.0, 444.0, 2503.0, 62.0, 132.0, 515.0, 112.0, 99.0,
    228.0, 146.0, 57633.0, 144.0, 97.0, 166.0, 62.0, 98.0, 1681.0, 191.0,
];

super::impl_reference_spectrum!(
    MetaflumizoneSpectrum,
    metaflumizone,
    METAFLUMIZONE_PRECURSOR_MZ,
    METAFLUMIZONE_MZ,
    METAFLUMIZONE_INTENSITIES
);
