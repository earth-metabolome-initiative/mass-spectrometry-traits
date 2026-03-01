//! Submodule providing data for arachidonic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of arachidonic acid.
pub trait ArachidonicAcidSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of arachidonic acid.
    fn arachidonic_acid() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for arachidonic acid.
pub const ARACHIDONIC_ACID_PRECURSOR_MZ: f32 = 303.232;

/// The mass over charge values for arachidonic acid.
pub const ARACHIDONIC_ACID_MZ: [f32; 18] = [
    59.012424, 65.679092, 71.013329, 72.420601, 83.050598, 89.333054, 99.288834, 163.097504,
    177.099335, 191.118622, 205.209518, 231.225479, 259.261658, 260.263458, 285.242432, 301.234131,
    303.26181, 304.26239,
];
/// The intensities for arachidonic acid.
pub const ARACHIDONIC_ACID_INTENSITIES: [f32; 18] = [
    23562038.0,
    251464.15625,
    671767.4375,
    863844.9375,
    2070102.75,
    267005.65625,
    284449.71875,
    1835264.625,
    1152345.25,
    589106.125,
    6655762.0,
    2919716.25,
    23238930.0,
    2808957.75,
    3692310.5,
    1066056.0,
    164966880.0,
    17073884.0,
];

super::impl_reference_spectrum!(
    ArachidonicAcidSpectrum,
    arachidonic_acid,
    ARACHIDONIC_ACID_PRECURSOR_MZ,
    ARACHIDONIC_ACID_MZ,
    ARACHIDONIC_ACID_INTENSITIES
);
