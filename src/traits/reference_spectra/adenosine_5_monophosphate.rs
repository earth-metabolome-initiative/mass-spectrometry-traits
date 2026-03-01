//! Submodule providing data for adenosine 5 monophosphate.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of adenosine 5 monophosphate.
pub trait Adenosine5MonophosphateSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of adenosine 5 monophosphate.
    fn adenosine_5_monophosphate() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for adenosine 5 monophosphate.
pub const ADENOSINE_5_MONOPHOSPHATE_PRECURSOR_MZ: f32 = 348.071;

/// The mass over charge values for adenosine 5 monophosphate.
pub const ADENOSINE_5_MONOPHOSPHATE_MZ: [f32; 16] = [
    53.16864, 54.022614, 55.029789, 56.351715, 67.030807, 72.564934, 78.628639, 97.031715,
    119.039558, 136.067947, 137.071198, 178.083191, 232.09903, 250.113251, 348.10614, 349.103241,
];
/// The intensities for adenosine 5 monophosphate.
pub const ADENOSINE_5_MONOPHOSPHATE_INTENSITIES: [f32; 16] = [
    183522.75,
    77341.0,
    327396.3125,
    94952.304688,
    185159.515625,
    167656.46875,
    108692.148438,
    17950362.0,
    148895.328125,
    363874208.0,
    1917089.5,
    111951.851562,
    147844.703125,
    1312412.25,
    29370612.0,
    537890.4375,
];

super::impl_reference_spectrum!(
    Adenosine5MonophosphateSpectrum,
    adenosine_5_monophosphate,
    ADENOSINE_5_MONOPHOSPHATE_PRECURSOR_MZ,
    ADENOSINE_5_MONOPHOSPHATE_MZ,
    ADENOSINE_5_MONOPHOSPHATE_INTENSITIES
);
