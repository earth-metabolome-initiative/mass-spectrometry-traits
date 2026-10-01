//! Submodule providing data for flonicamid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of flonicamid.
pub trait FlonicamidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of flonicamid.
    fn flonicamid() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for flonicamid.
pub const FLONICAMID_PRECURSOR_MZ: f32 = 228.039;

/// The mass over charge values for flonicamid.
pub const FLONICAMID_MZ: [f32; 21] = [
    71.488609, 75.193062, 75.756073, 81.009056, 82.399002, 95.646156, 108.28009, 112.773926,
    125.065948, 129.004944, 140.082657, 144.127625, 144.952179, 146.022461, 151.297333, 153.8685,
    188.02681, 204.467758, 208.03299, 222.037994, 228.039719,
];
/// The intensities for flonicamid.
pub const FLONICAMID_INTENSITIES: [f32; 21] = [
    10459.902344,
    9991.97168,
    9608.24707,
    3064015.75,
    10668.648438,
    9890.357422,
    9473.55957,
    10638.604492,
    11491.947266,
    10350.730469,
    10583.995117,
    11081.182617,
    10602.414062,
    649327.125,
    10217.958984,
    10301.329102,
    161841.53125,
    10327.65332,
    17237.978516,
    11834.775391,
    2163029.75,
];

super::impl_reference_spectrum!(
    FlonicamidSpectrum,
    flonicamid,
    FLONICAMID_PRECURSOR_MZ,
    FLONICAMID_MZ,
    FLONICAMID_INTENSITIES
);
