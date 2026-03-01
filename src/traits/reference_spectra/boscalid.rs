//! Submodule providing data for boscalid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of boscalid.
pub trait BoscalidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of boscalid.
    fn boscalid() -> Self;
}

/// The precursor mass over charge value for boscalid.
pub const BOSCALID_PRECURSOR_MZ: f32 = 367.203;

/// The mass over charge values for boscalid.
pub const BOSCALID_MZ: [f32; 17] = [
    82.528824, 93.773705, 147.951981, 148.747986, 148.960236, 149.058548, 149.309525, 149.491135,
    154.200317, 165.050537, 189.064651, 204.141037, 310.129486, 311.138489, 331.095856, 366.384247,
    367.198914,
];
/// The intensities for boscalid.
pub const BOSCALID_INTENSITIES: [f32; 17] = [
    151.0, 124.0, 285.0, 129.0, 138.0, 7237.0, 100.0, 97.0, 77.0, 230.0, 250.0, 106.0, 371.0, 91.0,
    145.0, 169.0, 1752.0,
];

super::impl_reference_spectrum!(
    BoscalidSpectrum,
    boscalid,
    BOSCALID_PRECURSOR_MZ,
    BOSCALID_MZ,
    BOSCALID_INTENSITIES
);
