//! Submodule providing data for elaidic acid.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of elaidic acid.
pub trait ElaidicAcidSpectrum: SpectrumAlloc {
    /// Create a new spectrum of elaidic acid.
    fn elaidic_acid() -> Self;
}

/// The precursor mass over charge value for elaidic acid.
pub const ELAIDIC_ACID_PRECURSOR_MZ: f32 = 281.28;

/// The mass over charge values for elaidic acid.
pub const ELAIDIC_ACID_MZ: [f32; 10] = [
    66.540428, 71.039917, 74.134857, 74.706421, 96.398811, 210.813889, 217.960938, 262.298859,
    281.269531, 282.274506,
];
/// The intensities for elaidic acid.
pub const ELAIDIC_ACID_INTENSITIES: [f32; 10] = [
    14092.856445,
    158053.390625,
    14685.456055,
    14086.728516,
    14699.806641,
    12922.707031,
    15444.878906,
    18232.212891,
    26642488.0,
    2537840.75,
];

super::impl_reference_spectrum!(
    ElaidicAcidSpectrum,
    elaidic_acid,
    ELAIDIC_ACID_PRECURSOR_MZ,
    ELAIDIC_ACID_MZ,
    ELAIDIC_ACID_INTENSITIES
);
