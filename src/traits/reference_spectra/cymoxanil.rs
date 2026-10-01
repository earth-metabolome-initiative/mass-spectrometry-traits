//! Submodule providing data for cymoxanil.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of cymoxanil.
pub trait CymoxanilSpectrum: SpectrumAlloc {
    /// Create a new spectrum of cymoxanil.
    fn cymoxanil() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for cymoxanil.
pub const CYMOXANIL_PRECURSOR_MZ: f32 = 235.109;

/// The mass over charge values for cymoxanil.
pub const CYMOXANIL_MZ: [f32; 36] = [
    72.044838, 76.033432, 78.220329, 78.539062, 88.795517, 89.476204, 91.926003, 92.050064,
    92.790001, 97.448891, 98.061066, 99.112335, 101.068611, 108.082916, 109.065331, 109.823921,
    116.071747, 116.07412, 117.846771, 123.001793, 128.995926, 129.484299, 133.886871, 136.515533,
    136.55336, 137.799179, 142.051178, 142.418442, 166.217941, 166.993073, 175.452194, 194.987793,
    199.135468, 214.99379, 238.119476, 259.231232,
];
/// The intensities for cymoxanil.
pub const CYMOXANIL_INTENSITIES: [f32; 36] = [
    32489.287109,
    2112.94751,
    1926.926514,
    2485.62207,
    2288.297363,
    2342.605469,
    2233.634033,
    1932.575806,
    1883.942993,
    2170.159912,
    5182.72168,
    2339.184326,
    2237.38623,
    2194.788818,
    2068.92627,
    2320.863525,
    472248.65625,
    2962.013428,
    2535.399414,
    2827.762451,
    2620.922119,
    2288.770752,
    2127.6604,
    2280.920654,
    2182.244629,
    2270.344971,
    5092.520996,
    2245.243652,
    2177.834229,
    11341.500977,
    2187.179443,
    18826.222656,
    2634.912354,
    13804.333008,
    2551.169434,
    2510.677734,
];

super::impl_reference_spectrum!(
    CymoxanilSpectrum,
    cymoxanil,
    CYMOXANIL_PRECURSOR_MZ,
    CYMOXANIL_MZ,
    CYMOXANIL_INTENSITIES
);
