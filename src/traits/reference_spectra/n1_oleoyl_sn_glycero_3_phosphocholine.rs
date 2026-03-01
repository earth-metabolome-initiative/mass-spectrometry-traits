//! Submodule providing data for n1 oleoyl sn glycero 3 phosphocholine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of n1 oleoyl sn glycero 3 phosphocholine.
pub trait N1OleoylSnGlycero3PhosphocholineSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of n1 oleoyl sn glycero 3 phosphocholine.
    fn n1_oleoyl_sn_glycero_3_phosphocholine() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for n1 oleoyl sn glycero 3 phosphocholine.
pub const N1_OLEOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_PRECURSOR_MZ: f32 = 522.356;

/// The mass over charge values for n1 oleoyl sn glycero 3 phosphocholine.
pub const N1_OLEOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_MZ: [f32; 34] = [
    56.050465, 58.066898, 59.07431, 60.081871, 65.937714, 67.055557, 67.739624, 69.071526,
    71.074745, 81.072548, 83.087875, 86.09903, 92.954948, 95.087791, 97.104416, 98.986626,
    104.110794, 121.105835, 125.005058, 128.048172, 141.244171, 163.021622, 181.035233, 184.084412,
    197.907867, 199.048965, 213.136703, 240.119171, 258.134827, 339.324036, 362.436462, 445.316742,
    504.40741, 522.408142,
];
/// The intensities for n1 oleoyl sn glycero 3 phosphocholine.
pub const N1_OLEOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_INTENSITIES: [f32; 34] = [
    1213305.625,
    13339284.0,
    1786472.875,
    45427008.0,
    4670012.0,
    1382453.5,
    1331578.5,
    1382107.5,
    7062875.5,
    3507316.5,
    2351463.75,
    196134032.0,
    1272726.125,
    5280955.5,
    3122201.0,
    1570968.0,
    1554161664.0,
    2696536.5,
    40033944.0,
    1342755.875,
    1348200.0,
    4867282.0,
    7821372.5,
    2019690496.0,
    1418892.625,
    7051761.0,
    1555774.625,
    2812621.5,
    40825316.0,
    59300656.0,
    1306675.75,
    20468510.0,
    105828248.0,
    70303520.0,
];

super::impl_reference_spectrum!(
    N1OleoylSnGlycero3PhosphocholineSpectrum,
    n1_oleoyl_sn_glycero_3_phosphocholine,
    N1_OLEOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_PRECURSOR_MZ,
    N1_OLEOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_MZ,
    N1_OLEOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_INTENSITIES
);
