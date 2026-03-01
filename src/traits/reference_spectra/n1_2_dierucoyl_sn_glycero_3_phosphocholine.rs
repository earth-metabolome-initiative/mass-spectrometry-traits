//! Submodule providing data for n1 2 dierucoyl sn glycero 3 phosphocholine.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of n1 2 dierucoyl sn glycero 3 phosphocholine.
pub trait N12DierucoylSnGlycero3PhosphocholineSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of n1 2 dierucoyl sn glycero 3 phosphocholine.
    fn n1_2_dierucoyl_sn_glycero_3_phosphocholine()
    -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for n1 2 dierucoyl sn glycero 3 phosphocholine.
pub const N1_2_DIERUCOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_PRECURSOR_MZ: f32 = 898.727;

/// The mass over charge values for n1 2 dierucoyl sn glycero 3 phosphocholine.
pub const N1_2_DIERUCOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_MZ: [f32; 23] = [
    66.137428, 71.073753, 74.413353, 80.975517, 86.09819, 91.968964, 97.102364, 98.986351,
    104.109001, 110.867332, 125.004265, 184.083572, 250.575409, 330.388519, 341.566345, 415.455444,
    472.335663, 489.3862, 560.446716, 630.475891, 678.328552, 725.424438, 866.950623,
];
/// The intensities for n1 2 dierucoyl sn glycero 3 phosphocholine.
pub const N1_2_DIERUCOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_INTENSITIES: [f32; 23] = [
    428881.90625,
    3867711.75,
    406184.09375,
    501394.34375,
    69647360.0,
    375753.5,
    381627.0625,
    2695920.25,
    11314820.0,
    376947.34375,
    28576308.0,
    1251055616.0,
    390975.46875,
    384256.625,
    387469.25,
    453481.25,
    501323.625,
    416779.59375,
    1631406.625,
    663449.3125,
    469969.625,
    852037.3125,
    455642.8125,
];

super::impl_reference_spectrum!(
    N12DierucoylSnGlycero3PhosphocholineSpectrum,
    n1_2_dierucoyl_sn_glycero_3_phosphocholine,
    N1_2_DIERUCOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_PRECURSOR_MZ,
    N1_2_DIERUCOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_MZ,
    N1_2_DIERUCOYL_SN_GLYCERO_3_PHOSPHOCHOLINE_INTENSITIES
);
