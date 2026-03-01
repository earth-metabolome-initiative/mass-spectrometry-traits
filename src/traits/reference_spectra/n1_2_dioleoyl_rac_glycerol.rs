//! Submodule providing data for n1 2 dioleoyl rac glycerol.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of n1 2 dioleoyl rac glycerol.
pub trait N12DioleoylRacGlycerolSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of n1 2 dioleoyl rac glycerol.
    fn n1_2_dioleoyl_rac_glycerol() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for n1 2 dioleoyl rac glycerol.
pub const N1_2_DIOLEOYL_RAC_GLYCEROL_PRECURSOR_MZ: f32 = 638.572;

/// The mass over charge values for n1 2 dioleoyl rac glycerol.
pub const N1_2_DIOLEOYL_RAC_GLYCEROL_MZ: [f32; 50] = [
    50.945892, 55.059273, 57.059525, 63.637402, 67.055725, 69.071114, 71.079208, 79.055817,
    81.072189, 83.087914, 85.092667, 93.072632, 95.089478, 96.029053, 97.099899, 99.084808,
    101.062492, 107.089516, 109.104607, 111.111977, 113.099854, 115.078468, 121.106178, 123.122314,
    125.117599, 127.115433, 135.122864, 137.137894, 139.117599, 149.140305, 151.155045, 153.133698,
    158.824615, 163.15744, 165.171539, 167.151276, 177.173584, 179.187012, 181.167252, 191.188797,
    205.205658, 209.203598, 247.255539, 265.276306, 339.323303, 340.324585, 447.389801, 603.61969,
    604.578552, 632.542236,
];
/// The intensities for n1 2 dioleoyl rac glycerol.
pub const N1_2_DIOLEOYL_RAC_GLYCEROL_INTENSITIES: [f32; 50] = [
    2844001.0,
    67081604.0,
    149672320.0,
    2866352.0,
    148838672.0,
    230096304.0,
    73315712.0,
    28389958.0,
    262218496.0,
    281579104.0,
    62029432.0,
    58596672.0,
    322971296.0,
    5372351.0,
    291956192.0,
    15964003.0,
    2996671.5,
    74576712.0,
    198341616.0,
    164105696.0,
    6020476.5,
    3257781.25,
    238404912.0,
    76046808.0,
    54939928.0,
    11647947.0,
    225475008.0,
    28242366.0,
    31392528.0,
    110327856.0,
    41570512.0,
    30875582.0,
    3167000.5,
    51461444.0,
    29585152.0,
    17858180.0,
    24024350.0,
    4034035.75,
    6927290.0,
    14903496.0,
    6142258.0,
    6064671.0,
    117531448.0,
    220604608.0,
    2278431744.0,
    23924674.0,
    3065698.75,
    305274624.0,
    5050710.5,
    3383829.5,
];

super::impl_reference_spectrum!(
    N12DioleoylRacGlycerolSpectrum,
    n1_2_dioleoyl_rac_glycerol,
    N1_2_DIOLEOYL_RAC_GLYCEROL_PRECURSOR_MZ,
    N1_2_DIOLEOYL_RAC_GLYCEROL_MZ,
    N1_2_DIOLEOYL_RAC_GLYCEROL_INTENSITIES
);
