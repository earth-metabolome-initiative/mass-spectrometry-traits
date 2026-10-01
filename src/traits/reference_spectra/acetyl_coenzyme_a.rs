//! Submodule providing data for acetyl coenzyme a.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of acetyl coenzyme a.
pub trait AcetylCoenzymeASpectrum: SpectrumAlloc {
    /// Create a new spectrum of acetyl coenzyme a.
    fn acetyl_coenzyme_a() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for acetyl coenzyme a.
pub const ACETYL_COENZYME_A_PRECURSOR_MZ: f32 = 810.134;

/// The mass over charge values for acetyl coenzyme a.
pub const ACETYL_COENZYME_A_MZ: [f32; 50] = [
    56.699314, 59.669636, 61.012016, 72.045563, 85.067047, 88.023438, 97.03109, 99.058075,
    103.024994, 114.040779, 120.051292, 124.080612, 130.083542, 132.053238, 136.067657, 141.058105,
    142.092651, 159.06665, 166.093719, 174.066849, 175.063568, 177.003876, 184.107941, 191.095322,
    201.082718, 205.074677, 217.078522, 243.136307, 257.152313, 260.139984, 261.149811, 271.130768,
    273.145355, 285.14856, 303.165771, 312.073608, 330.091919, 341.122498, 357.176483, 383.138733,
    401.155548, 410.070404, 428.085236, 508.055969, 575.761719, 592.094177, 665.603027, 725.55304,
    726.852844, 810.241272,
];
/// The intensities for acetyl coenzyme a.
pub const ACETYL_COENZYME_A_INTENSITIES: [f32; 50] = [
    2543.476074,
    2595.475586,
    2604.205322,
    8423.014648,
    5009.116211,
    37566.316406,
    14452.125977,
    128788.3125,
    58216.050781,
    26580.705078,
    10576.795898,
    14647.025391,
    17490.917969,
    25767.755859,
    708005.5,
    14606.863281,
    64855.636719,
    448516.46875,
    12795.192383,
    58247.546875,
    71415.007812,
    3053.031006,
    150729.609375,
    5521.765137,
    564433.0,
    8376.095703,
    5697.430176,
    35701.628906,
    4515.775879,
    3410.620117,
    62447.960938,
    4368.574219,
    8134.462891,
    15628.324219,
    3423450.5,
    10327.137695,
    58569.890625,
    11417.602539,
    3029.675049,
    10614.200195,
    27749.167969,
    40382.171875,
    514052.03125,
    24185.453125,
    2574.585693,
    2550.734131,
    2621.323242,
    3515.374756,
    2427.372559,
    29246.314453,
];

super::impl_reference_spectrum!(
    AcetylCoenzymeASpectrum,
    acetyl_coenzyme_a,
    ACETYL_COENZYME_A_PRECURSOR_MZ,
    ACETYL_COENZYME_A_MZ,
    ACETYL_COENZYME_A_INTENSITIES
);
