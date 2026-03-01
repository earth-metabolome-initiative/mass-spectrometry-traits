//! Submodule providing data for forchlorfenuron.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of forchlorfenuron.
pub trait ForchlorfenuronSpectrum: SpectrumAlloc {
    /// Create a new spectrum of forchlorfenuron.
    fn forchlorfenuron() -> Self;
}

/// The precursor mass over charge value for forchlorfenuron.
pub const FORCHLORFENURON_PRECURSOR_MZ: f32 = 246.044;

/// The mass over charge values for forchlorfenuron.
pub const FORCHLORFENURON_MZ: [f32; 37] = [
    70.289421, 71.438873, 74.574997, 75.907471, 79.525673, 82.010315, 82.248947, 82.700256,
    85.936028, 91.029884, 92.050423, 101.35421, 101.923286, 110.177071, 121.272697, 126.968719,
    126.996384, 127.003166, 127.007233, 127.04425, 127.046623, 127.332687, 145.64238, 152.986496,
    165.913391, 175.603226, 177.348831, 183.222946, 185.695435, 189.437851, 190.24855, 192.428085,
    203.051865, 229.039017, 246.044846, 251.947693, 260.145111,
];
/// The intensities for forchlorfenuron.
pub const FORCHLORFENURON_INTENSITIES: [f32; 37] = [
    12235.875,
    11594.264648,
    12430.673828,
    11386.40918,
    11941.973633,
    13164.25,
    15189.087891,
    12931.595703,
    13813.427734,
    2710412.25,
    311507.53125,
    12669.960938,
    12677.509766,
    13334.1875,
    12633.004883,
    136388.234375,
    15176.239258,
    182541.609375,
    34249832.0,
    42121.414062,
    68119.5625,
    13004.168945,
    13413.945312,
    15291.621094,
    13092.302734,
    12761.53125,
    13346.677734,
    12318.283203,
    15095.069336,
    15187.913086,
    12913.290039,
    12093.927734,
    13177.487305,
    13223.240234,
    14766.540039,
    13573.495117,
    13934.783203,
];

super::impl_reference_spectrum!(
    ForchlorfenuronSpectrum,
    forchlorfenuron,
    FORCHLORFENURON_PRECURSOR_MZ,
    FORCHLORFENURON_MZ,
    FORCHLORFENURON_INTENSITIES
);
