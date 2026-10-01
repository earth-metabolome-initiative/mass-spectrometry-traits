//! Submodule providing data for flutolanil.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of flutolanil.
pub trait FlutolanilSpectrum: SpectrumAlloc {
    /// Create a new spectrum of flutolanil.
    fn flutolanil() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for flutolanil.
pub const FLUTOLANIL_PRECURSOR_MZ: f32 = 322.106;

/// The mass over charge values for flutolanil.
pub const FLUTOLANIL_MZ: [f32; 38] = [
    83.313423, 92.995506, 94.994263, 100.395416, 102.948746, 134.024734, 144.97908, 144.981567,
    145.027115, 145.073929, 146.961319, 160.018463, 174.880981, 174.956131, 176.072159, 189.016937,
    192.045914, 195.935303, 205.99321, 214.011902, 220.04068, 222.570877, 240.047104, 240.053329,
    241.993011, 255.990158, 258.313202, 259.468933, 260.052673, 279.051605, 280.059143, 281.987335,
    300.611328, 301.994171, 320.111267, 321.981445, 322.011566, 322.106171,
];
/// The intensities for flutolanil.
pub const FLUTOLANIL_INTENSITIES: [f32; 38] = [
    302.183502,
    310.898224,
    330.41507,
    308.671539,
    332.488831,
    18744.966797,
    396.435333,
    422.584808,
    289510.5625,
    498.356628,
    359.518311,
    333.799438,
    292.356293,
    701.205811,
    1235.146484,
    67469.21875,
    517.275696,
    320.338745,
    407.726624,
    456.26181,
    367.43512,
    337.108215,
    14434.174805,
    505.596436,
    510.866669,
    499.402039,
    363.951996,
    323.601227,
    12699.446289,
    1592.707764,
    518.533447,
    558.223816,
    294.680634,
    400.485291,
    511.602905,
    468.020966,
    518.046204,
    97915.03125,
];

super::impl_reference_spectrum!(
    FlutolanilSpectrum,
    flutolanil,
    FLUTOLANIL_PRECURSOR_MZ,
    FLUTOLANIL_MZ,
    FLUTOLANIL_INTENSITIES
);
