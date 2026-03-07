//! Submodule providing data for dinotefuran.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of dinotefuran.
pub trait DinotefuranSpectrum: SpectrumAlloc {
    /// Create a new spectrum of dinotefuran.
    fn dinotefuran() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for dinotefuran.
pub const DINOTEFURAN_PRECURSOR_MZ: f32 = 201.1;

/// The mass over charge values for dinotefuran.
pub const DINOTEFURAN_MZ: [f32; 46] = [
    71.027733, 71.537651, 73.040215, 78.542358, 79.402397, 80.05011, 80.236748, 82.029366,
    82.065849, 82.205826, 84.609589, 86.132095, 88.951668, 89.339272, 90.426201, 92.617485,
    94.010681, 96.374969, 98.06105, 98.136162, 100.053505, 100.076767, 107.061668, 109.077179,
    110.061188, 111.092697, 115.974823, 117.042122, 121.077255, 124.0644, 126.325096, 127.087761,
    127.980865, 129.149963, 132.287125, 138.188187, 139.087799, 144.948044, 155.09462, 157.098145,
    171.089096, 187.020874, 200.998276, 201.021927, 201.09967, 211.037918,
];
/// The intensities for dinotefuran.
pub const DINOTEFURAN_INTENSITIES: [f32; 46] = [
    23093.925781,
    19237.535156,
    401028.25,
    19775.939453,
    19141.873047,
    125420.25,
    21995.064453,
    142619.046875,
    33280.800781,
    19280.5,
    19903.478516,
    19740.677734,
    20399.4375,
    18848.279297,
    25833.730469,
    23899.625,
    19055.25,
    21463.513672,
    98215.664062,
    19611.330078,
    19291.736328,
    125469.53125,
    19407.107422,
    614907.1875,
    114163.554688,
    47289.8125,
    21509.005859,
    48021.691406,
    31274.802734,
    22565.787109,
    23084.607422,
    45155.34375,
    20589.125,
    20198.210938,
    21266.578125,
    21035.707031,
    431434.6875,
    21595.826172,
    22858.933594,
    165819.203125,
    32269.355469,
    27100.730469,
    50467.4375,
    27232.902344,
    8135313.5,
    25212.037109,
];

super::impl_reference_spectrum!(
    DinotefuranSpectrum,
    dinotefuran,
    DINOTEFURAN_PRECURSOR_MZ,
    DINOTEFURAN_MZ,
    DINOTEFURAN_INTENSITIES
);
