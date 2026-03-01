//! Submodule providing data for hydramethylnon.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of hydramethylnon.
pub trait HydramethylnonSpectrum: SpectrumAlloc {
    /// Create a new spectrum of hydramethylnon.
    fn hydramethylnon() -> Self;
}

/// The precursor mass over charge value for hydramethylnon.
pub const HYDRAMETHYLNON_PRECURSOR_MZ: f32 = 493.184;

/// The mass over charge values for hydramethylnon.
pub const HYDRAMETHYLNON_MZ: [f32; 50] = [
    124.085495, 126.103798, 139.751297, 144.04361, 145.025772, 147.323715, 159.047073, 164.045685,
    169.018173, 170.012665, 177.099213, 186.054199, 189.015671, 196.035873, 196.092331, 198.056,
    202.112762, 213.318878, 220.06694, 237.065659, 238.059921, 240.099792, 242.123978, 247.111572,
    255.110855, 263.066284, 276.133179, 277.581055, 293.583496, 294.121368, 295.960846, 296.138702,
    297.128998, 298.056061, 299.002167, 315.073853, 334.14444, 335.083801, 347.147766, 353.075592,
    355.090485, 356.094025, 363.093231, 367.089539, 381.084442, 383.099091, 417.109436, 437.118408,
    473.183563, 493.185791,
];
/// The intensities for hydramethylnon.
pub const HYDRAMETHYLNON_INTENSITIES: [f32; 50] = [
    181.0, 237.0, 281.0, 198.0, 6988.0, 179.0, 841.0, 323.0, 248.0, 250.0, 290.0, 326.0, 957.0,
    244.0, 211.0, 473.0, 189.0, 336.0, 184.0, 16344.0, 1635.0, 184.0, 590.0, 205.0, 988.0, 250.0,
    9926.0, 180.0, 192.0, 1541.0, 222.0, 15056.0, 209.0, 313.0, 241.0, 320.0, 429.0, 420.0, 7614.0,
    511.0, 4193.0, 249.0, 618.0, 231.0, 2032.0, 6201.0, 316.0, 1599.0, 289.0, 28977.0,
];

super::impl_reference_spectrum!(
    HydramethylnonSpectrum,
    hydramethylnon,
    HYDRAMETHYLNON_PRECURSOR_MZ,
    HYDRAMETHYLNON_MZ,
    HYDRAMETHYLNON_INTENSITIES
);
