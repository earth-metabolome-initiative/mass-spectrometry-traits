//! Submodule providing data for fipronil.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of fipronil.
pub trait FipronilSpectrum: SpectrumAlloc + Sized {
    /// Create a new spectrum of fipronil.
    fn fipronil() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError>;
}

/// The precursor mass over charge value for fipronil.
pub const FIPRONIL_PRECURSOR_MZ: f32 = 434.931;

/// The mass over charge values for fipronil.
pub const FIPRONIL_MZ: [f32; 45] = [
    93.45182, 96.816017, 131.315475, 149.970917, 162.584579, 170.006256, 179.135818, 182.00441,
    183.018814, 217.988495, 235.938675, 242.983215, 243.990479, 244.920349, 245.98819, 249.221024,
    249.960327, 252.995941, 253.987289, 267.992706, 277.957001, 278.014679, 278.989136, 281.996643,
    287.961212, 288.110107, 317.275757, 317.969757, 318.74527, 319.963287, 321.125732, 329.574677,
    329.817413, 329.95993, 332.210022, 332.961914, 333.2742, 333.346558, 348.93222, 350.988983,
    364.93689, 365.931549, 398.961151, 434.940216, 462.452972,
];
/// The intensities for fipronil.
pub const FIPRONIL_INTENSITIES: [f32; 45] = [
    86.0, 68.0, 174.0, 78.0, 128.0, 553.0, 118.0, 264.0, 924.0, 892.0, 185.0, 87.0, 174.0, 140.0,
    240.0, 241.0, 2380.0, 63.0, 388.0, 195.0, 1898.0, 192.0, 84.0, 359.0, 394.0, 94.0, 151.0,
    4346.0, 38.0, 204.0, 218.0, 87.0, 158.0, 12113.0, 253.0, 281.0, 143.0, 44.0, 363.0, 255.0,
    90.0, 943.0, 351.0, 229.0, 30.0,
];

super::impl_reference_spectrum!(
    FipronilSpectrum,
    fipronil,
    FIPRONIL_PRECURSOR_MZ,
    FIPRONIL_MZ,
    FIPRONIL_INTENSITIES
);
