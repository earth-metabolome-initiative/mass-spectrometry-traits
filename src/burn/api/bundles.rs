//! Tensor-bundle types for the spectral kernel API.
//!
//! [`SpectrumBatch`] and [`PairwiseParams`] are the user-facing input bundles
//! consumed by [`crate::burn::api::paired_kernel`],
//! [`crate::burn::api::cross_kernel`], and [`crate::burn::api::ranking_kernel`].
//! [`SpectrumPrimitive`] and [`PairwisePrimitive`] are their primitive-tensor
//! equivalents, used at the [`crate::burn::api::SpectralKernelBackend`] trait
//! layer.

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;

/// One side of a paired / cross input: m/z and intensity dense arrays
/// `[batch, peak_width]` plus a precursor mass `[batch]`. All three tensors
/// must live on the same device.
///
/// Zero-cost user-facing bundle for [`crate::burn::api::paired_kernel`],
/// [`crate::burn::api::cross_kernel`], and
/// [`crate::burn::api::ranking_kernel`]. The kernels destructure it and
/// verify the same-device invariant inside.
///
/// `Clone` is cheap: the inner Burn tensors are refcounted device-handle
/// wrappers, so reusing the same batch across multiple kernel launches via
/// `.clone()` does not copy GPU data, it just bumps the handle's strong
/// count. Build a `SpectrumBatch` once and clone it per kernel call.
#[derive(Clone, Debug)]
pub struct SpectrumBatch<B: Backend> {
    pub mz: BurnTensor<B, 2>,
    pub intensity: BurnTensor<B, 2>,
    pub precursor: BurnTensor<B, 1>,
}

impl<B: Backend> SpectrumBatch<B> {
    /// Build from the three tensors directly.
    #[inline]
    pub fn new(
        mz: BurnTensor<B, 2>,
        intensity: BurnTensor<B, 2>,
        precursor: BurnTensor<B, 1>,
    ) -> Self {
        Self {
            mz,
            intensity,
            precursor,
        }
    }

    #[inline]
    pub(crate) fn into_primitive(self) -> SpectrumPrimitive<B> {
        SpectrumPrimitive {
            mz: self.mz.into_primitive().tensor(),
            intensity: self.intensity.into_primitive().tensor(),
            precursor: self.precursor.into_primitive().tensor(),
        }
    }
}

/// Per-row scoring parameters for the paired kernel: three `[batch]` tensors
/// so that each pair can use its own `mz_power`, `intensity_power`, and
/// `mz_tolerance`. Callers who want uniform parameters can pass three
/// constant-filled tensors.
#[derive(Clone, Debug)]
pub struct PairwiseParams<B: Backend> {
    pub mz_power: BurnTensor<B, 1>,
    pub intensity_power: BurnTensor<B, 1>,
    pub mz_tolerance: BurnTensor<B, 1>,
}

impl<B: Backend> PairwiseParams<B> {
    #[inline]
    pub fn new(
        mz_power: BurnTensor<B, 1>,
        intensity_power: BurnTensor<B, 1>,
        mz_tolerance: BurnTensor<B, 1>,
    ) -> Self {
        Self {
            mz_power,
            intensity_power,
            mz_tolerance,
        }
    }

    #[inline]
    pub(crate) fn into_primitive(self) -> PairwisePrimitive<B> {
        PairwisePrimitive {
            mz_power: self.mz_power.into_primitive().tensor(),
            intensity_power: self.intensity_power.into_primitive().tensor(),
            mz_tolerance: self.mz_tolerance.into_primitive().tensor(),
        }
    }
}

/// Primitive-tensor equivalent of [`SpectrumBatch`], used at the trait /
/// `FloatTensor<B>` layer.
pub struct SpectrumPrimitive<B: Backend> {
    pub mz: FloatTensor<B>,
    pub intensity: FloatTensor<B>,
    pub precursor: FloatTensor<B>,
}

/// Primitive-tensor equivalent of [`PairwiseParams`].
pub struct PairwisePrimitive<B: Backend> {
    pub mz_power: FloatTensor<B>,
    pub intensity_power: FloatTensor<B>,
    pub mz_tolerance: FloatTensor<B>,
}
