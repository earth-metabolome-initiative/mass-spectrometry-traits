//! Per-metric `#[cube]` scorer implementations and shared peak helpers.
//!
//! Each metric lives in its own submodule and implements
//! [`crate::burn::SpectralPairScorer`] for its zero-sized marker type. The
//! launchers in [`crate::burn::launchers`] are generic over the marker, so the
//! per-pair scoring routine is monomorphized per metric.

pub mod cosine;
pub mod entropy;
pub mod modified_dp;
pub mod peak_ops;
