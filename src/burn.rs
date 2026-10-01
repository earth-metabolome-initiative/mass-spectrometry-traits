//! Burn + CubeCL GPU kernels for spectral similarity metrics.
//!
//! Each metric is exposed as a zero-sized marker type that selects a GPU
//! scoring routine at the type level. This commit ships the foundation
//! plus [`LinearCosineMetric`]. Additional metrics
//! ([`ModifiedLinearCosineMetric`], [`LinearEntropyMetric`],
//! [`ModifiedLinearEntropyMetric`]) land in subsequent commits.
//!
//! Three launch shapes are available per metric:
//!
//! * [`paired_kernel`] — `[B, P] x [B, P] -> [B]`, one score per batch row,
//!   with per-row `mz_power` / `intensity_power` / `mz_tolerance` tensors.
//! * [`cross_kernel`] — `[M, P] x [N, P] -> [M, N]`, full all-pairs matrix
//!   with scalar parameters.
//! * [`ranking_kernel`] — deterministic LCG sampling of `k` non-self partners
//!   per anchor inside a teacher cache, returning a [`RankingOutput<B>`] with
//!   named `candidate_index[B, k]`, `best_position[B]`, `top2_gap[B]`, and
//!   `candidate_scores[B, k]` fields.
//!
//! ## Architecture
//!
//! The keystone is the [`SpectralPairScorer`] `#[cube] trait`. Its single
//! method `score_rows::<F>` takes two preprocessed `[batch, peak_width]` rows
//! and returns a scalar similarity. Each metric implements this trait once.
//! The three launch drivers in [`launchers`] are generic over
//! `M: SpectralPairScorer` and call `M::score_rows::<F>(...)` from inside
//! their respective grid topologies (1D for paired/ranking, 2D for cross).
//! The bulk of the GPU pipeline (launch grid sizing, output allocation,
//! device assertions) is written once in the internal `cube_backend` module
//! as a blanket
//! `impl<R, F, I, BT, M> SpectralKernelBackend<M> for CubeBackend<R, F, I, BT>`,
//! so a new metric marker that implements [`SpectralPairScorer`] +
//! [`KernelMetric`] picks up every shape on every backend automatically.
//!
//! ## Adding a new metric
//!
//! 1. Add a zero-sized marker struct in [`metrics`] and implement
//!    [`KernelMetric`] for it (stable name, precursor/entropy flags, and three
//!    `&'static str` fusion op names). If the metric supports a weighted
//!    prepass, also implement [`EntropyMetric`] for it. That gates the
//!    `with_weighted` builder method on every config to your marker at
//!    compile time.
//! 2. Add `#[cube] impl SpectralPairScorer for MyMarker` in [`kernels`],
//!    usually a thin wrapper around a `#[cube] fn` that contains the
//!    scoring body.
//! 3. Re-export the marker from this module.
//!
//! ## Feature flags
//!
//! * `burn` — base CubeCL kernels + `CubeBackend<R, F, I, BT>` impl. Implies `std`.
//! * `burn-cuda` — pulls in `burn-cuda`. Required to run any of the kernels
//!   on a CUDA GPU.
//! * `burn-wgpu` — alt-runtime selector (WebGPU). Untested in the kernel suite.
//!
//! ## GPU constraints
//!
//! `max_peaks` (the comptime peak-width bound) must be `<= 256`. Above this
//! value the per-thread scratch arrays start to spill out of register /
//! local-memory budget on consumer NVIDIA hardware.
//!
//! ## No-std
//!
//! Burn requires `std`. The `burn` feature transitively enables `std`. The
//! crate as a whole remains `no_std` by default (without `burn`).

pub mod api;
#[cfg(feature = "burn-autodiff")]
mod autodiff;
pub(crate) mod cube_backend;
#[cfg(feature = "burn-fusion")]
mod fusion;
pub mod kernels;
pub mod launchers;
pub mod metrics;
mod scorer_trait;

#[cfg(test)]
mod tests;

pub use api::{
    AllMetricsBackend, CrossConfig, MAX_PEAKS_LIMIT, PairedConfig, PairwiseParams, RankingConfig,
    RankingOutput, RankingWindow, ScoringParams, SpectralKernelBackend, SpectrumBatch,
    cross_kernel, paired_kernel, ranking_kernel,
};
pub use metrics::{
    EntropyMetric, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric,
};
pub use scorer_trait::SpectralPairScorer;
