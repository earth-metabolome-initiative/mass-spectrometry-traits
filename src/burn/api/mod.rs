//! Public Rust-level API for the Burn / CubeCL spectral similarity kernels.
//!
//! The module is split into two submodules for navigability:
//!
//! - [`configs`]: builder-style config types ([`ScoringParams`],
//!   [`RankingWindow`], [`PairedConfig`], [`CrossConfig`], [`RankingConfig`]).
//! - [`bundles`]: tensor-bundle structs ([`SpectrumBatch`],
//!   [`PairwiseParams`], [`SpectrumPrimitive`], [`PairwisePrimitive`]).
//!
//! Everything is re-exported here via `pub use {configs, bundles}::*;` so
//! external callers continue to import from `crate::burn::api::*` directly.
//! The submodule organization is internal navigation, not part of the public
//! path contract.

use burn::tensor::Int as TensorInt;
use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};

use crate::burn::metrics::{
    KernelMetric, LinearCosineMetric, LinearEntropyMetric, ModifiedLinearCosineMetric,
    ModifiedLinearEntropyMetric,
};

pub mod bundles;
pub mod configs;

pub use bundles::*;
pub use configs::*;

/// Hard upper bound on `max_peaks` for any kernel launch. Above this value
/// the modified-variant per-thread scratch arrays start to spill out of
/// register / local-memory budget on consumer NVIDIA hardware.
pub const MAX_PEAKS_LIMIT: usize = 256;

/// Backend trait for the spectral similarity GPU kernels.
///
/// Generic over the metric marker `M`. Implemented as a single blanket
/// `impl<…, M> SpectralKernelBackend<M> for …` on each backend type, so
/// new metrics get every shape on every backend with no extra wiring.
pub trait SpectralKernelBackend<M: KernelMetric>: Backend {
    /// Paired (1-to-1 batch) kernel: `score[i] = sim(left[i], right[i])`.
    ///
    /// All tensors in `left`, `right`, and `params` must live on the same
    /// device. Cosine metrics ignore the precursor tensors but still require
    /// them so the trait shape is uniform across metrics.
    fn paired_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        params: PairwisePrimitive<Self>,
        config: PairedConfig<M>,
    ) -> FloatTensor<Self>;

    /// Cross / all-pairs kernel: `output[i, j] = sim(left[i], right[j])`.
    ///
    /// `left` has shape `[M, P]`, `right` has shape `[N, P]`. Output is
    /// `[M, N]`. Scoring parameters live in [`CrossConfig`] as scalars, see
    /// its docs for the broadcasting rationale.
    fn cross_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        config: CrossConfig<M>,
    ) -> FloatTensor<Self>;

    /// Ranking kernel: deterministic LCG sampling of `k` non-self partners per
    /// anchor inside the teacher cache, plus top-2 reduce.
    ///
    /// `teacher` has shape `[N, P]`. Returns
    /// `(candidate_index[B, k], best_position[B], top2_gap[B], candidate_scores[B, k])`
    /// with `B = config.batch_items()`. `candidate_scores[i, j]` is the
    /// teacher similarity between anchor `i` and `candidate_index[i, j]`,
    /// surfaced so downstream consumers can compute rank-correlation
    /// diagnostics without re-running the scorer.
    fn ranking_score(
        teacher: SpectrumPrimitive<Self>,
        config: RankingConfig<M>,
    ) -> (
        IntTensor<Self>,
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );
}

/// Convenience super-trait satisfied by any backend that implements
/// [`SpectralKernelBackend`] for every built-in metric marker
/// ([`LinearCosineMetric`], [`ModifiedLinearCosineMetric`],
/// [`LinearEntropyMetric`], [`ModifiedLinearEntropyMetric`]).
///
/// Use this when a function or struct needs to be generic in the backend but
/// wants access to *every* metric at runtime, typically because the metric
/// choice comes from configuration or a runtime enum and the dispatch happens
/// inside the function body. The alternative is repeating four
/// `SpectralKernelBackend<M>` bounds at every use site.
///
/// ```ignore
/// fn run_teacher<B: AllMetricsBackend>(teacher: SpectrumBatch<B>, metric: MyEnum) {
///     match metric {
///         MyEnum::Cosine => ranking_kernel(teacher, LinearCosineMetric::ranking_config()),
///         MyEnum::Entropy => ranking_kernel(teacher, LinearEntropyMetric::ranking_config()),
///         /* ... */
///     };
/// }
/// ```
///
/// This trait is auto-implemented for every backend that satisfies the four
/// underlying bounds via a blanket impl, so the `CubeBackend<R, F, I, BT>`,
/// `Autodiff<B, C>`, and `Fusion<B>` impls produced by the rest of this
/// module automatically gain `AllMetricsBackend` with no extra wiring.
///
/// If you only need one or two metrics, prefer naming them directly
/// (`B: SpectralKernelBackend<LinearCosineMetric>`). The narrower bound
/// produces clearer compiler errors and doesn't require the backend to
/// implement metrics you don't use.
pub trait AllMetricsBackend:
    Backend
    + SpectralKernelBackend<LinearCosineMetric>
    + SpectralKernelBackend<ModifiedLinearCosineMetric>
    + SpectralKernelBackend<LinearEntropyMetric>
    + SpectralKernelBackend<ModifiedLinearEntropyMetric>
{
}

impl<B> AllMetricsBackend for B where
    B: Backend
        + SpectralKernelBackend<LinearCosineMetric>
        + SpectralKernelBackend<ModifiedLinearCosineMetric>
        + SpectralKernelBackend<LinearEntropyMetric>
        + SpectralKernelBackend<ModifiedLinearEntropyMetric>
{
}

/// Public wrapper around [`SpectralKernelBackend::paired_score`].
///
/// `left` and `right` are `[batch, peak_width]` plus a `[batch]` precursor
/// each. `params` carries the three `[batch]` per-row scoring tensors so
/// each pair can score under its own `(mz_power, intensity_power,
/// mz_tolerance)` triple. Returns `[batch]`.
///
/// ```
/// // Prefer CUDA when both runtimes are enabled; fall back to the MLIR CPU
/// // runtime (`burn-cpu`) so the same example runs under the GPU-free CI
/// // feature set. When neither runtime is enabled, the body is skipped and
/// // the doctest is a no-op (still compiles, still passes).
/// #[cfg(any(feature = "burn-cuda", feature = "burn-cpu"))]
/// fn run() {
///     use burn::tensor::{Tensor, TensorData};
///     use mass_spectrometry::burn::{
///         KernelMetric, LinearCosineMetric, PairwiseParams, SpectrumBatch,
///         paired_kernel,
///     };
///
///     #[cfg(feature = "burn-cuda")]
///     type B = burn::backend::Cuda<f32, i32>;
///     #[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
///     type B = burn::backend::Cpu<f32, i32>;
///     type Dev = burn::tensor::Device<B>;
///
///     let device = Dev::default();
///
///     let left = SpectrumBatch::<B>::new(
///         Tensor::from_data(TensorData::new(vec![100.0_f32, 200.0], [1, 2]), &device),
///         Tensor::from_data(TensorData::new(vec![10.0_f32, 20.0], [1, 2]), &device),
///         Tensor::from_data(TensorData::new(vec![500.0_f32], [1]), &device),
///     );
///     let right = SpectrumBatch::<B>::new(
///         Tensor::from_data(TensorData::new(vec![100.05_f32, 200.05], [1, 2]), &device),
///         Tensor::from_data(TensorData::new(vec![10.0_f32, 20.0], [1, 2]), &device),
///         Tensor::from_data(TensorData::new(vec![500.0_f32], [1]), &device),
///     );
///     let params = PairwiseParams::<B>::new(
///         Tensor::from_data(TensorData::new(vec![0.0_f32], [1]), &device),
///         Tensor::from_data(TensorData::new(vec![1.0_f32], [1]), &device),
///         Tensor::from_data(TensorData::new(vec![0.1_f32], [1]), &device),
///     );
///     let config = LinearCosineMetric::paired_config()
///         .with_max_peaks(128)
///         .with_epsilon(1.0e-8);
///
///     let scores = paired_kernel::<B, LinearCosineMetric>(left, right, params, config);
///     let score = scores.into_data().to_vec::<f32>().unwrap()[0];
///     assert!(score > 0.99);
/// }
/// #[cfg(not(any(feature = "burn-cuda", feature = "burn-cpu")))]
/// fn run() {}
///
/// fn main() {
///     run();
/// }
/// ```
pub fn paired_kernel<B, M>(
    left: SpectrumBatch<B>,
    right: SpectrumBatch<B>,
    params: PairwiseParams<B>,
    config: PairedConfig<M>,
) -> BurnTensor<B, 1>
where
    B: SpectralKernelBackend<M>,
    M: KernelMetric,
{
    let out = B::paired_score(
        left.into_primitive(),
        right.into_primitive(),
        params.into_primitive(),
        config,
    );
    BurnTensor::from_primitive(TensorPrimitive::Float(out))
}

/// Public wrapper around [`SpectralKernelBackend::cross_score`].
///
/// `left` has shape `[M, P]`, `right` has shape `[N, P]`. Returns `[M, N]`.
/// All MxN pairs share the same scalar scoring parameters from `config`
/// (see [`CrossConfig`] for the broadcasting rationale).
///
/// ```
/// #[cfg(any(feature = "burn-cuda", feature = "burn-cpu"))]
/// fn run() {
///     use burn::tensor::{Tensor, TensorData};
///     use mass_spectrometry::burn::{
///         KernelMetric, LinearCosineMetric, SpectrumBatch, cross_kernel,
///     };
///
///     #[cfg(feature = "burn-cuda")]
///     type B = burn::backend::Cuda<f32, i32>;
///     #[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
///     type B = burn::backend::Cpu<f32, i32>;
///     type Dev = burn::tensor::Device<B>;
///
///     let device = Dev::default();
///
///     // [2 spectra x 2 peaks] on each side -> [2 x 2] score matrix.
///     let left = SpectrumBatch::<B>::new(
///         Tensor::from_data(
///             TensorData::new(vec![100.0_f32, 200.0, 110.0, 210.0], [2, 2]),
///             &device,
///         ),
///         Tensor::from_data(
///             TensorData::new(vec![10.0_f32, 20.0, 12.0, 22.0], [2, 2]),
///             &device,
///         ),
///         Tensor::from_data(TensorData::new(vec![500.0_f32, 510.0], [2]), &device),
///     );
///     let right = left.clone(); // Cheap: Burn tensors are refcounted handles.
///
///     let config = LinearCosineMetric::cross_config()
///         .with_mz_power(0.0)
///         .with_intensity_power(1.0)
///         .with_mz_tolerance(0.1)
///         .with_max_peaks(128)
///         .with_epsilon(1.0e-8);
///
///     let scores = cross_kernel::<B, LinearCosineMetric>(left, right, config);
///     assert_eq!(scores.dims(), [2, 2]);
/// }
/// #[cfg(not(any(feature = "burn-cuda", feature = "burn-cpu")))]
/// fn run() {}
///
/// fn main() {
///     run();
/// }
/// ```
pub fn cross_kernel<B, M>(
    left: SpectrumBatch<B>,
    right: SpectrumBatch<B>,
    config: CrossConfig<M>,
) -> BurnTensor<B, 2>
where
    B: SpectralKernelBackend<M>,
    M: KernelMetric,
{
    let out = B::cross_score(left.into_primitive(), right.into_primitive(), config);
    BurnTensor::from_primitive(TensorPrimitive::Float(out))
}

/// Named output of [`ranking_kernel`]. Mirrors the four-tensor result with
/// labelled fields so downstream callers don't repeat the tuple destructure
/// at every call site.
///
/// * `candidate_index`: `[batch_items, k]` int tensor, partner row indices
///   inside the teacher cache (still in the cache's index space, so add
///   `config.batch_start()` if you need absolute indices).
/// * `best_position`: `[batch_items]` int tensor, the column of
///   `candidate_index` that holds the top-1 partner per anchor.
/// * `top2_gap`: `[batch_items]` float tensor, `score(top-1) - score(top-2)`
///   per anchor, clamped to `[0, 1]`.
/// * `candidate_scores`: `[batch_items, k]` float tensor, column `j` holds
///   the teacher similarity between anchor `i` and `candidate_index[i, j]`,
///   computed by the same `KernelMetric` used by the rest of the kernel.
///   Surfaced so downstream consumers can compute rank-correlation
///   diagnostics (Pearson, Spearman) against student logits without
///   re-running the scorer. Values are in `[0, 1]` for the built-in cosine
///   and entropy metrics. No gradient flows through this tensor under
///   `Autodiff`.
#[derive(Debug)]
pub struct RankingOutput<B: Backend> {
    pub candidate_index: BurnTensor<B, 2, TensorInt>,
    pub best_position: BurnTensor<B, 1, TensorInt>,
    pub top2_gap: BurnTensor<B, 1>,
    pub candidate_scores: BurnTensor<B, 2>,
}

/// Public wrapper around [`SpectralKernelBackend::ranking_score`].
///
/// `teacher` carries the `[N, P]` peak grids and `[N]` precursor masses for
/// the full cache. See [`RankingOutput`] for the shape of the returned
/// tensors.
///
/// ```
/// #[cfg(any(feature = "burn-cuda", feature = "burn-cpu"))]
/// fn run() {
///     use burn::tensor::{Tensor, TensorData};
///     use mass_spectrometry::burn::{
///         KernelMetric, LinearCosineMetric, SpectrumBatch, ranking_kernel,
///     };
///
///     #[cfg(feature = "burn-cuda")]
///     type B = burn::backend::Cuda<f32, i32>;
///     #[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
///     type B = burn::backend::Cpu<f32, i32>;
///     type Dev = burn::tensor::Device<B>;
///
///     let device = Dev::default();
///
///     // 4-spectrum teacher cache: each row has 2 peaks plus a precursor mass.
///     let teacher = SpectrumBatch::<B>::new(
///         Tensor::from_data(
///             TensorData::new(
///                 vec![100.0_f32, 200.0, 105.0, 205.0, 300.0, 400.0, 310.0, 410.0],
///                 [4, 2],
///             ),
///             &device,
///         ),
///         Tensor::from_data(
///             TensorData::new(
///                 vec![10.0_f32, 20.0, 11.0, 21.0, 5.0, 15.0, 6.0, 16.0],
///                 [4, 2],
///             ),
///             &device,
///         ),
///         Tensor::from_data(
///             TensorData::new(vec![500.0_f32, 510.0, 600.0, 610.0], [4]),
///             &device,
///         ),
///     );
///
///     // Score every row in the cache, sampling 2 non-self partners per anchor.
///     let config = LinearCosineMetric::ranking_config()
///         .with_batch_start(0)
///         .with_batch_items(4)
///         .with_candidates_per_anchor(2)
///         .with_mz_power(0.0)
///         .with_intensity_power(1.0)
///         .with_mz_tolerance(0.1)
///         .with_max_peaks(128)
///         .with_seed(42)
///         .with_epsilon(1.0e-8);
///
///     let output = ranking_kernel::<B, LinearCosineMetric>(teacher, config);
///     assert_eq!(output.candidate_index.dims(), [4, 2]);
///     assert_eq!(output.best_position.dims(), [4]);
///     assert_eq!(output.top2_gap.dims(), [4]);
///     assert_eq!(output.candidate_scores.dims(), [4, 2]);
/// }
/// #[cfg(not(any(feature = "burn-cuda", feature = "burn-cpu")))]
/// fn run() {}
///
/// fn main() {
///     run();
/// }
/// ```
pub fn ranking_kernel<B, M>(teacher: SpectrumBatch<B>, config: RankingConfig<M>) -> RankingOutput<B>
where
    B: SpectralKernelBackend<M>,
    M: KernelMetric,
{
    let (candidate_index, best_position, top2_gap, candidate_scores) =
        B::ranking_score(teacher.into_primitive(), config);

    RankingOutput {
        candidate_index: BurnTensor::new(candidate_index),
        best_position: BurnTensor::new(best_position),
        top2_gap: BurnTensor::from_primitive(TensorPrimitive::Float(top2_gap)),
        candidate_scores: BurnTensor::from_primitive(TensorPrimitive::Float(candidate_scores)),
    }
}
