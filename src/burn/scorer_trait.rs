use burn_cubecl::cubecl::prelude::*;

/// `#[cube]` trait implemented by every spectral similarity metric.
///
/// One implementation per metric marker type. The three launch drivers
/// (`paired_forward`, `cross_forward`, `ranking_forward`) take the marker as
/// a non-generic trait bound and call [`SpectralPairScorer::score_rows`] with
/// the launcher's own `F: Float` generic, so the per-pair scoring routine is
/// monomorphized per `(metric, float_precision)` at kernel-compile time with no
/// runtime dispatch.
///
/// Precursor tensors are passed even for non-modified variants. Those impls
/// simply ignore them. Keeping the signature uniform across metrics lets every
/// launcher call any scorer identically.
#[cube]
pub trait SpectralPairScorer: 'static + Send + Sync {
    /// Score the row pair `(left_row, right_row)`. Both side rows are laid out
    /// as `[batch, peak_width]` dense tensors with zero intensity marking
    /// padding. `eps` is the numerical-stability term added to denominators.
    /// `max_peaks` bounds compile-time scratch allocations inside the scorer.
    /// `weighted` toggles the entropy-weighted prepass (cosine impls ignore it).
    fn score_rows<F: Float>(
        left_mz: &Tensor<F>,
        left_intensity: &Tensor<F>,
        left_precursor: &Tensor<F>,
        left_row: usize,
        right_mz: &Tensor<F>,
        right_intensity: &Tensor<F>,
        right_precursor: &Tensor<F>,
        right_row: usize,
        mz_p: F,
        intensity_p: F,
        tolerance: F,
        eps: F,
        #[comptime] max_peaks: u32,
        #[comptime] weighted: bool,
    ) -> F;
}
