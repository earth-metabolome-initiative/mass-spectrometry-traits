//! Builder-style config types for the spectral kernel API.
//!
//! Five types in two layers:
//!
//! - [`ScoringParams<M>`] and [`RankingWindow`] are the primitive builders.
//! - [`PairedConfig<M>`], [`CrossConfig<M>`], and [`RankingConfig<M>`] are
//!   the per-shape configs the public wrappers consume. Each owns a
//!   [`ScoringParams<M>`] (and ranking additionally owns a
//!   [`RankingWindow`]) and forwards its `with_*` setters via delegation.
//!
//! Every config is generic in the metric marker `M`. The entropy-only
//! `with_weighted` setter on each lives in a separate
//! `impl<M: EntropyMetric>` block so cosine call sites that try to toggle the
//! weighted prepass fail at compile time.

use core::marker::PhantomData;

use crate::burn::metrics::EntropyMetric;

use super::MAX_PEAKS_LIMIT;

#[inline]
#[track_caller]
fn check_max_peaks(value: usize) {
    assert!(
        value <= MAX_PEAKS_LIMIT,
        "max_peaks={value} exceeds the GPU scratch budget ({MAX_PEAKS_LIMIT})",
    );
}

/// Scoring parameters shared by the cross and ranking kernels.
///
/// Generic in the metric marker `M`. The entropy-only `with_weighted` setter
/// lives in a separate `impl` block bound by [`EntropyMetric`], so passing
/// `with_weighted(true)` to a cosine config fails at compile time instead of
/// being silently dropped.
///
/// ```ignore
/// // Either type-annotate the binding:
/// let params: ScoringParams<LinearCosineMetric> = ScoringParams::new()
///     .with_mz_power(0.15)
///     .with_intensity_power(0.7);
///
/// // ...or use the typed-default helper on the metric marker (preferred):
/// let params = LinearCosineMetric::scoring_params()
///     .with_mz_power(0.15)
///     .with_intensity_power(0.7);
///
/// // .with_weighted(true) // <- compile error: cosine metric, no such method
/// ```
///
/// `with_max_peaks` validates `value <= MAX_PEAKS_LIMIT` so the runtime
/// panic surfaces at the builder rather than inside the kernel launcher.
///
/// **Type inference note**: `ScoringParams::new()` leaves `M` unresolved
/// until it's consumed by something that pins it (typically a typed
/// `let` binding, or feeding the value into a `CrossConfig<KnownMetric>`).
/// If you write `let p = ScoringParams::new();` with no further context,
/// the compiler reports an unconstrained type parameter. The
/// [`crate::burn::metrics::KernelMetric::scoring_params`] helper avoids
/// this entirely.
#[derive(Clone, Copy, Debug)]
pub struct ScoringParams<M> {
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    max_peaks: usize,
    epsilon: f32,
    weighted: bool,
    _marker: PhantomData<M>,
}

impl<M> Default for ScoringParams<M> {
    fn default() -> Self {
        Self {
            mz_power: 0.0,
            intensity_power: 1.0,
            mz_tolerance: 0.02,
            max_peaks: 128,
            epsilon: 1.0e-8,
            weighted: false,
            _marker: PhantomData,
        }
    }
}

impl<M> ScoringParams<M> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    #[must_use]
    pub fn with_mz_power(mut self, value: f32) -> Self {
        self.mz_power = value;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_intensity_power(mut self, value: f32) -> Self {
        self.intensity_power = value;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_mz_tolerance(mut self, value: f32) -> Self {
        self.mz_tolerance = value;
        self
    }

    #[inline]
    #[must_use]
    #[track_caller]
    pub fn with_max_peaks(mut self, value: usize) -> Self {
        check_max_peaks(value);
        self.max_peaks = value;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_epsilon(mut self, value: f32) -> Self {
        self.epsilon = value;
        self
    }

    #[inline]
    pub fn mz_power(&self) -> f32 {
        self.mz_power
    }

    #[inline]
    pub fn intensity_power(&self) -> f32 {
        self.intensity_power
    }

    #[inline]
    pub fn mz_tolerance(&self) -> f32 {
        self.mz_tolerance
    }

    #[inline]
    pub fn max_peaks(&self) -> usize {
        self.max_peaks
    }

    #[inline]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    #[inline]
    pub fn weighted(&self) -> bool {
        self.weighted
    }
}

impl<M: EntropyMetric> ScoringParams<M> {
    /// Toggle the entropy-weighted prepass. Only available on entropy metrics.
    #[inline]
    #[must_use]
    pub fn with_weighted(mut self, value: bool) -> Self {
        self.weighted = value;
        self
    }
}

/// Ranking-specific launch window: which slice of the teacher cache to score
/// and how to seed the candidate sampler.
///
/// ```ignore
/// let window = RankingWindow::new()
///     .with_batch_start(0)
///     .with_batch_items(10)
///     .with_candidates_per_anchor(7)
///     .with_seed(12_345);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RankingWindow {
    batch_start: usize,
    batch_items: usize,
    candidates_per_anchor: usize,
    seed: u64,
}

impl Default for RankingWindow {
    fn default() -> Self {
        Self {
            batch_start: 0,
            batch_items: 0,
            candidates_per_anchor: 2,
            seed: 0,
        }
    }
}

impl RankingWindow {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    #[must_use]
    pub fn with_batch_start(mut self, value: usize) -> Self {
        self.batch_start = value;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_batch_items(mut self, value: usize) -> Self {
        self.batch_items = value;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_candidates_per_anchor(mut self, value: usize) -> Self {
        self.candidates_per_anchor = value;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_seed(mut self, value: u64) -> Self {
        self.seed = value;
        self
    }

    #[inline]
    pub fn batch_start(&self) -> usize {
        self.batch_start
    }

    #[inline]
    pub fn batch_items(&self) -> usize {
        self.batch_items
    }

    #[inline]
    pub fn candidates_per_anchor(&self) -> usize {
        self.candidates_per_anchor
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Numerics for the paired (1-to-1) kernel. Generic in the metric marker
/// `M`. The scoring scalars (`mz_power`, `intensity_power`, `mz_tolerance`)
/// are passed as per-row tensors to the paired kernel, so they live there,
/// not here. The entropy-only `with_weighted` setter is gated on
/// [`EntropyMetric`].
///
/// ```ignore
/// let config: PairedConfig<LinearCosineMetric> = PairedConfig::new()
///     .with_max_peaks(128)
///     .with_epsilon(1.0e-8);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PairedConfig<M> {
    max_peaks: usize,
    epsilon: f32,
    weighted: bool,
    _marker: PhantomData<M>,
}

impl<M> Default for PairedConfig<M> {
    fn default() -> Self {
        Self {
            max_peaks: 128,
            epsilon: 1.0e-8,
            weighted: false,
            _marker: PhantomData,
        }
    }
}

impl<M> PairedConfig<M> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    #[must_use]
    #[track_caller]
    pub fn with_max_peaks(mut self, value: usize) -> Self {
        check_max_peaks(value);
        self.max_peaks = value;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_epsilon(mut self, value: f32) -> Self {
        self.epsilon = value;
        self
    }

    #[inline]
    pub fn max_peaks(&self) -> usize {
        self.max_peaks
    }

    #[inline]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    #[inline]
    pub fn weighted(&self) -> bool {
        self.weighted
    }
}

impl<M: EntropyMetric> PairedConfig<M> {
    /// Toggle the entropy-weighted prepass. Only available on entropy metrics.
    #[inline]
    #[must_use]
    pub fn with_weighted(mut self, value: bool) -> Self {
        self.weighted = value;
        self
    }
}

/// Cross / all-pairs scoring configuration. Generic in `M`, owns one
/// [`ScoringParams<M>`].
///
/// All MxN pairs share the same scoring scalars. Broadcasting per-pair
/// tensors onto the 2D grid would force an `[M*N]` parameter tensor, so the
/// cross kernel takes them as plain scalars. `with_weighted` is gated on
/// [`EntropyMetric`].
///
/// ```ignore
/// let scoring: ScoringParams<LinearCosineMetric> =
///     ScoringParams::new().with_mz_power(0.15);
/// let config = CrossConfig::from_scoring(scoring);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CrossConfig<M> {
    scoring: ScoringParams<M>,
}

impl<M> Default for CrossConfig<M> {
    fn default() -> Self {
        Self {
            scoring: ScoringParams::default(),
        }
    }
}

impl<M> CrossConfig<M> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn from_scoring(scoring: ScoringParams<M>) -> Self {
        Self { scoring }
    }

    #[inline]
    pub fn scoring(&self) -> &ScoringParams<M> {
        &self.scoring
    }

    #[inline]
    #[must_use]
    pub fn with_mz_power(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_mz_power(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_intensity_power(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_intensity_power(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_mz_tolerance(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_mz_tolerance(value);
        self
    }

    #[inline]
    #[must_use]
    #[track_caller]
    pub fn with_max_peaks(mut self, value: usize) -> Self {
        self.scoring = self.scoring.with_max_peaks(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_epsilon(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_epsilon(value);
        self
    }

    #[inline]
    pub fn mz_power(&self) -> f32 {
        self.scoring.mz_power
    }

    #[inline]
    pub fn intensity_power(&self) -> f32 {
        self.scoring.intensity_power
    }

    #[inline]
    pub fn mz_tolerance(&self) -> f32 {
        self.scoring.mz_tolerance
    }

    #[inline]
    pub fn max_peaks(&self) -> usize {
        self.scoring.max_peaks
    }

    #[inline]
    pub fn epsilon(&self) -> f32 {
        self.scoring.epsilon
    }

    #[inline]
    pub fn weighted(&self) -> bool {
        self.scoring.weighted
    }
}

impl<M: EntropyMetric> CrossConfig<M> {
    /// Toggle the entropy-weighted prepass. Only available on entropy metrics.
    #[inline]
    #[must_use]
    pub fn with_weighted(mut self, value: bool) -> Self {
        self.scoring = self.scoring.with_weighted(value);
        self
    }
}

/// Ranking-kernel configuration. Generic in `M`, owns a
/// [`ScoringParams<M>`] plus a [`RankingWindow`]. `with_weighted` is gated
/// on [`EntropyMetric`].
///
/// ```ignore
/// let scoring: ScoringParams<LinearCosineMetric> =
///     ScoringParams::new().with_mz_power(0.15);
/// let window = RankingWindow::new()
///     .with_batch_start(0)
///     .with_batch_items(10)
///     .with_candidates_per_anchor(7)
///     .with_seed(12_345);
/// let config = RankingConfig::from_parts(scoring, window);
/// ```
///
/// `effective_candidates_per_anchor()` returns the clamped count actually
/// used by the kernel: `candidates_per_anchor.max(2).min(batch_items - 1)`.
#[derive(Clone, Copy, Debug)]
pub struct RankingConfig<M> {
    scoring: ScoringParams<M>,
    window: RankingWindow,
}

impl<M> Default for RankingConfig<M> {
    fn default() -> Self {
        Self {
            scoring: ScoringParams::default(),
            window: RankingWindow::default(),
        }
    }
}

impl<M> RankingConfig<M> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn from_parts(scoring: ScoringParams<M>, window: RankingWindow) -> Self {
        Self { scoring, window }
    }

    #[inline]
    pub fn scoring(&self) -> &ScoringParams<M> {
        &self.scoring
    }

    #[inline]
    pub fn window(&self) -> &RankingWindow {
        &self.window
    }

    #[inline]
    #[must_use]
    pub fn with_mz_power(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_mz_power(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_intensity_power(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_intensity_power(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_mz_tolerance(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_mz_tolerance(value);
        self
    }

    #[inline]
    #[must_use]
    #[track_caller]
    pub fn with_max_peaks(mut self, value: usize) -> Self {
        self.scoring = self.scoring.with_max_peaks(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_epsilon(mut self, value: f32) -> Self {
        self.scoring = self.scoring.with_epsilon(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_batch_start(mut self, value: usize) -> Self {
        self.window = self.window.with_batch_start(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_batch_items(mut self, value: usize) -> Self {
        self.window = self.window.with_batch_items(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_candidates_per_anchor(mut self, value: usize) -> Self {
        self.window = self.window.with_candidates_per_anchor(value);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_seed(mut self, value: u64) -> Self {
        self.window = self.window.with_seed(value);
        self
    }

    #[inline]
    pub fn mz_power(&self) -> f32 {
        self.scoring.mz_power
    }

    #[inline]
    pub fn intensity_power(&self) -> f32 {
        self.scoring.intensity_power
    }

    #[inline]
    pub fn mz_tolerance(&self) -> f32 {
        self.scoring.mz_tolerance
    }

    #[inline]
    pub fn max_peaks(&self) -> usize {
        self.scoring.max_peaks
    }

    #[inline]
    pub fn epsilon(&self) -> f32 {
        self.scoring.epsilon
    }

    #[inline]
    pub fn weighted(&self) -> bool {
        self.scoring.weighted
    }

    #[inline]
    pub fn batch_start(&self) -> usize {
        self.window.batch_start
    }

    #[inline]
    pub fn batch_items(&self) -> usize {
        self.window.batch_items
    }

    #[inline]
    pub fn candidates_per_anchor(&self) -> usize {
        self.window.candidates_per_anchor
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.window.seed
    }

    /// Effective candidate count actually used by the kernel: clamped to the
    /// range `[2, batch_items - 1]`.
    #[inline]
    pub fn effective_candidates_per_anchor(&self) -> usize {
        self.window
            .candidates_per_anchor
            .max(2)
            .min(self.window.batch_items.saturating_sub(1))
    }
}

impl<M: EntropyMetric> RankingConfig<M> {
    /// Toggle the entropy-weighted prepass. Only available on entropy metrics.
    #[inline]
    #[must_use]
    pub fn with_weighted(mut self, value: bool) -> Self {
        self.scoring = self.scoring.with_weighted(value);
        self
    }
}
