/// Rust-side static metadata for a GPU similarity metric.
///
/// Each marker that implements [`crate::burn::SpectralPairScorer`] also implements
/// this trait so callers can introspect the metric (stable name, whether it
/// reads the precursor cache, whether it uses entropy scoring) without touching
/// the GPU side. The marker types are zero-sized and selected as a type
/// parameter on the backend trait.
///
/// The trait also provides typed-default constructors for every config type
/// in [`crate::burn::api`]. They exist so callers can write
/// `LinearCosineMetric::cross_config()` instead of the turbofish-laden
/// `CrossConfig::<LinearCosineMetric>::new()`. Pure ergonomics, no behaviour
/// change.
pub trait KernelMetric: 'static + Send + Sync + Copy + core::fmt::Debug {
    /// Stable string identifier used in error messages.
    const NAME: &'static str;
    /// True when the metric reads the per-row precursor cache (modified variants).
    const USES_PRECURSOR: bool;
    /// True when the metric uses entropy-pair scoring instead of cosine product scoring.
    const IS_ENTROPY: bool;
    /// `CustomOpIr` name for the paired-shape kernel under Burn's fusion runtime.
    /// `&'static str` because `burn_ir::CustomOpIr::new` requires it.
    const PAIRED_FUSION_NAME: &'static str;
    /// `CustomOpIr` name for the cross-shape kernel.
    const CROSS_FUSION_NAME: &'static str;
    /// `CustomOpIr` name for the ranking-shape kernel.
    const RANKING_FUSION_NAME: &'static str;

    /// Typed default [`crate::burn::api::ScoringParams`].
    #[inline]
    fn scoring_params() -> crate::burn::api::ScoringParams<Self>
    where
        Self: Sized,
    {
        crate::burn::api::ScoringParams::new()
    }

    /// Typed default [`crate::burn::api::PairedConfig`].
    #[inline]
    fn paired_config() -> crate::burn::api::PairedConfig<Self>
    where
        Self: Sized,
    {
        crate::burn::api::PairedConfig::new()
    }

    /// Typed default [`crate::burn::api::CrossConfig`].
    #[inline]
    fn cross_config() -> crate::burn::api::CrossConfig<Self>
    where
        Self: Sized,
    {
        crate::burn::api::CrossConfig::new()
    }

    /// Typed default [`crate::burn::api::RankingConfig`].
    #[inline]
    fn ranking_config() -> crate::burn::api::RankingConfig<Self>
    where
        Self: Sized,
    {
        crate::burn::api::RankingConfig::new()
    }
}

/// Marker for the linear cosine similarity metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearCosineMetric;

impl KernelMetric for LinearCosineMetric {
    const NAME: &'static str = "linear_cosine";
    const USES_PRECURSOR: bool = false;
    const IS_ENTROPY: bool = false;
    const PAIRED_FUSION_NAME: &'static str = "spectral_paired_forward__linear_cosine";
    const CROSS_FUSION_NAME: &'static str = "spectral_cross_forward__linear_cosine";
    const RANKING_FUSION_NAME: &'static str = "spectral_ranking_forward__linear_cosine";
}

/// Marker for the modified (precursor-shifted) linear cosine similarity metric.
///
/// When `|left_precursor - right_precursor| > tolerance`, the scorer
/// additionally considers peak pairs shifted by the precursor delta, then
/// runs a DP on the resulting conflict graph to pick the optimal one-to-one
/// assignment.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModifiedLinearCosineMetric;

impl KernelMetric for ModifiedLinearCosineMetric {
    const NAME: &'static str = "modified_linear_cosine";
    const USES_PRECURSOR: bool = true;
    const IS_ENTROPY: bool = false;
    const PAIRED_FUSION_NAME: &'static str = "spectral_paired_forward__modified_linear_cosine";
    const CROSS_FUSION_NAME: &'static str = "spectral_cross_forward__modified_linear_cosine";
    const RANKING_FUSION_NAME: &'static str = "spectral_ranking_forward__modified_linear_cosine";
}

/// Marker trait satisfied only by entropy metrics. Used as a typestate
/// bound so that the entropy-only `with_weighted` builder method on the
/// config structs is inaccessible to cosine metrics at compile time.
pub trait EntropyMetric: KernelMetric {}

/// Marker for the linear entropy similarity metric (Li et al.).
///
/// The `weighted` toggle on the kernel config selects between unweighted
/// entropy and Shannon-entropy-reweighted entropy (the standard MS-entropy
/// variant). Both share this marker since the toggle is a runtime parameter,
/// not a metric class.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearEntropyMetric;

impl KernelMetric for LinearEntropyMetric {
    const NAME: &'static str = "linear_entropy";
    const USES_PRECURSOR: bool = false;
    const IS_ENTROPY: bool = true;
    const PAIRED_FUSION_NAME: &'static str = "spectral_paired_forward__linear_entropy";
    const CROSS_FUSION_NAME: &'static str = "spectral_cross_forward__linear_entropy";
    const RANKING_FUSION_NAME: &'static str = "spectral_ranking_forward__linear_entropy";
}

impl EntropyMetric for LinearEntropyMetric {}

/// Marker for the modified (precursor-shifted) linear entropy similarity metric.
///
/// Combines the precursor-shift candidate sweep + DP-on-paths assignment of
/// [`ModifiedLinearCosineMetric`] with the entropy pair scoring of
/// [`LinearEntropyMetric`]. The `weighted` toggle controls Shannon-entropy
/// re-weighting in the prepass.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModifiedLinearEntropyMetric;

impl KernelMetric for ModifiedLinearEntropyMetric {
    const NAME: &'static str = "modified_linear_entropy";
    const USES_PRECURSOR: bool = true;
    const IS_ENTROPY: bool = true;
    const PAIRED_FUSION_NAME: &'static str = "spectral_paired_forward__modified_linear_entropy";
    const CROSS_FUSION_NAME: &'static str = "spectral_cross_forward__modified_linear_entropy";
    const RANKING_FUSION_NAME: &'static str = "spectral_ranking_forward__modified_linear_entropy";
}

impl EntropyMetric for ModifiedLinearEntropyMetric {}
