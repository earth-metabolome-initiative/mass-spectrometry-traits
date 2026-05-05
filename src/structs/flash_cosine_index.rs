//! Flash inverted m/z index for cosine spectral similarity.
//!
//! Provides O(Q_peaks * log(P_total)) library-scale search instead of
//! O(N * pairwise_cost). Exact equivalence to [`super::LinearCosine`] on
//! well-separated spectra (consecutive peaks > 2 * tolerance apart).

use alloc::vec::Vec;
#[cfg(feature = "rayon")]
use core::ops::Range;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::cosine_common::{
    ensure_finite, normalized_peak_products, validate_non_negative_tolerance,
    validate_well_separated,
};
use super::flash_common::{
    DEFAULT_COSINE_SPECTRUM_BLOCK_SIZE, DirectThresholdSearch, FlashIndex, FlashIndexBuildPhase,
    FlashIndexBuildProgress, FlashKernel, FlashSearchResult, NoopFlashIndexBuildProgress,
    PepmassFilter, PreparedFlashSpectra, PreparedFlashSpectrum, SearchState,
    SpectrumBlockProductIndex, SpectrumBlockUpperBoundIndex, SpectrumIdMap, TopKSearchResults,
    TopKSearchState, convert_flash_value, convert_flash_values, flash_values_to_f64,
    progress_len_from_size_hint, reorder_prepared_spectra_by_signature,
};
use super::similarity_errors::{
    SimilarityComputationError, SimilarityConfigError, SpectraIndexSetupError,
};
use crate::traits::{SpectraIndex, Spectrum, SpectrumFloat};

const THRESHOLD_INDEX_ONE_SIDED_MIN_THRESHOLD: f64 = 0.85;
// ---------------------------------------------------------------------------
// CosineKernel
// ---------------------------------------------------------------------------

pub(crate) struct CosineKernel;

/// Norm stored per library spectrum.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CosineNorm(f64);

impl FlashKernel for CosineKernel {
    type SpectrumMeta = CosineNorm;

    #[inline]
    fn spectrum_meta<P: SpectrumFloat>(peak_data: &[P]) -> CosineNorm {
        let sum_sq: f64 = peak_data
            .iter()
            .map(|&v| {
                let v = v.to_f64();
                v * v
            })
            .sum();
        CosineNorm(sum_sq.sqrt())
    }

    #[inline]
    fn pair_score(query: f64, library: f64) -> f64 {
        query * library
    }

    #[inline]
    fn finalize(
        raw: f64,
        _n_matches: usize,
        query_meta: &CosineNorm,
        lib_meta: &CosineNorm,
    ) -> f64 {
        let denom = query_meta.0 * lib_meta.0;
        if denom == 0.0 {
            return 0.0;
        }
        let sim = raw / denom;
        if sim > 1.0 { 1.0 } else { sim }
    }
}

fn prepare_cosine_query<S>(
    query: &S,
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
) -> Result<(Vec<f64>, Vec<f64>), SimilarityComputationError>
where
    S: Spectrum,
{
    let mz_vals: Vec<f64> = query.mz().map(SpectrumFloat::to_f64).collect();
    let data_vals = normalized_peak_products(query, mz_power, intensity_power)?;

    validate_well_separated(&mz_vals, tolerance, "query spectrum")?;

    Ok((mz_vals, data_vals))
}

/// Validate shared cosine index construction parameters.
fn validate_cosine_index_config(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
) -> Result<(), FlashCosineIndexError> {
    validate_non_negative_tolerance(mz_tolerance).map_err(FlashCosineIndexError::Config)?;
    ensure_finite(mz_power, "mz_power").map_err(FlashCosineIndexError::Computation)?;
    ensure_finite(intensity_power, "intensity_power")
        .map_err(FlashCosineIndexError::Computation)?;
    Ok(())
}

/// Validate cosine threshold-index construction parameters.
fn validate_cosine_threshold_index_config(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    score_threshold: f64,
) -> Result<(), FlashCosineIndexError> {
    validate_cosine_index_config(mz_power, intensity_power, mz_tolerance)?;
    ensure_finite(score_threshold, "score_threshold")
        .map_err(FlashCosineIndexError::Computation)?;
    Ok(())
}

#[cfg(feature = "rayon")]
#[derive(Clone, Copy)]
struct CosineSelfSimilarityProfile {
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    score_threshold: f64,
    top_k: usize,
    pepmass_filter: PepmassFilter,
}

#[cfg(feature = "rayon")]
fn validate_cosine_self_similarity_index_config(
    profile: CosineSelfSimilarityProfile,
) -> Result<(), FlashCosineIndexError> {
    validate_cosine_threshold_index_config(
        profile.mz_power,
        profile.intensity_power,
        profile.mz_tolerance,
        profile.score_threshold,
    )?;
    if profile.top_k == 0 {
        return Err(FlashCosineIndexError::Config(
            SimilarityConfigError::InvalidParameter("top_k"),
        ));
    }
    if !profile.pepmass_filter.is_enabled() {
        return Err(FlashCosineIndexError::Config(
            SimilarityConfigError::InvalidParameter("pepmass_filter"),
        ));
    }
    Ok(())
}

/// Prepare one library spectrum into the packed representation used by the
/// shared Flash builder.
fn prepare_cosine_spectrum<P, S>(
    spectrum: &S,
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
) -> Result<PreparedFlashSpectrum<P>, FlashCosineIndexError>
where
    P: SpectrumFloat,
    S: Spectrum,
{
    let mz_vals = convert_flash_values(spectrum.mz().map(SpectrumFloat::to_f64), "mz")
        .map_err(FlashCosineIndexError::Computation)?;
    let mz_vals_f64 = flash_values_to_f64(&mz_vals);
    validate_well_separated(&mz_vals_f64, tolerance, "library spectrum")
        .map_err(FlashCosineIndexError::Computation)?;

    let data_vals_f64 = normalized_peak_products(spectrum, mz_power, intensity_power)
        .map_err(FlashCosineIndexError::Computation)?;
    let data_vals = convert_flash_values(data_vals_f64, "peak_product")
        .map_err(FlashCosineIndexError::Computation)?;

    let precursor_mz = convert_flash_value(
        ensure_finite(spectrum.precursor_mz().to_f64(), "precursor_mz")
            .map_err(FlashCosineIndexError::Computation)?,
        "precursor_mz",
    )
    .map_err(FlashCosineIndexError::Computation)?;

    Ok(PreparedFlashSpectrum {
        precursor_mz,
        mz: mz_vals,
        data: data_vals,
    })
}

/// Prepare a library sequentially from any borrowed-spectrum iterator.
fn prepare_cosine_library<'a, P, S>(
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
    spectra: impl IntoIterator<Item = &'a S>,
    progress: &(impl FlashIndexBuildProgress + ?Sized),
) -> Result<PreparedFlashSpectra<P>, FlashCosineIndexError>
where
    P: SpectrumFloat,
    S: Spectrum + 'a,
{
    let spectra = spectra.into_iter();
    progress.start_phase(
        FlashIndexBuildPhase::PrepareSpectra,
        progress_len_from_size_hint(spectra.size_hint()),
    );
    spectra
        .map(|spectrum| {
            let result =
                prepare_cosine_spectrum::<P, S>(spectrum, mz_power, intensity_power, tolerance);
            progress.inc(1);
            result
        })
        .collect()
}

/// Prepare a library in parallel from a Rayon-compatible borrowed-spectrum
/// collection.
#[cfg(feature = "rayon")]
fn prepare_cosine_library_parallel<'a, P, S, I>(
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
    spectra: I,
    progress: &(impl FlashIndexBuildProgress + Sync + ?Sized),
) -> Result<PreparedFlashSpectra<P>, FlashCosineIndexError>
where
    P: SpectrumFloat + Send,
    S: Spectrum + Sync + 'a,
    I: IntoParallelIterator<Item = &'a S>,
{
    let spectra = spectra.into_par_iter();
    progress.start_phase(
        FlashIndexBuildPhase::PrepareSpectra,
        spectra.opt_len().and_then(|len| u64::try_from(len).ok()),
    );
    spectra
        .map(|spectrum| {
            let result =
                prepare_cosine_spectrum::<P, S>(spectrum, mz_power, intensity_power, tolerance);
            progress.inc(1);
            result
        })
        .collect()
}

fn build_cosine_block_upper_bounds<P: SpectrumFloat>(
    prepared: &[PreparedFlashSpectrum<P>],
    mz_tolerance: f64,
) -> Result<SpectrumBlockUpperBoundIndex, SimilarityComputationError> {
    build_cosine_block_upper_bounds_with_block_size(
        prepared,
        mz_tolerance,
        DEFAULT_COSINE_SPECTRUM_BLOCK_SIZE,
    )
}

fn build_cosine_block_upper_bounds_with_block_size<P: SpectrumFloat>(
    prepared: &[PreparedFlashSpectrum<P>],
    mz_tolerance: f64,
    block_size: usize,
) -> Result<SpectrumBlockUpperBoundIndex, SimilarityComputationError> {
    SpectrumBlockUpperBoundIndex::build(prepared, mz_tolerance, block_size, |_, spectrum| {
        let norm = CosineKernel::spectrum_meta(&spectrum.data).0;
        if norm == 0.0 {
            return Ok(alloc::vec![0.0; spectrum.data.len()]);
        }

        Ok(spectrum
            .data
            .iter()
            .map(|&data| data.to_f64().abs() / norm)
            .collect())
    })
}

#[cfg(feature = "rayon")]
fn self_similarity_block_size(n_spectra: usize, top_k: usize, score_threshold: f64) -> usize {
    const MIN_BLOCK_SIZE: usize = 128;
    let mut block_size = DEFAULT_COSINE_SPECTRUM_BLOCK_SIZE;

    if score_threshold >= 0.95 {
        block_size /= 4;
    } else if score_threshold >= 0.90 {
        block_size /= 2;
    }

    if top_k <= 4 {
        block_size /= 2;
    } else if top_k <= 16 {
        block_size = block_size.saturating_mul(3) / 4;
    }

    if n_spectra >= 1_000_000 && top_k <= 16 && score_threshold >= 0.90 {
        block_size = block_size.min(256);
    }

    block_size.clamp(MIN_BLOCK_SIZE, DEFAULT_COSINE_SPECTRUM_BLOCK_SIZE)
}

/// Run exact cosine top-k over a prepared query while using the current kth
/// score as an adaptive pruning bound.
///
/// Query peaks are scanned from largest to smallest contribution. Once the
/// remaining query suffix norm cannot reach the current pruning score, no
/// unseen spectrum can enter the result set, so the index scan stops early.
fn for_each_cosine_top_k_prepared<P, Q, Emit>(
    inner: &FlashIndex<CosineKernel, P>,
    search: DirectThresholdSearch<'_, CosineKernel, Q>,
    k: usize,
    state: &mut SearchState,
    top_k_state: &mut TopKSearchState,
    emit: Emit,
) where
    P: SpectrumFloat + Sync,
    Q: SpectrumFloat,
    Emit: FnMut(FlashSearchResult),
{
    state.reset_diagnostics();
    let mut top_k = TopKSearchResults::new(k, search.score_threshold, top_k_state);
    if k == 0 || search.score_threshold > 1.0 || inner.n_spectra == 0 || search.query_mz.is_empty()
    {
        state.add_results_emitted(top_k.len());
        top_k.emit(emit);
        return;
    }

    let query_norm = search.query_meta.0;
    if query_norm == 0.0 {
        state.add_results_emitted(top_k.len());
        top_k.emit(emit);
        return;
    }

    state.prepare_threshold_order(search.query_data);
    state.ensure_candidate_capacity(inner.n_spectra as usize);

    let query_order_len = state.query_order().len();
    for order_position in 0..query_order_len {
        let query_index = state.query_order()[order_position];
        let qmz = search.query_mz[query_index];

        let visited =
            inner.for_each_product_spectrum_in_window(qmz, search.query_precursor_mz, |spec_id| {
                if state.is_candidate(spec_id) {
                    return;
                }
                state.mark_candidate(spec_id);

                let lib_meta = inner.spectrum_meta(spec_id);
                if lib_meta.0 == 0.0 {
                    return;
                }

                let target_raw = top_k.pruning_score() * query_norm * lib_meta.0;
                let (raw, count) =
                    inner.direct_score_for_spectrum(search.query_mz, search.query_data, spec_id);
                if raw < target_raw {
                    return;
                }

                let score = CosineKernel::finalize(raw, count, search.query_meta, lib_meta);
                if score > 0.0 {
                    top_k.push(FlashSearchResult {
                        spectrum_id: inner.public_spectrum_id(spec_id),
                        score,
                        n_matches: count,
                    });
                }
            });
        state.add_product_postings_visited(visited);

        let processed_prefix_len = order_position + 1;
        let pruning_score = top_k.pruning_score();
        if state.query_suffix_bound_at(processed_prefix_len) < pruning_score * query_norm {
            break;
        }
    }

    state.reset_candidates();
    state.add_results_emitted(top_k.len());
    top_k.emit(emit);
}

// ---------------------------------------------------------------------------
// FlashCosineIndex
// ---------------------------------------------------------------------------

/// Flash inverted m/z index for cosine spectral similarity.
///
/// Build once from a library of spectra, then search many queries against it.
/// Produces the same scores as [`super::LinearCosine`] on well-separated
/// spectra.
///
/// # Example
///
/// ```
/// use mass_spectrometry::prelude::*;
///
/// let library: [GenericSpectrum; 2] = [
///     GenericSpectrum::cocaine().unwrap(),
///     GenericSpectrum::glucose().unwrap(),
/// ];
/// let index = FlashCosineIndex::<f64>::new(1.0, 1.0, 0.1, library.iter())
///     .expect("index build should succeed");
///
/// let query: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
/// let results = index.search(&query).expect("search should succeed");
/// assert!(results.iter().any(|r| r.spectrum_id == 0 && r.score > 0.99));
/// ```
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct FlashCosineIndex<P: SpectrumFloat = f64> {
    inner: FlashIndex<CosineKernel, P>,
    mz_power: f64,
    intensity_power: f64,
}

impl<P: SpectrumFloat + Sync> FlashCosineIndex<P> {
    /// Returns the m/z power used for scoring.
    #[inline]
    pub fn mz_power(&self) -> f64 {
        self.mz_power
    }

    /// Returns the intensity power used for scoring.
    #[inline]
    pub fn intensity_power(&self) -> f64 {
        self.intensity_power
    }

    /// Returns the m/z tolerance used for matching.
    #[inline]
    pub fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    /// Returns the number of library spectra in the index.
    #[inline]
    pub fn n_spectra(&self) -> u32 {
        self.inner.n_spectra
    }

    /// Build a new cosine flash index from an iterator of spectra.
    ///
    /// Each library spectrum must satisfy the well-separated precondition:
    /// consecutive peaks must be more than `2 * mz_tolerance` apart.
    /// Spectra are reordered internally by a deterministic peak-signature
    /// heuristic; returned spectrum ids and indexed query ids remain in the
    /// original insertion order.
    ///
    /// # Errors
    ///
    /// - [`SimilarityConfigError`] if numeric parameters are invalid.
    /// - [`SimilarityComputationError`] if any spectrum violates the
    ///   well-separated precondition or contains non-representable values.
    pub fn new<'a, S>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        spectra: impl IntoIterator<Item = &'a S>,
    ) -> Result<Self, FlashCosineIndexError>
    where
        S: Spectrum + 'a,
    {
        let progress = NoopFlashIndexBuildProgress;
        Self::new_with_progress(mz_power, intensity_power, mz_tolerance, spectra, &progress)
    }

    /// Build a new cosine flash index while reporting construction progress.
    ///
    /// The `progress` sink receives coarse phase changes and per-spectrum
    /// preparation ticks. With the `indicatif` feature enabled, pass an
    /// `indicatif::ProgressBar` to display progress during construction.
    pub fn new_with_progress<'a, S, G>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        spectra: impl IntoIterator<Item = &'a S>,
        progress: &G,
    ) -> Result<Self, FlashCosineIndexError>
    where
        S: Spectrum + 'a,
        G: FlashIndexBuildProgress + ?Sized,
    {
        validate_cosine_index_config(mz_power, intensity_power, mz_tolerance)?;
        let prepared = prepare_cosine_library::<P, S>(
            mz_power,
            intensity_power,
            mz_tolerance,
            spectra,
            progress,
        )?;
        Self::from_prepared_library_with_builder(
            mz_power,
            intensity_power,
            mz_tolerance,
            prepared,
            progress,
            FlashIndex::<CosineKernel, P>::build_with_spectrum_id_map_and_progress,
        )
    }

    fn from_prepared_library_with_builder<G, BuildInner>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        prepared: PreparedFlashSpectra<P>,
        progress: &G,
        build_inner: BuildInner,
    ) -> Result<Self, FlashCosineIndexError>
    where
        G: FlashIndexBuildProgress + ?Sized,
        BuildInner: FnOnce(
            f64,
            PreparedFlashSpectra<P>,
            SpectrumIdMap,
            &G,
        )
            -> Result<FlashIndex<CosineKernel, P>, SimilarityComputationError>,
    {
        progress.start_phase(FlashIndexBuildPhase::ReorderSpectra, Some(1));
        let (prepared, spectrum_id_map) =
            reorder_prepared_spectra_by_signature(prepared, mz_tolerance)
                .map_err(FlashCosineIndexError::Computation)?;
        progress.inc(1);
        let inner = build_inner(mz_tolerance, prepared, spectrum_id_map, progress)
            .map_err(FlashCosineIndexError::Computation)?;
        progress.finish();

        Ok(Self {
            inner,
            mz_power,
            intensity_power,
        })
    }

    /// Build a new cosine flash index with Rayon-backed library preparation
    /// and internal index sorting.
    /// Uses the same internal spectrum reordering as [`Self::new`].
    ///
    /// This constructor is available with the `rayon` feature. It accepts any
    /// Rayon-compatible collection yielding borrowed spectra, such as `&[S]`
    /// or `&Vec<S>`.
    #[cfg(feature = "rayon")]
    pub fn new_parallel<'a, S, I>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        spectra: I,
    ) -> Result<Self, FlashCosineIndexError>
    where
        P: Send,
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
    {
        let progress = NoopFlashIndexBuildProgress;
        Self::new_parallel_with_progress(
            mz_power,
            intensity_power,
            mz_tolerance,
            spectra,
            &progress,
        )
    }

    /// Build a new cosine flash index with Rayon-backed preparation and
    /// progress reporting.
    #[cfg(feature = "rayon")]
    pub fn new_parallel_with_progress<'a, S, I, G>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        spectra: I,
        progress: &G,
    ) -> Result<Self, FlashCosineIndexError>
    where
        P: Send,
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
        G: FlashIndexBuildProgress + Sync + ?Sized,
    {
        validate_cosine_index_config(mz_power, intensity_power, mz_tolerance)?;
        let prepared = prepare_cosine_library_parallel::<P, S, I>(
            mz_power,
            intensity_power,
            mz_tolerance,
            spectra,
            progress,
        )?;
        Self::from_prepared_library_with_builder(
            mz_power,
            intensity_power,
            mz_tolerance,
            prepared,
            progress,
            FlashIndex::<CosineKernel, P>::build_parallel_with_spectrum_id_map_and_progress,
        )
    }

    /// Create a [`SearchState`] sized for this index, suitable for reuse
    /// across multiple queries to avoid per-query allocation.
    pub fn new_search_state(&self) -> SearchState {
        self.inner.new_search_state()
    }

    /// Direct (unshifted) search against the library.
    ///
    /// Returns results for all library spectra that share at least one
    /// matching peak with the query.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityComputationError`] if the query violates the
    /// well-separated precondition or contains non-representable values.
    pub fn search<S>(&self, query: &S) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_precursor_mz = self.inner.query_precursor_mz_for_filter(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        Ok(self
            .inner
            .search_direct(&query_mz, &query_data, &query_meta, query_precursor_mz))
    }

    /// Direct search using a caller-provided [`SearchState`] to avoid
    /// per-query allocation. Create one via [`Self::new_search_state`].
    pub fn search_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_precursor_mz = self.inner.query_precursor_mz_for_filter(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        Ok(self.inner.search_direct_with_state(
            &query_mz,
            &query_data,
            &query_meta,
            query_precursor_mz,
            state,
        ))
    }

    /// Direct search that returns the best `k` results by descending score.
    ///
    /// Ties are ordered by descending match count and then ascending spectrum
    /// id to keep output deterministic. This is exact: it returns the same
    /// first `k` entries as sorting [`Self::search`] by that ranking.
    pub fn search_top_k<S>(
        &self,
        query: &S,
        k: usize,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut state = self.new_search_state();
        self.search_top_k_with_state(query, k, &mut state)
    }

    /// Top-k direct search using caller-provided scratch state.
    pub fn search_top_k_with_state<S>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search_top_k_threshold_with_state(query, k, 0.0, state)
    }

    /// Stream the best `k` direct cosine results using caller-provided
    /// scratch state.
    pub fn for_each_top_k_with_state<S, Emit>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        self.for_each_top_k_threshold_with_state(query, k, 0.0, state, top_k_state, emit)
    }

    /// Direct search that returns the best `k` results with
    /// `score >= score_threshold`.
    ///
    /// This threads `score_threshold` into the cosine search path. Once `k`
    /// results have been found, the current kth score becomes
    /// the pruning cutoff for the remaining index scan.
    ///
    /// # Example
    ///
    /// ```
    /// use mass_spectrometry::prelude::*;
    ///
    /// let mut left: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
    /// left.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();
    /// let mut right: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
    /// right.add_peaks([(100.05, 10.0), (200.05, 20.0)]).unwrap();
    ///
    /// let spectra = vec![left, right];
    /// let index = FlashCosineIndex::<f64>::new(0.0, 1.0, 0.1, &spectra).unwrap();
    /// let hits = index.search_top_k_threshold(&spectra[0], 1, 0.8).unwrap();
    ///
    /// assert_eq!(hits.len(), 1);
    /// assert_eq!(hits[0].spectrum_id, 0);
    /// ```
    pub fn search_top_k_threshold<S>(
        &self,
        query: &S,
        k: usize,
        score_threshold: f64,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut state = self.new_search_state();
        self.search_top_k_threshold_with_state(query, k, score_threshold, &mut state)
    }

    /// Thresholded top-k direct search using caller-provided scratch state.
    pub fn search_top_k_threshold_with_state<S>(
        &self,
        query: &S,
        k: usize,
        score_threshold: f64,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        ensure_finite(score_threshold, "score_threshold")?;
        let mut top_k_state = TopKSearchState::new();
        let mut results = Vec::new();
        self.for_each_top_k_threshold_with_state(
            query,
            k,
            score_threshold,
            state,
            &mut top_k_state,
            |result| results.push(result),
        )?;
        Ok(results)
    }

    /// Thresholded top-k direct search that emits each selected result.
    ///
    /// Reuse both `state` and `top_k_state` across queries in high-throughput
    /// workloads to avoid per-query scratch allocation. The search is exact
    /// and raises its pruning cutoff to the current kth score as soon as the
    /// bounded result set is full.
    pub fn for_each_top_k_threshold_with_state<S, Emit>(
        &self,
        query: &S,
        k: usize,
        score_threshold: f64,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        ensure_finite(score_threshold, "score_threshold")?;
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_precursor_mz = self.inner.query_precursor_mz_for_filter(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        for_each_cosine_top_k_prepared(
            &self.inner,
            DirectThresholdSearch {
                query_mz: &query_mz,
                query_data: &query_data,
                query_meta: &query_meta,
                score_threshold,
                query_precursor_mz,
            },
            k,
            state,
            top_k_state,
            emit,
        );
        Ok(())
    }

    /// Direct search that returns only results with `score >= score_threshold`.
    ///
    /// Unlike filtering [`Self::search`] afterward, this threads the threshold
    /// into the index scan and prunes candidates whose best possible remaining
    /// cosine score cannot reach the threshold.
    ///
    /// Thresholds less than or equal to zero are equivalent to [`Self::search`].
    /// Thresholds greater than one return no results.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityComputationError`] if the query violates the
    /// well-separated precondition, contains non-representable values, or if
    /// `score_threshold` is not finite.
    pub fn search_threshold<S>(
        &self,
        query: &S,
        score_threshold: f64,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut state = self.new_search_state();
        self.search_threshold_with_state(query, score_threshold, &mut state)
    }

    /// Thresholded direct search using a caller-provided [`SearchState`] to
    /// avoid per-query allocation.
    pub fn search_threshold_with_state<S>(
        &self,
        query: &S,
        score_threshold: f64,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut results = Vec::new();
        self.for_each_threshold_with_state(query, score_threshold, state, |result| {
            results.push(result);
        })?;
        Ok(results)
    }

    /// Thresholded direct search that emits each result to `emit`.
    ///
    /// This is the lowest-allocation API for graph construction: callers can
    /// reuse `state` across queries and write accepted edges directly into
    /// their graph builder.
    pub fn for_each_threshold_with_state<S, Emit>(
        &self,
        query: &S,
        score_threshold: f64,
        state: &mut SearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        ensure_finite(score_threshold, "score_threshold")?;
        if score_threshold <= 0.0 {
            let (query_mz, query_data) = self.prepare_query(query)?;
            let query_precursor_mz = self.inner.query_precursor_mz_for_filter(query)?;
            let query_meta = CosineKernel::spectrum_meta(&query_data);
            self.inner.for_each_direct_with_state(
                &query_mz,
                &query_data,
                &query_meta,
                query_precursor_mz,
                state,
                emit,
            );
            return Ok(());
        }
        if score_threshold > 1.0 {
            let _ = self.prepare_query(query)?;
            let _ = self.inner.query_precursor_mz_for_filter(query)?;
            return Ok(());
        }

        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_precursor_mz = self.inner.query_precursor_mz_for_filter(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        let query_norm = query_meta.0;

        self.inner.for_each_direct_threshold_with_state(
            DirectThresholdSearch {
                query_mz: &query_mz,
                query_data: &query_data,
                query_meta: &query_meta,
                score_threshold,
                query_precursor_mz,
            },
            state,
            emit,
            |lib_meta| score_threshold * query_norm * lib_meta.0,
            |lib_meta| lib_meta.0,
        );
        Ok(())
    }

    /// Modified (direct + shifted) search against the library.
    ///
    /// Phase 1: direct m/z matches. Phase 2: neutral-loss (shifted) matches
    /// with anti-double-counting. This defines a new heuristic, not an
    /// emulation of existing modified-cosine variants.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityComputationError`] if the query violates the
    /// well-separated precondition or contains non-representable values.
    pub fn search_modified<S>(
        &self,
        query: &S,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        let precursor_f64 = ensure_finite(query.precursor_mz().to_f64(), "query_precursor_mz")?;
        Ok(self
            .inner
            .search_modified(&query_mz, &query_data, &query_meta, precursor_f64))
    }

    /// Modified search using a caller-provided [`SearchState`] to avoid
    /// per-query allocation. Create one via [`Self::new_search_state`].
    pub fn search_modified_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        let precursor_f64 = ensure_finite(query.precursor_mz().to_f64(), "query_precursor_mz")?;
        Ok(self.inner.search_modified_with_state(
            &query_mz,
            &query_data,
            &query_meta,
            precursor_f64,
            state,
        ))
    }

    /// Prepare query peaks: compute products, collect m/z, validate.
    fn prepare_query<S>(
        &self,
        query: &S,
    ) -> Result<(Vec<f64>, Vec<f64>), SimilarityComputationError>
    where
        S: Spectrum,
    {
        prepare_cosine_query(
            query,
            self.mz_power,
            self.intensity_power,
            self.inner.tolerance,
        )
    }
}

// ---------------------------------------------------------------------------
// FlashCosineThresholdIndex
// ---------------------------------------------------------------------------

/// Threshold-specialized Flash index for direct cosine graph construction.
///
/// Unlike [`FlashCosineIndex::search_threshold`], this index fixes the score
/// threshold at construction time and builds block-level upper bounds for that
/// threshold. High-threshold searches prune whole spectrum blocks before
/// exact-scoring candidates from the surviving block-local product postings.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct FlashCosineThresholdIndex<P: SpectrumFloat = f64> {
    inner: FlashIndex<CosineKernel, P>,
    mz_power: f64,
    intensity_power: f64,
    score_threshold: f64,
    block_upper_bounds: SpectrumBlockUpperBoundIndex,
    block_products: SpectrumBlockProductIndex<P>,
}

impl<P: SpectrumFloat + Sync> FlashCosineThresholdIndex<P> {
    /// Build a threshold-specialized cosine index from an iterator of spectra.
    ///
    /// The threshold is part of the index. Build one index per cutoff, or one
    /// index at the lowest cutoff you need if you can accept less pruning for
    /// higher cutoffs.
    /// Spectra are reordered internally by a deterministic peak-signature
    /// heuristic; returned spectrum ids and indexed query ids remain in the
    /// original insertion order.
    pub fn new<'a, S>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        score_threshold: f64,
        spectra: impl IntoIterator<Item = &'a S>,
    ) -> Result<Self, FlashCosineIndexError>
    where
        S: Spectrum + 'a,
    {
        let progress = NoopFlashIndexBuildProgress;
        Self::new_with_progress(
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
            spectra,
            &progress,
        )
    }

    /// Build a threshold-specialized cosine index while reporting construction
    /// progress.
    ///
    /// With the `indicatif` feature enabled, pass an `indicatif::ProgressBar`
    /// to display the current construction phase.
    pub fn new_with_progress<'a, S, G>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        score_threshold: f64,
        spectra: impl IntoIterator<Item = &'a S>,
        progress: &G,
    ) -> Result<Self, FlashCosineIndexError>
    where
        S: Spectrum + 'a,
        G: FlashIndexBuildProgress + ?Sized,
    {
        validate_cosine_threshold_index_config(
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
        )?;
        let prepared = prepare_cosine_library::<P, S>(
            mz_power,
            intensity_power,
            mz_tolerance,
            spectra,
            progress,
        )?;
        Self::from_prepared_threshold_library_with_builder(
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
            prepared,
            progress,
            FlashIndex::<CosineKernel, P>::build_with_spectrum_id_map_and_progress,
        )
    }

    fn from_prepared_threshold_library_with_builder<G, BuildInner>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        score_threshold: f64,
        prepared: PreparedFlashSpectra<P>,
        progress: &G,
        build_inner: BuildInner,
    ) -> Result<Self, FlashCosineIndexError>
    where
        G: FlashIndexBuildProgress + ?Sized,
        BuildInner: FnOnce(
            f64,
            PreparedFlashSpectra<P>,
            SpectrumIdMap,
            &G,
        )
            -> Result<FlashIndex<CosineKernel, P>, SimilarityComputationError>,
    {
        progress.start_phase(FlashIndexBuildPhase::ReorderSpectra, Some(1));
        let (prepared, spectrum_id_map) =
            reorder_prepared_spectra_by_signature(prepared, mz_tolerance)
                .map_err(FlashCosineIndexError::Computation)?;
        progress.inc(1);
        progress.start_phase(FlashIndexBuildPhase::BuildBlockUpperBounds, Some(1));
        let block_upper_bounds = build_cosine_block_upper_bounds(&prepared, mz_tolerance)
            .map_err(FlashCosineIndexError::Computation)?;
        progress.inc(1);
        progress.start_phase(FlashIndexBuildPhase::BuildBlockProductIndex, Some(1));
        let block_products =
            SpectrumBlockProductIndex::build(&prepared, DEFAULT_COSINE_SPECTRUM_BLOCK_SIZE)
                .map_err(FlashCosineIndexError::Computation)?;
        progress.inc(1);
        let inner = build_inner(mz_tolerance, prepared, spectrum_id_map, progress)
            .map_err(FlashCosineIndexError::Computation)?;
        progress.finish();

        Ok(Self {
            inner,
            mz_power,
            intensity_power,
            score_threshold,
            block_upper_bounds,
            block_products,
        })
    }

    /// Build a threshold-specialized cosine index with Rayon-backed library
    /// preparation and internal index sorting.
    /// Uses the same internal spectrum reordering as [`Self::new`].
    ///
    /// This constructor is available with the `rayon` feature. It accepts any
    /// Rayon-compatible collection yielding borrowed spectra, such as `&[S]`
    /// or `&Vec<S>`.
    #[cfg(feature = "rayon")]
    pub fn new_parallel<'a, S, I>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        score_threshold: f64,
        spectra: I,
    ) -> Result<Self, FlashCosineIndexError>
    where
        P: Send,
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
    {
        let progress = NoopFlashIndexBuildProgress;
        Self::new_parallel_with_progress(
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
            spectra,
            &progress,
        )
    }

    /// Build a threshold-specialized cosine index with Rayon-backed
    /// preparation and progress reporting.
    #[cfg(feature = "rayon")]
    pub fn new_parallel_with_progress<'a, S, I, G>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        score_threshold: f64,
        spectra: I,
        progress: &G,
    ) -> Result<Self, FlashCosineIndexError>
    where
        P: Send,
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
        G: FlashIndexBuildProgress + Sync + ?Sized,
    {
        validate_cosine_threshold_index_config(
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
        )?;
        let prepared = prepare_cosine_library_parallel::<P, S, I>(
            mz_power,
            intensity_power,
            mz_tolerance,
            spectra,
            progress,
        )?;
        Self::from_prepared_threshold_library_with_builder(
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
            prepared,
            progress,
            FlashIndex::<CosineKernel, P>::build_parallel_with_spectrum_id_map_and_progress,
        )
    }

    /// Returns the threshold this index was built for.
    #[inline]
    pub fn score_threshold(&self) -> f64 {
        self.score_threshold
    }

    /// Returns the m/z power used for scoring.
    #[inline]
    pub fn mz_power(&self) -> f64 {
        self.mz_power
    }

    /// Returns the intensity power used for scoring.
    #[inline]
    pub fn intensity_power(&self) -> f64 {
        self.intensity_power
    }

    /// Returns the m/z tolerance used for matching.
    #[inline]
    pub fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    /// Returns the number of library spectra in the index.
    #[inline]
    pub fn n_spectra(&self) -> u32 {
        self.inner.n_spectra
    }

    /// Create a [`SearchState`] suitable for reuse across queries.
    pub fn new_search_state(&self) -> SearchState {
        self.inner.new_search_state()
    }

    /// Search an external query and return all results above the fixed
    /// threshold.
    pub fn search<S>(&self, query: &S) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut state = self.new_search_state();
        self.search_with_state(query, &mut state)
    }

    /// Search an external query with caller-provided scratch state.
    pub fn search_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut results = Vec::new();
        self.for_each_with_state(query, state, |result| results.push(result))?;
        Ok(results)
    }

    /// Search an external query and return the best `k` thresholded results.
    ///
    /// This is exact and uses the fixed index threshold until the bounded
    /// result set fills, then prunes with the current kth score.
    pub fn search_top_k<S>(
        &self,
        query: &S,
        k: usize,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut state = self.new_search_state();
        self.search_top_k_with_state(query, k, &mut state)
    }

    /// Search an external query and return the best `k` thresholded results
    /// using caller-provided scratch state.
    pub fn search_top_k_with_state<S>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let mut top_k_state = TopKSearchState::new();
        let mut results = Vec::new();
        self.for_each_top_k_with_state(query, k, state, &mut top_k_state, |result| {
            results.push(result);
        })?;
        Ok(results)
    }

    /// Stream the best `k` thresholded results for an external query using
    /// caller-provided scratch state.
    ///
    /// This is the low-allocation variant of [`Self::search_top_k`].
    pub fn for_each_top_k_with_state<S, Emit>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        let (query_mz, query_data) = prepare_cosine_query(
            query,
            self.mz_power,
            self.intensity_power,
            self.inner.tolerance,
        )?;
        let query_precursor_mz = self.inner.query_precursor_mz_for_filter(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        state.reset_diagnostics();
        self.for_each_top_k_prepared_with_state(
            DirectThresholdSearch {
                query_mz: &query_mz,
                query_data: &query_data,
                query_meta: &query_meta,
                score_threshold: self.score_threshold,
                query_precursor_mz,
            },
            k,
            state,
            top_k_state,
            emit,
        );
        Ok(())
    }

    /// Stream thresholded results for an external query.
    pub fn for_each_with_state<S, Emit>(
        &self,
        query: &S,
        state: &mut SearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        let (query_mz, query_data) = prepare_cosine_query(
            query,
            self.mz_power,
            self.intensity_power,
            self.inner.tolerance,
        )?;
        let query_precursor_mz = self.inner.query_precursor_mz_for_filter(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        state.reset_diagnostics();
        self.for_each_prepared_with_state(
            &query_mz,
            &query_data,
            &query_meta,
            query_precursor_mz,
            state,
            emit,
        );
        Ok(())
    }

    /// Stream thresholded results for a query spectrum that is already in this
    /// index.
    ///
    /// This avoids per-query normalization, which is the intended path for
    /// all-pairs graph construction over the indexed library.
    pub fn for_each_indexed_with_state<Emit>(
        &self,
        query_id: u32,
        state: &mut SearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        Emit: FnMut(FlashSearchResult),
    {
        let internal_query_id = self.inner.internal_spectrum_id(query_id)?;
        let (query_mz, query_data) = self.inner.spectrum_slices(internal_query_id);
        let query_meta = *self.inner.spectrum_meta(internal_query_id);
        let query_precursor_mz = Some(self.inner.spectrum_precursor_mz(internal_query_id));

        state.reset_diagnostics();
        self.for_each_prepared_with_state(
            query_mz,
            query_data,
            &query_meta,
            query_precursor_mz,
            state,
            emit,
        );
        Ok(())
    }

    /// Return the best `k` thresholded results for a query spectrum that is
    /// already in this index.
    ///
    /// This is exact and uses adaptive kth-score pruning on top of the fixed
    /// index threshold.
    ///
    /// # Example
    ///
    /// ```
    /// use mass_spectrometry::prelude::*;
    ///
    /// let mut left: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
    /// left.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();
    /// let mut right: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
    /// right.add_peaks([(100.05, 10.0), (200.05, 20.0)]).unwrap();
    ///
    /// let spectra = vec![left, right];
    /// let index = FlashCosineThresholdIndex::<f64>::new(0.0, 1.0, 0.1, 0.8, &spectra).unwrap();
    /// let hits = index.search_top_k_indexed(0, 2).unwrap();
    ///
    /// assert_eq!(hits[0].spectrum_id, 0);
    /// assert!(hits.iter().any(|hit| hit.spectrum_id == 1));
    /// ```
    pub fn search_top_k_indexed(
        &self,
        query_id: u32,
        k: usize,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError> {
        let mut state = self.new_search_state();
        self.search_top_k_indexed_with_state(query_id, k, &mut state)
    }

    /// Return the best `k` thresholded results for an indexed query using
    /// caller-provided scratch state.
    pub fn search_top_k_indexed_with_state(
        &self,
        query_id: u32,
        k: usize,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError> {
        let mut top_k_state = TopKSearchState::new();
        let mut results = Vec::new();
        self.for_each_top_k_indexed_with_state(query_id, k, state, &mut top_k_state, |result| {
            results.push(result)
        })?;
        Ok(results)
    }

    /// Stream the best `k` thresholded results for an indexed query using
    /// caller-provided scratch state.
    ///
    /// This is the intended low-allocation indexed-query top-k API.
    pub fn for_each_top_k_indexed_with_state<Emit>(
        &self,
        query_id: u32,
        k: usize,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        Emit: FnMut(FlashSearchResult),
    {
        let internal_query_id = self.inner.internal_spectrum_id(query_id)?;
        let (query_mz, query_data) = self.inner.spectrum_slices(internal_query_id);
        let query_meta = *self.inner.spectrum_meta(internal_query_id);
        let query_precursor_mz = Some(self.inner.spectrum_precursor_mz(internal_query_id));

        state.reset_diagnostics();
        self.for_each_top_k_prepared_with_state(
            DirectThresholdSearch {
                query_mz,
                query_data,
                query_meta: &query_meta,
                score_threshold: self.score_threshold,
                query_precursor_mz,
            },
            k,
            state,
            top_k_state,
            emit,
        );
        Ok(())
    }

    fn prepare_block_pruned_candidate_blocks<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &CosineNorm,
        minimum_score: f64,
        state: &mut SearchState,
    ) {
        self.block_upper_bounds.prepare_allowed_blocks(
            query_mz,
            self.inner.tolerance,
            minimum_score,
            state,
            |query_index, block_max_weight| {
                if query_meta.0 == 0.0 {
                    return 0.0;
                }
                query_data[query_index].to_f64().abs() / query_meta.0 * block_max_weight
            },
        );
    }

    fn for_each_prepared_with_state<Q: SpectrumFloat, Emit>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &CosineNorm,
        query_precursor_mz: Option<f64>,
        state: &mut SearchState,
        mut emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        if self.score_threshold <= 0.0 {
            self.inner.for_each_direct_with_state(
                query_mz,
                query_data,
                query_meta,
                query_precursor_mz,
                state,
                emit,
            );
            return;
        }
        if self.score_threshold > 1.0 || self.inner.n_spectra == 0 || query_mz.is_empty() {
            return;
        }
        if self.score_threshold < THRESHOLD_INDEX_ONE_SIDED_MIN_THRESHOLD {
            self.for_each_direct_threshold_prepared(
                query_mz,
                query_data,
                query_meta,
                query_precursor_mz,
                state,
                emit,
            );
            return;
        }

        self.prepare_block_pruned_candidate_blocks(
            query_mz,
            query_data,
            query_meta,
            self.score_threshold,
            state,
        );
        self.emit_allowed_block_scores(
            query_mz,
            query_data,
            query_meta,
            query_precursor_mz,
            state,
            &mut emit,
        );
    }

    fn for_each_top_k_prepared_with_state<Q: SpectrumFloat, Emit>(
        &self,
        search: DirectThresholdSearch<'_, CosineKernel, Q>,
        k: usize,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        if self.score_threshold <= 0.0
            || self.score_threshold < THRESHOLD_INDEX_ONE_SIDED_MIN_THRESHOLD
        {
            for_each_cosine_top_k_prepared(&self.inner, search, k, state, top_k_state, emit);
            return;
        }

        let mut top_k = TopKSearchResults::new(k, self.score_threshold, top_k_state);
        if k == 0
            || self.score_threshold > 1.0
            || self.inner.n_spectra == 0
            || search.query_mz.is_empty()
        {
            top_k.emit(emit);
            return;
        }

        self.prepare_block_pruned_candidate_blocks(
            search.query_mz,
            search.query_data,
            search.query_meta,
            top_k.pruning_score(),
            state,
        );
        self.push_allowed_block_top_k_candidates(search, state, &mut top_k);
        state.add_results_emitted(top_k.len());
        top_k.emit(emit);
    }

    fn emit_allowed_block_scores<Q: SpectrumFloat, Emit>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &CosineNorm,
        query_precursor_mz: Option<f64>,
        state: &mut SearchState,
        emit: &mut Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        let mut results_emitted = 0usize;
        self.inner.for_each_allowed_block_raw_score(
            query_mz,
            query_data,
            query_precursor_mz,
            &self.block_products,
            state,
            |spec_id, raw, count| {
                let lib_meta = self.inner.spectrum_meta(spec_id);
                let score = CosineKernel::finalize(raw, count, query_meta, lib_meta);
                if score > 0.0 && score >= self.score_threshold {
                    results_emitted = results_emitted.saturating_add(1);
                    emit(FlashSearchResult {
                        spectrum_id: self.inner.public_spectrum_id(spec_id),
                        score,
                        n_matches: count,
                    });
                }
            },
        );
        state.add_results_emitted(results_emitted);
    }

    fn push_allowed_block_top_k_candidates<Q: SpectrumFloat>(
        &self,
        search: DirectThresholdSearch<'_, CosineKernel, Q>,
        state: &mut SearchState,
        top_k: &mut TopKSearchResults<'_>,
    ) {
        let query_norm = search.query_meta.0;
        if query_norm == 0.0 {
            state.reset_allowed_spectrum_blocks();
            return;
        }

        state.prepare_threshold_order(search.query_data);
        self.inner.score_allowed_block_candidates_by_query_order(
            search.query_mz,
            search.query_precursor_mz,
            &self.block_products,
            state,
            top_k.pruning_score() * query_norm,
            |spec_id| {
                let lib_meta = self.inner.spectrum_meta(spec_id);
                if lib_meta.0 == 0.0 {
                    return top_k.pruning_score() * query_norm;
                }

                let target_raw = top_k.pruning_score() * query_norm * lib_meta.0;
                let (raw, count) = self.inner.direct_score_for_spectrum(
                    search.query_mz,
                    search.query_data,
                    spec_id,
                );
                if raw < target_raw {
                    return top_k.pruning_score() * query_norm;
                }

                let score = CosineKernel::finalize(raw, count, search.query_meta, lib_meta);
                if score > 0.0 && score >= self.score_threshold {
                    top_k.push(FlashSearchResult {
                        spectrum_id: self.inner.public_spectrum_id(spec_id),
                        score,
                        n_matches: count,
                    });
                }
                top_k.pruning_score() * query_norm
            },
        );
    }

    fn for_each_direct_threshold_prepared<Q: SpectrumFloat, Emit>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &CosineNorm,
        query_precursor_mz: Option<f64>,
        state: &mut SearchState,
        emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        let query_norm = query_meta.0;
        self.inner.for_each_direct_threshold_with_state(
            DirectThresholdSearch {
                query_mz,
                query_data,
                query_meta,
                score_threshold: self.score_threshold,
                query_precursor_mz,
            },
            state,
            emit,
            |lib_meta| self.score_threshold * query_norm * lib_meta.0,
            |lib_meta| lib_meta.0,
        );
    }
}

// ---------------------------------------------------------------------------
// FlashCosineSelfSimilarityIndex
// ---------------------------------------------------------------------------

/// One-shot cosine index for exact thresholded top-k self-similarity.
///
/// This index is specialized for the case where every query is already part of
/// the indexed library, the score threshold and `k` are fixed at construction
/// time, and a precursor-mass filter is always active. It does not implement
/// [`SpectraIndex`] because it intentionally has no external-query API.
///
/// # Example
///
/// ```
/// use mass_spectrometry::prelude::*;
/// use rayon::prelude::*;
///
/// let mut left: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
/// left.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();
/// let mut right: GenericSpectrum = GenericSpectrum::try_with_capacity(500.2, 2).unwrap();
/// right.add_peaks([(100.05, 10.0), (200.05, 20.0)]).unwrap();
///
/// let spectra = vec![left, right];
/// let index = FlashCosineSelfSimilarityIndex::<f64>::with_pepmass_tolerance(
///     0.0, 1.0, 0.1, 0.8, 1, 1.0, &spectra,
/// )
/// .unwrap();
///
/// let mut rows: Vec<_> = index.par_top_k_rows().map(Result::unwrap).collect();
/// rows.sort_by_key(|row| row.0);
///
/// assert_eq!(rows[0].0, 0);
/// assert_eq!(rows[0].1[0].spectrum_id, 1);
/// assert!(rows[0].1[0].score >= 0.8);
/// ```
#[cfg(feature = "rayon")]
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct FlashCosineSelfSimilarityIndex<P: SpectrumFloat = f64> {
    inner: FlashIndex<CosineKernel, P>,
    mz_power: f64,
    intensity_power: f64,
    score_threshold: f64,
    top_k: usize,
    block_upper_bounds: SpectrumBlockUpperBoundIndex,
    block_products: SpectrumBlockProductIndex<P>,
}

#[cfg(feature = "rayon")]
impl<P: SpectrumFloat + Send + Sync> FlashCosineSelfSimilarityIndex<P> {
    /// Build a self-similarity index with a required precursor-mass filter.
    ///
    /// The input collection must support Rayon borrowed iteration, such as
    /// `&[S]` or `&Vec<S>`. Returned row ids and hit ids use the original
    /// public insertion order even though spectra are reordered internally.
    pub fn new<'a, S, I>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        score_threshold: f64,
        top_k: usize,
        pepmass_filter: PepmassFilter,
        spectra: I,
    ) -> Result<Self, FlashCosineIndexError>
    where
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
    {
        let profile = CosineSelfSimilarityProfile {
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
            top_k,
            pepmass_filter,
        };
        let progress = NoopFlashIndexBuildProgress;
        Self::new_with_profile(profile, spectra, &progress)
    }

    /// Build a self-similarity index with an absolute precursor m/z tolerance.
    pub fn with_pepmass_tolerance<'a, S, I>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        score_threshold: f64,
        top_k: usize,
        pepmass_tolerance: f64,
        spectra: I,
    ) -> Result<Self, FlashCosineIndexError>
    where
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
    {
        Self::new(
            mz_power,
            intensity_power,
            mz_tolerance,
            score_threshold,
            top_k,
            PepmassFilter::within_tolerance(pepmass_tolerance)
                .map_err(FlashCosineIndexError::Config)?,
            spectra,
        )
    }

    fn new_with_profile<'a, S, I, G>(
        profile: CosineSelfSimilarityProfile,
        spectra: I,
        progress: &G,
    ) -> Result<Self, FlashCosineIndexError>
    where
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
        G: FlashIndexBuildProgress + Sync + ?Sized,
    {
        validate_cosine_self_similarity_index_config(profile)?;
        let prepared = prepare_cosine_library_parallel::<P, S, I>(
            profile.mz_power,
            profile.intensity_power,
            profile.mz_tolerance,
            spectra,
            progress,
        )?;
        Self::from_prepared_self_similarity_library(profile, prepared, progress)
    }

    fn from_prepared_self_similarity_library<G>(
        profile: CosineSelfSimilarityProfile,
        prepared: PreparedFlashSpectra<P>,
        progress: &G,
    ) -> Result<Self, FlashCosineIndexError>
    where
        G: FlashIndexBuildProgress + Sync + ?Sized,
    {
        progress.start_phase(FlashIndexBuildPhase::ReorderSpectra, Some(1));
        let (prepared, spectrum_id_map) =
            reorder_prepared_spectra_by_signature(prepared, profile.mz_tolerance)
                .map_err(FlashCosineIndexError::Computation)?;
        progress.inc(1);

        let block_size =
            self_similarity_block_size(prepared.len(), profile.top_k, profile.score_threshold);
        progress.start_phase(FlashIndexBuildPhase::BuildBlockUpperBounds, Some(1));
        let block_upper_bounds = build_cosine_block_upper_bounds_with_block_size(
            &prepared,
            profile.mz_tolerance,
            block_size,
        )
        .map_err(FlashCosineIndexError::Computation)?;
        progress.inc(1);
        progress.start_phase(FlashIndexBuildPhase::BuildBlockProductIndex, Some(1));
        let block_products = SpectrumBlockProductIndex::build(&prepared, block_size)
            .map_err(FlashCosineIndexError::Computation)?;
        progress.inc(1);

        let mut inner =
            FlashIndex::<CosineKernel, P>::build_parallel_with_spectrum_id_map_and_progress(
                profile.mz_tolerance,
                prepared,
                spectrum_id_map,
                progress,
            )
            .map_err(FlashCosineIndexError::Computation)?;
        inner
            .set_pepmass_filter_with_progress(profile.pepmass_filter, progress)
            .map_err(FlashCosineIndexError::Computation)?;

        Ok(Self {
            inner,
            mz_power: profile.mz_power,
            intensity_power: profile.intensity_power,
            score_threshold: profile.score_threshold,
            top_k: profile.top_k,
            block_upper_bounds,
            block_products,
        })
    }

    /// Returns the fixed score threshold.
    #[inline]
    pub fn score_threshold(&self) -> f64 {
        self.score_threshold
    }

    /// Returns the fixed number of neighbors retained per query row.
    #[inline]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Returns the m/z power used for scoring.
    #[inline]
    pub fn mz_power(&self) -> f64 {
        self.mz_power
    }

    /// Returns the intensity power used for scoring.
    #[inline]
    pub fn intensity_power(&self) -> f64 {
        self.intensity_power
    }

    /// Returns the m/z tolerance used for peak matching.
    #[inline]
    pub fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    /// Returns the number of spectra in the self-similarity matrix.
    #[inline]
    pub fn n_spectra(&self) -> u32 {
        self.inner.n_spectra
    }

    /// Returns the required precursor-mass filter.
    #[inline]
    pub fn pepmass_filter(&self) -> PepmassFilter {
        self.inner.pepmass_filter()
    }

    /// Return a Rayon iterator over every directed self-similarity row.
    ///
    /// Each item contains the public query spectrum id and its non-self top-k
    /// hits above the fixed threshold.
    pub fn par_top_k_rows(
        &self,
    ) -> impl ParallelIterator<
        Item = Result<(u32, Vec<FlashSearchResult>), SimilarityComputationError>,
    > + '_ {
        self.par_top_k_rows_in(0..self.n_spectra())
    }

    /// Return a Rayon iterator over a contiguous range of query rows.
    ///
    /// This is useful for chunked graph construction and large benchmarks. It
    /// uses the same exact scoring path as [`Self::par_top_k_rows`].
    pub fn par_top_k_rows_in(
        &self,
        query_ids: Range<u32>,
    ) -> impl ParallelIterator<
        Item = Result<(u32, Vec<FlashSearchResult>), SimilarityComputationError>,
    > + '_ {
        query_ids.into_par_iter().map_init(
            || (self.inner.new_search_state(), TopKSearchState::new()),
            |(state, top_k_state), query_id| {
                self.top_k_row_with_state(query_id, state, top_k_state)
            },
        )
    }

    /// Return a Rayon iterator over the provided public query row ids.
    ///
    /// This keeps per-worker scratch state internal while letting callers run
    /// reproducible shards that are not contiguous in public id order.
    pub fn par_top_k_rows_for<'a>(
        &'a self,
        query_ids: &'a [u32],
    ) -> impl ParallelIterator<
        Item = Result<(u32, Vec<FlashSearchResult>), SimilarityComputationError>,
    > + 'a {
        query_ids.par_iter().copied().map_init(
            || (self.inner.new_search_state(), TopKSearchState::new()),
            |(state, top_k_state), query_id| {
                self.top_k_row_with_state(query_id, state, top_k_state)
            },
        )
    }

    fn top_k_row_with_state(
        &self,
        query_id: u32,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
    ) -> Result<(u32, Vec<FlashSearchResult>), SimilarityComputationError> {
        let internal_query_id = self.inner.internal_spectrum_id(query_id)?;
        let (query_mz, query_data) = self.inner.spectrum_slices(internal_query_id);
        let query_meta = *self.inner.spectrum_meta(internal_query_id);
        let query_precursor_mz = Some(self.inner.spectrum_precursor_mz(internal_query_id));
        let mut results = Vec::with_capacity(self.top_k);

        self.for_each_top_k_prepared_excluding_self(
            internal_query_id,
            DirectThresholdSearch {
                query_mz,
                query_data,
                query_meta: &query_meta,
                score_threshold: self.score_threshold,
                query_precursor_mz,
            },
            state,
            top_k_state,
            |result| results.push(result),
        );

        Ok((query_id, results))
    }

    fn prepare_block_pruned_candidate_blocks<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &CosineNorm,
        minimum_score: f64,
        state: &mut SearchState,
    ) {
        self.block_upper_bounds.prepare_allowed_blocks(
            query_mz,
            self.inner.tolerance,
            minimum_score,
            state,
            |query_index, block_max_weight| {
                if query_meta.0 == 0.0 {
                    return 0.0;
                }
                query_data[query_index].to_f64().abs() / query_meta.0 * block_max_weight
            },
        );
    }

    fn for_each_top_k_prepared_excluding_self<Q: SpectrumFloat, Emit>(
        &self,
        internal_query_id: u32,
        search: DirectThresholdSearch<'_, CosineKernel, Q>,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        state.reset_diagnostics();
        let mut top_k = TopKSearchResults::new(self.top_k, self.score_threshold, top_k_state);
        if self.score_threshold > 1.0 || self.inner.n_spectra == 0 || search.query_mz.is_empty() {
            top_k.emit(emit);
            return;
        }

        let query_norm = search.query_meta.0;
        if query_norm == 0.0 {
            top_k.emit(emit);
            return;
        }

        if self.score_threshold < THRESHOLD_INDEX_ONE_SIDED_MIN_THRESHOLD {
            self.push_global_top_k_candidates_excluding_self(
                internal_query_id,
                search,
                state,
                &mut top_k,
            );
            state.add_results_emitted(top_k.len());
            top_k.emit(emit);
            return;
        }

        self.prepare_block_pruned_candidate_blocks(
            search.query_mz,
            search.query_data,
            search.query_meta,
            top_k.pruning_score(),
            state,
        );
        self.push_allowed_block_top_k_candidates_excluding_self(
            internal_query_id,
            search,
            state,
            &mut top_k,
        );
        state.add_results_emitted(top_k.len());
        top_k.emit(emit);
    }

    fn push_global_top_k_candidates_excluding_self<Q: SpectrumFloat>(
        &self,
        internal_query_id: u32,
        search: DirectThresholdSearch<'_, CosineKernel, Q>,
        state: &mut SearchState,
        top_k: &mut TopKSearchResults<'_>,
    ) {
        let query_norm = search.query_meta.0;
        state.prepare_threshold_order(search.query_data);
        state.ensure_candidate_capacity(self.inner.n_spectra as usize);

        let query_order_len = state.query_order().len();
        for order_position in 0..query_order_len {
            let query_index = state.query_order()[order_position];
            let qmz = search.query_mz[query_index];

            let visited = self.inner.for_each_product_spectrum_in_window(
                qmz,
                search.query_precursor_mz,
                |spec_id| {
                    if spec_id == internal_query_id || state.is_candidate(spec_id) {
                        return;
                    }
                    state.mark_candidate(spec_id);
                    self.score_top_k_candidate(&search, spec_id, top_k);
                },
            );
            state.add_product_postings_visited(visited);

            if state.query_suffix_bound_at(order_position + 1) < top_k.pruning_score() * query_norm
            {
                break;
            }
        }

        state.reset_candidates();
    }

    fn push_allowed_block_top_k_candidates_excluding_self<Q: SpectrumFloat>(
        &self,
        internal_query_id: u32,
        search: DirectThresholdSearch<'_, CosineKernel, Q>,
        state: &mut SearchState,
        top_k: &mut TopKSearchResults<'_>,
    ) {
        let query_norm = search.query_meta.0;
        self.inner.score_allowed_block_candidates_by_query_order(
            search.query_mz,
            search.query_precursor_mz,
            &self.block_products,
            state,
            top_k.pruning_score() * query_norm,
            |spec_id| {
                if spec_id != internal_query_id {
                    self.score_top_k_candidate(&search, spec_id, top_k);
                }
                top_k.pruning_score() * query_norm
            },
        );
    }

    fn score_top_k_candidate<Q: SpectrumFloat>(
        &self,
        search: &DirectThresholdSearch<'_, CosineKernel, Q>,
        spec_id: u32,
        top_k: &mut TopKSearchResults<'_>,
    ) {
        let query_norm = search.query_meta.0;
        let lib_meta = self.inner.spectrum_meta(spec_id);
        if lib_meta.0 == 0.0 {
            return;
        }

        let target_raw = top_k.pruning_score() * query_norm * lib_meta.0;
        let (raw, count) =
            self.inner
                .direct_score_for_spectrum(search.query_mz, search.query_data, spec_id);
        if raw < target_raw {
            return;
        }

        let score = CosineKernel::finalize(raw, count, search.query_meta, lib_meta);
        if score > 0.0 {
            top_k.push(FlashSearchResult {
                spectrum_id: self.inner.public_spectrum_id(spec_id),
                score,
                n_matches: count,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// SpectraIndex implementations
// ---------------------------------------------------------------------------

impl<P: SpectrumFloat + Sync> SpectraIndex for FlashCosineIndex<P> {
    fn n_spectra(&self) -> u32 {
        self.n_spectra()
    }

    fn tolerance(&self) -> f64 {
        self.tolerance()
    }

    fn new_search_state(&self) -> SearchState {
        self.new_search_state()
    }

    fn pepmass_filter(&self) -> PepmassFilter {
        self.inner.pepmass_filter()
    }

    fn with_pepmass_filter_and_progress<G>(
        mut self,
        filter: PepmassFilter,
        progress: &G,
    ) -> Result<Self, SpectraIndexSetupError>
    where
        G: FlashIndexBuildProgress + ?Sized,
    {
        self.inner
            .set_pepmass_filter_with_progress(filter, progress)?;
        Ok(self)
    }

    fn without_pepmass_filter(mut self) -> Self {
        self.inner.clear_pepmass_filter();
        self
    }

    fn search<S>(&self, query: &S) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search(query)
    }

    fn search_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search_with_state(query, state)
    }

    fn search_top_k<S>(
        &self,
        query: &S,
        k: usize,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search_top_k(query, k)
    }

    fn search_top_k_with_state<S>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search_top_k_with_state(query, k, state)
    }

    fn for_each_top_k_with_state<S, Emit>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        self.for_each_top_k_with_state(query, k, state, top_k_state, emit)
    }
}

impl<P: SpectrumFloat + Sync> SpectraIndex for FlashCosineThresholdIndex<P> {
    fn n_spectra(&self) -> u32 {
        self.n_spectra()
    }

    fn tolerance(&self) -> f64 {
        self.tolerance()
    }

    fn new_search_state(&self) -> SearchState {
        self.new_search_state()
    }

    fn pepmass_filter(&self) -> PepmassFilter {
        self.inner.pepmass_filter()
    }

    fn with_pepmass_filter_and_progress<G>(
        mut self,
        filter: PepmassFilter,
        progress: &G,
    ) -> Result<Self, SpectraIndexSetupError>
    where
        G: FlashIndexBuildProgress + ?Sized,
    {
        self.inner
            .set_pepmass_filter_with_progress(filter, progress)?;
        Ok(self)
    }

    fn without_pepmass_filter(mut self) -> Self {
        self.inner.clear_pepmass_filter();
        self
    }

    fn search<S>(&self, query: &S) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search(query)
    }

    fn search_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search_with_state(query, state)
    }

    fn search_top_k<S>(
        &self,
        query: &S,
        k: usize,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search_top_k(query, k)
    }

    fn search_top_k_with_state<S>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        self.search_top_k_with_state(query, k, state)
    }

    fn for_each_top_k_with_state<S, Emit>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
        top_k_state: &mut TopKSearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        self.for_each_top_k_with_state(query, k, state, top_k_state, emit)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by [`FlashCosineIndex`] construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FlashCosineIndexError {
    /// Invalid configuration parameter.
    #[error(transparent)]
    Config(SimilarityConfigError),
    /// Error during spectrum processing.
    #[error(transparent)]
    Computation(SimilarityComputationError),
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use half::f16;

    use super::*;
    use crate::structs::GenericSpectrum;
    use crate::traits::{Spectrum, SpectrumMut};

    #[derive(Clone)]
    struct RawSpectrum {
        precursor_mz: f64,
        peaks: Vec<(f64, f64)>,
    }

    impl Spectrum for RawSpectrum {
        type Precision = f64;

        type SortedIntensitiesIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedMzIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedPeaksIter<'a>
            = core::iter::Copied<core::slice::Iter<'a, (f64, f64)>>
        where
            Self: 'a;

        fn len(&self) -> usize {
            self.peaks.len()
        }

        fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
            self.peaks.iter().map(|peak| peak.1)
        }

        fn intensity_nth(&self, n: usize) -> f64 {
            self.peaks[n].1
        }

        fn mz(&self) -> Self::SortedMzIter<'_> {
            self.peaks.iter().map(|peak| peak.0)
        }

        fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
            self.peaks[index..].iter().map(|peak| peak.0)
        }

        fn mz_nth(&self, n: usize) -> f64 {
            self.peaks[n].0
        }

        fn peaks(&self) -> Self::SortedPeaksIter<'_> {
            self.peaks.iter().copied()
        }

        fn peak_nth(&self, n: usize) -> (f64, f64) {
            self.peaks[n]
        }

        fn precursor_mz(&self) -> f64 {
            self.precursor_mz
        }
    }

    fn make_spectrum(precursor_mz: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
        let mut spectrum = GenericSpectrum::try_with_capacity(precursor_mz, peaks.len())
            .expect("test spectrum allocation should succeed");
        for &(mz, intensity) in peaks {
            spectrum
                .add_peak(mz, intensity)
                .expect("test peaks should be valid and sorted");
        }
        spectrum
    }

    #[test]
    fn cosine_kernel_finalize_handles_zero_denominator_and_clamps() {
        assert_eq!(
            CosineKernel::finalize(1.0, 1, &CosineNorm(0.0), &CosineNorm(2.0)),
            0.0
        );
        assert_eq!(
            CosineKernel::finalize(5.0, 1, &CosineNorm(1.0), &CosineNorm(1.0)),
            1.0
        );
    }

    #[test]
    fn index_accessors_and_search_wrappers_round_trip() {
        let library = [make_spectrum(200.0, &[(100.0, 4.0)])];
        let index = FlashCosineIndex::<f64>::new(1.0, 2.0, 0.1, library.iter())
            .expect("index should build");

        assert_eq!(index.mz_power(), 1.0);
        assert_eq!(index.intensity_power(), 2.0);
        assert_eq!(index.tolerance(), 0.1);
        assert_eq!(index.n_spectra(), 1);

        let direct = index
            .search(&library[0])
            .expect("direct search should work");
        let mut direct_state = index.new_search_state();
        let direct_with_state = index
            .search_with_state(&library[0], &mut direct_state)
            .expect("stateful direct search should work");
        assert_eq!(direct, direct_with_state);

        let shifted_query = make_spectrum(210.0, &[(110.0, 4.0)]);
        let modified = index
            .search_modified(&shifted_query)
            .expect("modified search should work");
        let mut modified_state = index.new_search_state();
        let modified_with_state = index
            .search_modified_with_state(&shifted_query, &mut modified_state)
            .expect("stateful modified search should work");
        assert_eq!(modified, modified_with_state);
        assert_eq!(modified[0].spectrum_id, 0);
    }

    #[test]
    fn cosine_indices_support_lower_storage_precision() {
        let library = [
            make_spectrum(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
            make_spectrum(500.0, &[(100.05, 10.0), (200.05, 20.0)]),
            make_spectrum(500.0, &[(300.0, 10.0), (400.0, 20.0)]),
        ];

        let f32_index = FlashCosineIndex::<f32>::new(0.0, 1.0, 0.1, library.iter())
            .expect("f32 cosine index should build");
        let f32_hits = f32_index
            .search(&library[0])
            .expect("f32 cosine search should work");
        assert!(
            f32_hits
                .iter()
                .any(|hit| hit.spectrum_id == 0 && hit.score > 0.99)
        );

        let f32_threshold =
            FlashCosineThresholdIndex::<f32>::new(0.0, 1.0, 0.1, 0.8, library.iter())
                .expect("f32 threshold index should build");
        let indexed_hits = f32_threshold
            .search_top_k_indexed(0, 2)
            .expect("f32 indexed top-k should work");
        assert!(indexed_hits.iter().any(|hit| hit.spectrum_id == 0));

        let f16_index = FlashCosineIndex::<f16>::new(0.0, 1.0, 0.1, library.iter())
            .expect("f16 cosine index should build for representable peaks");
        let f16_hits = f16_index
            .search(&library[0])
            .expect("f16 cosine search should work");
        assert!(
            f16_hits
                .iter()
                .any(|hit| hit.spectrum_id == 0 && hit.score > 0.99)
        );
    }

    #[test]
    fn cosine_index_validates_spacing_after_storage_precision_conversion() {
        let library = [make_spectrum(200.0, &[(100.0, 1.0), (100.03, 1.0)])];

        FlashCosineIndex::<f32>::new(0.0, 1.0, 0.001, library.iter())
            .expect("f32 keeps these peaks separated");

        let error = match FlashCosineIndex::<f16>::new(0.0, 1.0, 0.001, library.iter()) {
            Ok(_) => panic!("f16 rounds these peaks onto the same stored m/z"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            FlashCosineIndexError::Computation(SimilarityComputationError::InvalidPeakSpacing(
                "library spectrum"
            ))
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_cosine_index_matches_sequential_constructor() {
        let library = [
            make_spectrum(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
            make_spectrum(500.0, &[(100.05, 10.0), (200.05, 20.0)]),
            make_spectrum(500.0, &[(300.0, 10.0), (400.0, 20.0)]),
        ];
        let sequential = FlashCosineIndex::<f64>::new(0.0, 1.0, 0.1, library.iter())
            .expect("sequential index should build");
        let parallel = FlashCosineIndex::<f64>::new_parallel(0.0, 1.0, 0.1, library.as_slice())
            .expect("parallel index should build");

        assert_eq!(sequential.search(&library[0]), parallel.search(&library[0]));
        assert_eq!(
            sequential.search_top_k_threshold(&library[0], 2, 0.8),
            parallel.search_top_k_threshold(&library[0], 2, 0.8)
        );

        let shifted_query = make_spectrum(510.0, &[(110.0, 10.0), (210.0, 20.0)]);
        assert_eq!(
            sequential.search_modified(&shifted_query),
            parallel.search_modified(&shifted_query)
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_threshold_index_matches_sequential_constructor() {
        let library = [
            make_spectrum(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
            make_spectrum(500.0, &[(100.05, 10.0), (200.05, 20.0)]),
            make_spectrum(500.0, &[(300.0, 10.0), (400.0, 20.0)]),
        ];
        let sequential = FlashCosineThresholdIndex::<f64>::new(0.0, 1.0, 0.1, 0.8, library.iter())
            .expect("sequential threshold index should build");
        let parallel =
            FlashCosineThresholdIndex::<f64>::new_parallel(0.0, 1.0, 0.1, 0.8, library.as_slice())
                .expect("parallel threshold index should build");

        assert_eq!(sequential.search(&library[0]), parallel.search(&library[0]));
        assert_eq!(
            sequential.search_top_k_indexed(0, 2),
            parallel.search_top_k_indexed(0, 2)
        );

        let mut sequential_state = sequential.new_search_state();
        let mut parallel_state = parallel.new_search_state();
        let mut sequential_hits = Vec::new();
        let mut parallel_hits = Vec::new();
        sequential
            .for_each_indexed_with_state(0, &mut sequential_state, |hit| {
                sequential_hits.push(hit);
            })
            .expect("sequential indexed search should work");
        parallel
            .for_each_indexed_with_state(0, &mut parallel_state, |hit| {
                parallel_hits.push(hit);
            })
            .expect("parallel indexed search should work");
        assert_eq!(sequential_hits, parallel_hits);
    }

    #[test]
    fn modified_search_rejects_non_finite_query_precursor() {
        let library = [make_spectrum(200.0, &[(100.0, 1.0)])];
        let index = FlashCosineIndex::<f64>::new(1.0, 1.0, 0.1, library.iter())
            .expect("index should build");
        let query = RawSpectrum {
            precursor_mz: f64::NAN,
            peaks: vec![(100.0, 1.0)],
        };

        let error = index
            .search_modified(&query)
            .expect_err("non-finite query precursor should be rejected");
        assert_eq!(
            error,
            SimilarityComputationError::NonFiniteValue("query_precursor_mz")
        );

        let mut state = index.new_search_state();
        let error = index
            .search_modified_with_state(&query, &mut state)
            .expect_err("stateful modified search should reject non-finite precursor");
        assert_eq!(
            error,
            SimilarityComputationError::NonFiniteValue("query_precursor_mz")
        );
    }
}
