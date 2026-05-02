//! Flash inverted m/z index for cosine spectral similarity.
//!
//! Provides O(Q_peaks * log(P_total)) library-scale search instead of
//! O(N * pairwise_cost). Exact equivalence to [`super::LinearCosine`] on
//! well-separated spectra (consecutive peaks > 2 * tolerance apart).

use alloc::vec::Vec;

use super::cosine_common::{
    ensure_finite, normalized_peak_products, validate_non_negative_tolerance,
    validate_well_separated,
};
use super::flash_common::{
    DirectThresholdSearch, FlashIndex, FlashKernel, FlashSearchResult, SearchState,
    ThresholdPrefixPostings, TopKSearchResults, TopKSearchState, l2_threshold_prefix_indices,
};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{SpectraIndex, Spectrum, SpectrumFloat};

const THRESHOLD_INDEX_ONE_SIDED_MIN_THRESHOLD: f64 = 0.85;
const THRESHOLD_INDEX_TWO_SIDED_MIN_THRESHOLD: f64 = 0.85;

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
    fn spectrum_meta(peak_data: &[f64]) -> CosineNorm {
        let sum_sq: f64 = peak_data.iter().map(|&v| v * v).sum();
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

/// Run exact cosine top-k over a prepared query while using the current kth
/// score as an adaptive pruning bound.
///
/// Query peaks are scanned from largest to smallest contribution. Once the
/// remaining query suffix norm cannot reach the current pruning score, no
/// unseen spectrum can enter the result set, so the index scan stops early.
fn for_each_cosine_top_k_prepared<Emit>(
    inner: &FlashIndex<CosineKernel>,
    search: DirectThresholdSearch<'_, CosineKernel>,
    k: usize,
    state: &mut SearchState,
    top_k_state: &mut TopKSearchState,
    emit: Emit,
) where
    Emit: FnMut(FlashSearchResult),
{
    let mut top_k = TopKSearchResults::new(k, search.score_threshold, top_k_state);
    if k == 0 || search.score_threshold > 1.0 || inner.n_spectra == 0 || search.query_mz.is_empty()
    {
        top_k.emit(emit);
        return;
    }

    let query_norm = search.query_meta.0;
    if query_norm == 0.0 {
        top_k.emit(emit);
        return;
    }

    state.prepare_threshold_order(search.query_data);
    state.ensure_candidate_capacity(inner.n_spectra as usize);

    let query_order_len = state.query_order().len();
    for order_position in 0..query_order_len {
        let query_index = state.query_order()[order_position];
        let qmz = search.query_mz[query_index];

        inner.for_each_product_spectrum_in_window(qmz, |spec_id| {
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
                    spectrum_id: spec_id,
                    score,
                    n_matches: count,
                });
            }
        });

        let processed_prefix_len = order_position + 1;
        let pruning_score = top_k.pruning_score();
        if state.query_suffix_bound_at(processed_prefix_len) < pruning_score * query_norm {
            break;
        }
    }

    state.reset_candidates();
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
/// let index = FlashCosineIndex::new(1.0, 1.0, 0.1, library.iter())
///     .expect("index build should succeed");
///
/// let query: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
/// let results = index.search(&query).expect("search should succeed");
/// assert!(results.iter().any(|r| r.spectrum_id == 0 && r.score > 0.99));
/// ```
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct FlashCosineIndex {
    inner: FlashIndex<CosineKernel>,
    mz_power: f64,
    intensity_power: f64,
}

impl FlashCosineIndex {
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
        // Validate config.
        validate_non_negative_tolerance(mz_tolerance).map_err(FlashCosineIndexError::Config)?;
        ensure_finite(mz_power, "mz_power").map_err(FlashCosineIndexError::Computation)?;
        ensure_finite(intensity_power, "intensity_power")
            .map_err(FlashCosineIndexError::Computation)?;

        let tolerance = mz_tolerance;

        // Prepare spectrum data.
        let mut prepared: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();

        for spectrum in spectra {
            let mz_vals: Vec<f64> = spectrum.mz().map(SpectrumFloat::to_f64).collect();
            let data_vals = normalized_peak_products(spectrum, mz_power, intensity_power)
                .map_err(FlashCosineIndexError::Computation)?;

            validate_well_separated(&mz_vals, tolerance, "library spectrum")
                .map_err(FlashCosineIndexError::Computation)?;

            let precursor_f64 = ensure_finite(spectrum.precursor_mz().to_f64(), "precursor_mz")
                .map_err(FlashCosineIndexError::Computation)?;

            prepared.push((precursor_f64, mz_vals, data_vals));
        }

        let inner = FlashIndex::<CosineKernel>::build(tolerance, prepared)
            .map_err(FlashCosineIndexError::Computation)?;

        Ok(Self {
            inner,
            mz_power,
            intensity_power,
        })
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
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        Ok(self
            .inner
            .search_direct(&query_mz, &query_data, &query_meta))
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
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        Ok(self
            .inner
            .search_direct_with_state(&query_mz, &query_data, &query_meta, state))
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
    /// let index = FlashCosineIndex::new(0.0, 1.0, 0.1, &spectra).unwrap();
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
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        for_each_cosine_top_k_prepared(
            &self.inner,
            DirectThresholdSearch {
                query_mz: &query_mz,
                query_data: &query_data,
                query_meta: &query_meta,
                score_threshold,
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
            let query_meta = CosineKernel::spectrum_meta(&query_data);
            self.inner
                .for_each_direct_with_state(&query_mz, &query_data, &query_meta, state, emit);
            return Ok(());
        }
        if score_threshold > 1.0 {
            let _ = self.prepare_query(query)?;
            return Ok(());
        }

        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        let query_norm = query_meta.0;

        self.inner.for_each_direct_threshold_with_state(
            DirectThresholdSearch {
                query_mz: &query_mz,
                query_data: &query_data,
                query_meta: &query_meta,
                score_threshold,
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
/// threshold at construction time and builds library-side prefix postings for
/// that threshold. Search first intersects query-prefix and library-prefix
/// candidate filters, then exactly re-scores only the surviving spectra.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct FlashCosineThresholdIndex {
    inner: FlashIndex<CosineKernel>,
    mz_power: f64,
    intensity_power: f64,
    score_threshold: f64,
    prefix_postings: ThresholdPrefixPostings,
}

impl FlashCosineThresholdIndex {
    /// Build a threshold-specialized cosine index from an iterator of spectra.
    ///
    /// The threshold is part of the index. Build one index per cutoff, or one
    /// index at the lowest cutoff you need if you can accept less pruning for
    /// higher cutoffs.
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
        validate_non_negative_tolerance(mz_tolerance).map_err(FlashCosineIndexError::Config)?;
        ensure_finite(mz_power, "mz_power").map_err(FlashCosineIndexError::Computation)?;
        ensure_finite(intensity_power, "intensity_power")
            .map_err(FlashCosineIndexError::Computation)?;
        ensure_finite(score_threshold, "score_threshold")
            .map_err(FlashCosineIndexError::Computation)?;

        let tolerance = mz_tolerance;
        let mut prepared: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();

        for spectrum in spectra {
            let mz_vals: Vec<f64> = spectrum.mz().map(SpectrumFloat::to_f64).collect();
            let data_vals = normalized_peak_products(spectrum, mz_power, intensity_power)
                .map_err(FlashCosineIndexError::Computation)?;

            validate_well_separated(&mz_vals, tolerance, "library spectrum")
                .map_err(FlashCosineIndexError::Computation)?;

            let precursor_f64 = ensure_finite(spectrum.precursor_mz().to_f64(), "precursor_mz")
                .map_err(FlashCosineIndexError::Computation)?;

            prepared.push((precursor_f64, mz_vals, data_vals));
        }

        let prefix_postings = ThresholdPrefixPostings::build(&prepared, |data_vals| {
            l2_threshold_prefix_indices(data_vals, score_threshold)
        })
        .map_err(FlashCosineIndexError::Computation)?;
        let inner = FlashIndex::<CosineKernel>::build(tolerance, prepared)
            .map_err(FlashCosineIndexError::Computation)?;

        Ok(Self {
            inner,
            mz_power,
            intensity_power,
            score_threshold,
            prefix_postings,
        })
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

    /// Returns the number of library prefix peaks stored by this threshold
    /// index.
    #[inline]
    pub fn n_prefix_peaks(&self) -> usize {
        self.prefix_postings.n_prefix_peaks()
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
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        for_each_cosine_top_k_prepared(
            &self.inner,
            DirectThresholdSearch {
                query_mz: &query_mz,
                query_data: &query_data,
                query_meta: &query_meta,
                score_threshold: self.score_threshold,
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
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        self.for_each_prepared_with_state(&query_mz, &query_data, &query_meta, state, emit);
        Ok(())
    }

    /// Stream thresholded results for a query spectrum that is already in this
    /// index.
    ///
    /// This avoids per-query normalization and prefix sorting, which is the
    /// intended path for all-pairs graph construction over the indexed library.
    pub fn for_each_indexed_with_state<Emit>(
        &self,
        query_id: u32,
        state: &mut SearchState,
        emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        Emit: FnMut(FlashSearchResult),
    {
        if query_id >= self.inner.n_spectra {
            return Err(SimilarityComputationError::IndexOverflow);
        }

        let (query_mz, query_data) = self.inner.spectrum_slices(query_id);
        let query_meta = *self.inner.spectrum_meta(query_id);
        let query_prefix_mz = self.prefix_postings.spectrum_prefix_mz(query_id);

        self.for_each_prepared_indexed_with_state(
            query_mz,
            query_data,
            &query_meta,
            query_prefix_mz,
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
    /// let index = FlashCosineThresholdIndex::new(0.0, 1.0, 0.1, 0.8, &spectra).unwrap();
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
        if query_id >= self.inner.n_spectra {
            return Err(SimilarityComputationError::IndexOverflow);
        }

        let (query_mz, query_data) = self.inner.spectrum_slices(query_id);
        let query_meta = *self.inner.spectrum_meta(query_id);
        for_each_cosine_top_k_prepared(
            &self.inner,
            DirectThresholdSearch {
                query_mz,
                query_data,
                query_meta: &query_meta,
                score_threshold: self.score_threshold,
            },
            k,
            state,
            top_k_state,
            emit,
        );
        Ok(())
    }

    fn for_each_prepared_with_state<Emit>(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &CosineNorm,
        state: &mut SearchState,
        mut emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        if self.score_threshold <= 0.0 {
            self.inner
                .for_each_direct_with_state(query_mz, query_data, query_meta, state, emit);
            return;
        }
        if self.score_threshold > 1.0 || self.inner.n_spectra == 0 || query_mz.is_empty() {
            return;
        }
        if self.score_threshold < THRESHOLD_INDEX_ONE_SIDED_MIN_THRESHOLD {
            self.for_each_direct_threshold_prepared(query_mz, query_data, query_meta, state, emit);
            return;
        }

        state.prepare_threshold_order(query_data);
        let prefix_len = state.threshold_prefix_len(self.score_threshold);
        let query_order: Vec<usize> = state.query_order()[..prefix_len].to_vec();

        self.inner
            .mark_candidates_from_query_prefix_indices(query_mz, &query_order, state);
        if self.score_threshold < THRESHOLD_INDEX_TWO_SIDED_MIN_THRESHOLD {
            self.emit_exact_query_prefix_candidates(
                query_mz, query_data, query_meta, state, &mut emit,
            );
        } else {
            self.inner.intersect_candidates_with_library_prefixes(
                query_mz,
                &self.prefix_postings,
                state,
            );
            self.emit_exact_intersected_candidates(
                query_mz, query_data, query_meta, state, &mut emit,
            );
        }
    }

    fn for_each_prepared_indexed_with_state<Emit>(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &CosineNorm,
        query_prefix_mz: &[f64],
        state: &mut SearchState,
        mut emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        if self.score_threshold <= 0.0 {
            self.inner
                .for_each_direct_with_state(query_mz, query_data, query_meta, state, emit);
            return;
        }
        if self.score_threshold > 1.0 || self.inner.n_spectra == 0 || query_mz.is_empty() {
            return;
        }
        if self.score_threshold < THRESHOLD_INDEX_ONE_SIDED_MIN_THRESHOLD {
            self.for_each_direct_threshold_prepared(query_mz, query_data, query_meta, state, emit);
            return;
        }

        self.inner
            .mark_candidates_from_query_prefix_mz(query_prefix_mz, state);
        if self.score_threshold < THRESHOLD_INDEX_TWO_SIDED_MIN_THRESHOLD {
            self.emit_exact_query_prefix_candidates(
                query_mz, query_data, query_meta, state, &mut emit,
            );
        } else {
            self.inner.intersect_candidates_with_library_prefixes(
                query_mz,
                &self.prefix_postings,
                state,
            );
            self.emit_exact_intersected_candidates(
                query_mz, query_data, query_meta, state, &mut emit,
            );
        }
    }

    fn for_each_direct_threshold_prepared<Emit>(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &CosineNorm,
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
            },
            state,
            emit,
            |lib_meta| self.score_threshold * query_norm * lib_meta.0,
            |lib_meta| lib_meta.0,
        );
    }

    fn emit_exact_query_prefix_candidates<Emit>(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &CosineNorm,
        state: &mut SearchState,
        emit: &mut Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        let query_norm = query_meta.0;
        let mut target_raw_score =
            |lib_meta: &CosineNorm| self.score_threshold * query_norm * lib_meta.0;
        let mut library_bound = |lib_meta: &CosineNorm| lib_meta.0;
        self.inner.emit_exact_primary_candidates(
            DirectThresholdSearch {
                query_mz,
                query_data,
                query_meta,
                score_threshold: self.score_threshold,
            },
            state,
            emit,
            &mut target_raw_score,
            &mut library_bound,
        );
    }

    fn emit_exact_intersected_candidates<Emit>(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &CosineNorm,
        state: &mut SearchState,
        emit: &mut Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        let query_norm = query_meta.0;
        let mut target_raw_score =
            |lib_meta: &CosineNorm| self.score_threshold * query_norm * lib_meta.0;
        let mut library_bound = |lib_meta: &CosineNorm| lib_meta.0;
        self.inner.emit_exact_secondary_candidates(
            DirectThresholdSearch {
                query_mz,
                query_data,
                query_meta,
                score_threshold: self.score_threshold,
            },
            state,
            emit,
            &mut target_raw_score,
            &mut library_bound,
        );
    }
}

// ---------------------------------------------------------------------------
// SpectraIndex implementations
// ---------------------------------------------------------------------------

impl SpectraIndex for FlashCosineIndex {
    fn n_spectra(&self) -> u32 {
        self.n_spectra()
    }

    fn tolerance(&self) -> f64 {
        self.tolerance()
    }

    fn new_search_state(&self) -> SearchState {
        self.new_search_state()
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

impl SpectraIndex for FlashCosineThresholdIndex {
    fn n_spectra(&self) -> u32 {
        self.n_spectra()
    }

    fn tolerance(&self) -> f64 {
        self.tolerance()
    }

    fn new_search_state(&self) -> SearchState {
        self.new_search_state()
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
        let index =
            FlashCosineIndex::new(1.0, 2.0, 0.1, library.iter()).expect("index should build");

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
    fn modified_search_rejects_non_finite_query_precursor() {
        let library = [make_spectrum(200.0, &[(100.0, 1.0)])];
        let index =
            FlashCosineIndex::new(1.0, 1.0, 0.1, library.iter()).expect("index should build");
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
