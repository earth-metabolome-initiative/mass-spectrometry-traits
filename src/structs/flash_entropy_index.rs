//! Flash inverted m/z index for spectral entropy similarity.
//!
//! Provides O(Q_peaks * log(P_total)) library-scale search instead of
//! O(N * pairwise_cost). Exact equivalence to [`super::LinearEntropy`] on
//! well-separated spectra (consecutive peaks > 2 * tolerance apart).

use alloc::vec::Vec;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::cosine_common::{
    ensure_finite, validate_non_negative_tolerance, validate_numeric_parameter,
    validate_well_separated,
};
use super::entropy_common::{entropy_pair, prepare_entropy_peaks};
use super::flash_common::{
    DirectThresholdSearch, FlashIndex, FlashKernel, FlashSearchResult, PreparedFlashSpectra,
    PreparedFlashSpectrum, SearchState, TopKSearchResults, TopKSearchState,
};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{SpectraIndex, Spectrum, SpectrumFloat};

// ---------------------------------------------------------------------------
// EntropyKernel
// ---------------------------------------------------------------------------

pub(crate) struct EntropyKernel;

impl FlashKernel for EntropyKernel {
    type SpectrumMeta = ();

    #[inline]
    fn spectrum_meta(_peak_data: &[f64]) {}

    #[inline]
    fn pair_score(query: f64, library: f64) -> f64 {
        entropy_pair(query, library)
    }

    #[inline]
    fn finalize(raw: f64, _n_matches: usize, _query_meta: &(), _lib_meta: &()) -> f64 {
        (raw / 2.0).clamp(0.0, 1.0)
    }
}

#[inline]
fn entropy_peak_upper_bound(intensity: f64) -> f64 {
    entropy_pair(intensity, 1.0)
}

#[inline]
fn entropy_raw_threshold(score_threshold: f64) -> f64 {
    score_threshold * 2.0
}

/// Validate entropy index construction parameters.
fn validate_entropy_index_config(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
) -> Result<(), FlashEntropyIndexError> {
    validate_numeric_parameter(mz_power, "mz_power").map_err(FlashEntropyIndexError::Config)?;
    validate_numeric_parameter(intensity_power, "intensity_power")
        .map_err(FlashEntropyIndexError::Config)?;
    validate_non_negative_tolerance(mz_tolerance).map_err(FlashEntropyIndexError::Config)?;
    Ok(())
}

/// Prepare one entropy-library spectrum into the shared Flash representation.
fn prepare_entropy_spectrum<S>(
    spectrum: &S,
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    weighted: bool,
) -> Result<PreparedFlashSpectrum, FlashEntropyIndexError>
where
    S: Spectrum,
{
    let peaks = prepare_entropy_peaks(spectrum, weighted, mz_power, intensity_power)
        .map_err(FlashEntropyIndexError::Computation)?;

    if peaks.int.is_empty() {
        let precursor_f64 = ensure_finite(spectrum.precursor_mz().to_f64(), "precursor_mz")
            .map_err(FlashEntropyIndexError::Computation)?;
        return Ok((precursor_f64, Vec::new(), Vec::new()));
    }

    validate_well_separated(&peaks.mz, mz_tolerance, "library spectrum")
        .map_err(FlashEntropyIndexError::Computation)?;

    let precursor_f64 = ensure_finite(spectrum.precursor_mz().to_f64(), "precursor_mz")
        .map_err(FlashEntropyIndexError::Computation)?;

    Ok((precursor_f64, peaks.mz, peaks.int))
}

/// Prepare an entropy library sequentially from any borrowed-spectrum iterator.
fn prepare_entropy_library<'a, S>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    weighted: bool,
    spectra: impl IntoIterator<Item = &'a S>,
) -> Result<PreparedFlashSpectra, FlashEntropyIndexError>
where
    S: Spectrum + 'a,
{
    spectra
        .into_iter()
        .map(|spectrum| {
            prepare_entropy_spectrum(spectrum, mz_power, intensity_power, mz_tolerance, weighted)
        })
        .collect()
}

/// Prepare an entropy library in parallel from a Rayon-compatible
/// borrowed-spectrum collection.
#[cfg(feature = "rayon")]
fn prepare_entropy_library_parallel<'a, S, I>(
    mz_power: f64,
    intensity_power: f64,
    mz_tolerance: f64,
    weighted: bool,
    spectra: I,
) -> Result<PreparedFlashSpectra, FlashEntropyIndexError>
where
    S: Spectrum + Sync + 'a,
    I: IntoParallelIterator<Item = &'a S>,
{
    spectra
        .into_par_iter()
        .map(|spectrum| {
            prepare_entropy_spectrum(spectrum, mz_power, intensity_power, mz_tolerance, weighted)
        })
        .collect()
}

fn for_each_entropy_threshold_prepared<Emit>(
    inner: &FlashIndex<EntropyKernel>,
    query_mz: &[f64],
    query_data: &[f64],
    score_threshold: f64,
    state: &mut SearchState,
    mut emit: Emit,
) where
    Emit: FnMut(FlashSearchResult),
{
    if score_threshold <= 0.0 {
        inner.for_each_direct_with_state(query_mz, query_data, &(), state, emit);
        return;
    }
    if score_threshold > 1.0 || inner.n_spectra == 0 || query_mz.is_empty() {
        return;
    }

    state.prepare_additive_threshold_order(query_data, entropy_peak_upper_bound);
    let prefix_len = state.threshold_prefix_len_by_target(entropy_raw_threshold(score_threshold));
    let query_order: Vec<usize> = state.query_order()[..prefix_len].to_vec();
    inner.mark_candidates_from_query_prefix_indices(query_mz, &query_order, state);

    let mut target_raw_score = |_: &()| entropy_raw_threshold(score_threshold);
    let mut library_bound = |_: &()| 1.0_f64;
    inner.emit_exact_primary_candidates(
        DirectThresholdSearch {
            query_mz,
            query_data,
            query_meta: &(),
            score_threshold,
        },
        state,
        &mut emit,
        &mut target_raw_score,
        &mut library_bound,
    );
}

fn for_each_entropy_top_k_prepared<Emit>(
    inner: &FlashIndex<EntropyKernel>,
    search: DirectThresholdSearch<'_, EntropyKernel>,
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

    state.prepare_additive_threshold_order(search.query_data, entropy_peak_upper_bound);
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

            let (raw, count) =
                inner.direct_score_for_spectrum(search.query_mz, search.query_data, spec_id);
            if raw < entropy_raw_threshold(top_k.pruning_score()) {
                return;
            }

            let score = EntropyKernel::finalize(raw, count, search.query_meta, &());
            if score > 0.0 {
                top_k.push(FlashSearchResult {
                    spectrum_id: spec_id,
                    score,
                    n_matches: count,
                });
            }
        });

        let processed_prefix_len = order_position + 1;
        if state.query_suffix_bound_at(processed_prefix_len)
            < entropy_raw_threshold(top_k.pruning_score())
        {
            break;
        }
    }

    state.reset_candidates();
    top_k.emit(emit);
}

// ---------------------------------------------------------------------------
// FlashEntropyIndex
// ---------------------------------------------------------------------------

/// Flash inverted m/z index for spectral entropy similarity.
///
/// Build once from a library of spectra, then search many queries against it.
/// Produces the same scores as [`super::LinearEntropy`] on well-separated
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
/// let index = FlashEntropyIndex::new(0.0, 1.0, 0.1, true, library.iter())
///     .expect("index build should succeed");
///
/// let query: GenericSpectrum = GenericSpectrum::cocaine().unwrap();
/// let results = index.search(&query).expect("search should succeed");
/// assert!(results.iter().any(|r| r.spectrum_id == 0 && r.score > 0.99));
/// ```
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct FlashEntropyIndex {
    inner: FlashIndex<EntropyKernel>,
    weighted: bool,
    mz_power_f64: f64,
    intensity_power_f64: f64,
}

impl FlashEntropyIndex {
    /// Returns whether entropy-based intensity weighting is enabled.
    #[inline]
    pub fn is_weighted(&self) -> bool {
        self.weighted
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

    /// Returns the m/z power used for peak weighting.
    #[inline]
    pub fn mz_power_f64(&self) -> f64 {
        self.mz_power_f64
    }

    /// Returns the intensity power used for peak weighting.
    #[inline]
    pub fn intensity_power_f64(&self) -> f64 {
        self.intensity_power_f64
    }

    /// Build a weighted entropy flash index with default powers (mz_power=0,
    /// intensity_power=1).
    pub fn weighted<'a, S>(
        mz_tolerance: f64,
        spectra: impl IntoIterator<Item = &'a S>,
    ) -> Result<Self, FlashEntropyIndexError>
    where
        S: Spectrum + 'a,
    {
        Self::new(0.0, 1.0, mz_tolerance, true, spectra)
    }

    /// Build a weighted entropy flash index with Rayon-backed library
    /// preparation, internal index sorting, and default powers
    /// (mz_power=0, intensity_power=1).
    #[cfg(feature = "rayon")]
    pub fn weighted_parallel<'a, S, I>(
        mz_tolerance: f64,
        spectra: I,
    ) -> Result<Self, FlashEntropyIndexError>
    where
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
    {
        Self::new_parallel(0.0, 1.0, mz_tolerance, true, spectra)
    }

    /// Build an unweighted entropy flash index with default powers (mz_power=0,
    /// intensity_power=1).
    pub fn unweighted<'a, S>(
        mz_tolerance: f64,
        spectra: impl IntoIterator<Item = &'a S>,
    ) -> Result<Self, FlashEntropyIndexError>
    where
        S: Spectrum + 'a,
    {
        Self::new(0.0, 1.0, mz_tolerance, false, spectra)
    }

    /// Build an unweighted entropy flash index with Rayon-backed library
    /// preparation, internal index sorting, and default powers
    /// (mz_power=0, intensity_power=1).
    #[cfg(feature = "rayon")]
    pub fn unweighted_parallel<'a, S, I>(
        mz_tolerance: f64,
        spectra: I,
    ) -> Result<Self, FlashEntropyIndexError>
    where
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
    {
        Self::new_parallel(0.0, 1.0, mz_tolerance, false, spectra)
    }

    /// Build a new entropy flash index from an iterator of spectra.
    ///
    /// Each library spectrum must satisfy the well-separated precondition:
    /// consecutive peaks must be more than `2 * mz_tolerance` apart.
    ///
    /// # Errors
    ///
    /// - [`SimilarityConfigError`] if `mz_tolerance` is invalid or power
    ///   parameters are non-finite.
    /// - [`SimilarityComputationError`] if any spectrum violates the
    ///   well-separated precondition or contains non-representable values.
    pub fn new<'a, S>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        weighted: bool,
        spectra: impl IntoIterator<Item = &'a S>,
    ) -> Result<Self, FlashEntropyIndexError>
    where
        S: Spectrum + 'a,
    {
        validate_entropy_index_config(mz_power, intensity_power, mz_tolerance)?;

        let prepared =
            prepare_entropy_library(mz_power, intensity_power, mz_tolerance, weighted, spectra)?;

        let inner = FlashIndex::<EntropyKernel>::build(mz_tolerance, prepared)
            .map_err(FlashEntropyIndexError::Computation)?;

        Ok(Self {
            inner,
            weighted,
            mz_power_f64: mz_power,
            intensity_power_f64: intensity_power,
        })
    }

    /// Build a new entropy flash index with Rayon-backed library preparation
    /// and internal index sorting.
    ///
    /// This constructor is available with the `rayon` feature. It accepts any
    /// Rayon-compatible collection yielding borrowed spectra, such as `&[S]`
    /// or `&Vec<S>`.
    #[cfg(feature = "rayon")]
    pub fn new_parallel<'a, S, I>(
        mz_power: f64,
        intensity_power: f64,
        mz_tolerance: f64,
        weighted: bool,
        spectra: I,
    ) -> Result<Self, FlashEntropyIndexError>
    where
        S: Spectrum + Sync + 'a,
        I: IntoParallelIterator<Item = &'a S>,
    {
        validate_entropy_index_config(mz_power, intensity_power, mz_tolerance)?;

        let prepared = prepare_entropy_library_parallel(
            mz_power,
            intensity_power,
            mz_tolerance,
            weighted,
            spectra,
        )?;

        let inner = FlashIndex::<EntropyKernel>::build_parallel(mz_tolerance, prepared)
            .map_err(FlashEntropyIndexError::Computation)?;

        Ok(Self {
            inner,
            weighted,
            mz_power_f64: mz_power,
            intensity_power_f64: intensity_power,
        })
    }

    /// Create a [`SearchState`] sized for this index, suitable for reuse
    /// across multiple queries to avoid per-query allocation.
    pub fn new_search_state(&self) -> SearchState {
        self.inner.new_search_state()
    }

    /// Direct (unshifted) search against the library.
    pub fn search<S>(&self, query: &S) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        Ok(self.inner.search_direct(&query_mz, &query_data, &()))
    }

    /// Direct search using a caller-provided [`SearchState`].
    pub fn search_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        Ok(self
            .inner
            .search_direct_with_state(&query_mz, &query_data, &(), state))
    }

    /// Direct search that returns only results with
    /// `score >= score_threshold`.
    ///
    /// The entropy threshold path uses a per-peak entropy upper bound to avoid
    /// rescoring spectra that cannot reach the requested cutoff.
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

    /// Thresholded direct search using caller-provided scratch state.
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

    /// Thresholded direct search that emits each selected result.
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
        let (query_mz, query_data) = self.prepare_query(query)?;
        for_each_entropy_threshold_prepared(
            &self.inner,
            &query_mz,
            &query_data,
            score_threshold,
            state,
            emit,
        );
        Ok(())
    }

    /// Direct search that returns the best `k` results by descending score.
    ///
    /// This is exact. Once `k` results have been found, the current kth score
    /// is used with the entropy upper bound to prune unseen candidates.
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
    /// let index = FlashEntropyIndex::weighted(0.1, &spectra).unwrap();
    /// let hits = index.search_top_k(&spectra[0], 1).unwrap();
    ///
    /// assert_eq!(hits.len(), 1);
    /// assert_eq!(hits[0].spectrum_id, 0);
    /// ```
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

    /// Direct search that returns the best `k` results with
    /// `score >= score_threshold`.
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

    /// Stream the best `k` direct entropy results using caller-provided
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

    /// Stream the best `k` thresholded entropy results using caller-provided
    /// scratch state.
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
        for_each_entropy_top_k_prepared(
            &self.inner,
            DirectThresholdSearch {
                query_mz: &query_mz,
                query_data: &query_data,
                query_meta: &(),
                score_threshold,
            },
            k,
            state,
            top_k_state,
            emit,
        );
        Ok(())
    }

    /// Modified (direct + shifted) search against the library.
    pub fn search_modified<S>(
        &self,
        query: &S,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let precursor_f64 = ensure_finite(query.precursor_mz().to_f64(), "query_precursor_mz")?;
        Ok(self
            .inner
            .search_modified(&query_mz, &query_data, &(), precursor_f64))
    }

    /// Modified search using a caller-provided [`SearchState`].
    pub fn search_modified_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let precursor_f64 = ensure_finite(query.precursor_mz().to_f64(), "query_precursor_mz")?;
        Ok(self
            .inner
            .search_modified_with_state(&query_mz, &query_data, &(), precursor_f64, state))
    }

    /// Prepare query peaks: normalize, optionally weight, validate.
    fn prepare_query<S>(
        &self,
        query: &S,
    ) -> Result<(Vec<f64>, Vec<f64>), SimilarityComputationError>
    where
        S: Spectrum,
    {
        let peaks = prepare_entropy_peaks(
            query,
            self.weighted,
            self.mz_power_f64,
            self.intensity_power_f64,
        )?;

        if peaks.int.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        validate_well_separated(&peaks.mz, self.inner.tolerance, "query spectrum")?;

        Ok((peaks.mz, peaks.int))
    }
}

// ---------------------------------------------------------------------------
// SpectraIndex implementations
// ---------------------------------------------------------------------------

impl SpectraIndex for FlashEntropyIndex {
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

/// Error returned by [`FlashEntropyIndex`] construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FlashEntropyIndexError {
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
    fn entropy_index_accessors_convenience_constructors_and_wrappers_round_trip() {
        let library = [make_spectrum(200.0, &[(100.0, 4.0)])];
        let weighted =
            FlashEntropyIndex::weighted(0.1, library.iter()).expect("weighted index should build");
        assert!(weighted.is_weighted());
        assert_eq!(weighted.mz_power_f64(), 0.0);
        assert_eq!(weighted.intensity_power_f64(), 1.0);
        assert_eq!(weighted.tolerance(), 0.1);
        assert_eq!(weighted.n_spectra(), 1);

        let direct = weighted
            .search(&library[0])
            .expect("direct search should work");
        let mut direct_state = weighted.new_search_state();
        let direct_with_state = weighted
            .search_with_state(&library[0], &mut direct_state)
            .expect("stateful direct search should work");
        assert_eq!(direct, direct_with_state);

        let shifted_query = make_spectrum(210.0, &[(110.0, 4.0)]);
        let modified = weighted
            .search_modified(&shifted_query)
            .expect("modified search should work");
        let mut modified_state = weighted.new_search_state();
        let modified_with_state = weighted
            .search_modified_with_state(&shifted_query, &mut modified_state)
            .expect("stateful modified search should work");
        assert_eq!(modified, modified_with_state);

        let unweighted = FlashEntropyIndex::unweighted(0.1, library.iter())
            .expect("unweighted index should build");
        assert!(!unweighted.is_weighted());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_entropy_index_matches_sequential_constructor() {
        let library = [
            make_spectrum(500.0, &[(100.0, 10.0), (200.0, 20.0)]),
            make_spectrum(500.0, &[(100.05, 10.0), (200.05, 20.0)]),
            make_spectrum(500.0, &[(300.0, 10.0), (400.0, 20.0)]),
        ];
        let sequential = FlashEntropyIndex::weighted(0.1, library.iter())
            .expect("sequential entropy index should build");
        let parallel = FlashEntropyIndex::weighted_parallel(0.1, library.as_slice())
            .expect("parallel entropy index should build");

        assert_eq!(sequential.search(&library[0]), parallel.search(&library[0]));
        assert_eq!(
            sequential.search_threshold(&library[0], 0.5),
            parallel.search_threshold(&library[0], 0.5)
        );
        assert_eq!(
            sequential.search_top_k_threshold(&library[0], 2, 0.5),
            parallel.search_top_k_threshold(&library[0], 2, 0.5)
        );

        let shifted_query = make_spectrum(510.0, &[(110.0, 10.0), (210.0, 20.0)]);
        assert_eq!(
            sequential.search_modified(&shifted_query),
            parallel.search_modified(&shifted_query)
        );

        let unweighted = FlashEntropyIndex::unweighted_parallel(0.1, library.as_slice())
            .expect("parallel unweighted index should build");
        assert!(!unweighted.is_weighted());
    }

    #[test]
    fn entropy_index_treats_underflowed_products_as_empty() {
        let tiny = make_spectrum(200.0, &[(100.0, f64::MIN_POSITIVE)]);
        let index = FlashEntropyIndex::new(0.0, 2.0, 0.1, false, core::iter::once(&tiny))
            .expect("empty-product library should still build");

        assert_eq!(index.n_spectra(), 1);
        assert!(
            index
                .search(&tiny)
                .expect("empty-product query should succeed")
                .is_empty()
        );
        assert!(
            index
                .search_modified(&tiny)
                .expect("modified empty-product query should succeed")
                .is_empty()
        );
    }

    #[test]
    fn modified_search_rejects_non_finite_query_precursor() {
        let library = [make_spectrum(200.0, &[(100.0, 1.0)])];
        let index = FlashEntropyIndex::weighted(0.1, library.iter()).expect("index should build");
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
