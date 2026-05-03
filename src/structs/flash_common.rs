//! Shared internals for the Flash inverted m/z index.
//!
//! All items are `pub(crate)` — individual variant modules (`flash_cosine_index`,
//! `flash_entropy_index`) expose only their public wrappers.

use alloc::vec::Vec;
use core::cmp::Ordering;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::similarity_errors::SimilarityComputationError;
use crate::traits::SpectrumFloat;

const PREFIX_PRUNING_MIN_THRESHOLD: f64 = 0.85;
pub(crate) const DEFAULT_COSINE_SPECTRUM_BLOCK_SIZE: usize = 1024;
pub(crate) const DEFAULT_ENTROPY_SPECTRUM_BLOCK_SIZE: usize = 256;
#[cfg(feature = "experimental_block_size_env")]
const SPECTRUM_BLOCK_SIZE_ENV: &str = "MASS_SPECTROMETRY_FLASH_SPECTRUM_BLOCK_SIZE";

pub(crate) struct PreparedFlashSpectrum<P: SpectrumFloat> {
    pub(crate) precursor_mz: P,
    pub(crate) mz: Vec<P>,
    pub(crate) data: Vec<P>,
}

pub(crate) type PreparedFlashSpectra<P> = Vec<PreparedFlashSpectrum<P>>;

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Default)]
pub(crate) struct SpectrumIdMap {
    internal_to_public: Vec<u32>,
    public_to_internal: Vec<u32>,
}

impl SpectrumIdMap {
    pub(crate) fn identity() -> Self {
        Self::default()
    }

    #[cfg(feature = "experimental_reordered_index")]
    fn from_internal_to_public(
        internal_to_public: Vec<u32>,
    ) -> Result<Self, SimilarityComputationError> {
        if internal_to_public.is_empty() {
            return Ok(Self::identity());
        }

        let mut public_to_internal = alloc::vec![u32::MAX; internal_to_public.len()];
        for (internal_id, &public_id) in internal_to_public.iter().enumerate() {
            let public_index = usize::try_from(public_id)
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
            if public_index >= public_to_internal.len()
                || public_to_internal[public_index] != u32::MAX
            {
                return Err(SimilarityComputationError::IndexOverflow);
            }
            public_to_internal[public_index] = u32::try_from(internal_id)
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
        }

        Ok(Self {
            internal_to_public,
            public_to_internal,
        })
    }

    #[inline]
    pub(crate) fn public_to_internal(&self, public_id: u32) -> Option<u32> {
        if self.public_to_internal.is_empty() {
            return Some(public_id);
        }
        self.public_to_internal.get(public_id as usize).copied()
    }

    #[inline]
    pub(crate) fn internal_to_public(&self, internal_id: u32) -> u32 {
        if self.internal_to_public.is_empty() {
            return internal_id;
        }
        self.internal_to_public[internal_id as usize]
    }
}

#[cfg(feature = "experimental_reordered_index")]
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct SpectrumReorderKey {
    top_mz_bins: [i64; 8],
    precursor_bin: i64,
    n_peaks: u32,
    public_id: u32,
}

#[cfg(feature = "experimental_reordered_index")]
pub(crate) fn reorder_prepared_spectra_by_signature<P: SpectrumFloat>(
    spectra: PreparedFlashSpectra<P>,
    tolerance: f64,
) -> Result<(PreparedFlashSpectra<P>, SpectrumIdMap), SimilarityComputationError> {
    let n_spectra: u32 =
        u32::try_from(spectra.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
    if n_spectra <= 1 {
        return Ok((spectra, SpectrumIdMap::identity()));
    }

    let bin_width = if tolerance > 0.0 {
        (10.0 * tolerance).max(tolerance)
    } else {
        1.0
    };
    let mut keyed = Vec::with_capacity(spectra.len());
    for (public_id, spectrum) in spectra.into_iter().enumerate() {
        let public_id =
            u32::try_from(public_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let key = spectrum_reorder_key(&spectrum, public_id, bin_width)?;
        keyed.push((key, spectrum));
    }

    keyed.sort_unstable_by_key(|(key, _)| *key);

    let mut internal_to_public = Vec::with_capacity(keyed.len());
    let mut reordered = Vec::with_capacity(keyed.len());
    for (key, spectrum) in keyed {
        internal_to_public.push(key.public_id);
        reordered.push(spectrum);
    }
    let id_map = SpectrumIdMap::from_internal_to_public(internal_to_public)?;

    Ok((reordered, id_map))
}

#[cfg(feature = "experimental_reordered_index")]
fn spectrum_reorder_key<P: SpectrumFloat>(
    spectrum: &PreparedFlashSpectrum<P>,
    public_id: u32,
    bin_width: f64,
) -> Result<SpectrumReorderKey, SimilarityComputationError> {
    let precursor_bin = mz_bin(spectrum.precursor_mz.to_f64(), bin_width)?;
    let n_peaks =
        u32::try_from(spectrum.mz.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
    let mut top_mz_bins = [i64::MAX; 8];
    for (target, &mz) in top_mz_bins.iter_mut().zip(spectrum.mz.iter()) {
        *target = mz_bin(mz.to_f64(), bin_width)?;
    }

    Ok(SpectrumReorderKey {
        top_mz_bins,
        precursor_bin,
        n_peaks,
        public_id,
    })
}

#[cfg(feature = "experimental_reordered_index")]
fn mz_bin(mz: f64, bin_width: f64) -> Result<i64, SimilarityComputationError> {
    if !mz.is_finite() || !bin_width.is_finite() || bin_width <= 0.0 {
        return Err(SimilarityComputationError::NonFiniteValue("mz"));
    }
    Ok((mz / bin_width).floor() as i64)
}

pub(crate) fn spectrum_block_size(default_size: usize) -> usize {
    #[cfg(feature = "experimental_block_size_env")]
    {
        std::env::var(SPECTRUM_BLOCK_SIZE_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(default_size)
    }

    #[cfg(not(feature = "experimental_block_size_env"))]
    {
        default_size
    }
}

#[derive(Clone, Copy)]
enum SortBackend {
    Sequential,
    #[cfg(feature = "rayon")]
    Parallel,
}

pub(crate) fn convert_flash_value<P: SpectrumFloat>(
    value: f64,
    name: &'static str,
) -> Result<P, SimilarityComputationError> {
    if !value.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue(name));
    }
    P::from_f64(value).ok_or(SimilarityComputationError::NonFiniteValue(name))
}

pub(crate) fn convert_flash_values<P: SpectrumFloat>(
    values: impl IntoIterator<Item = f64>,
    name: &'static str,
) -> Result<Vec<P>, SimilarityComputationError> {
    values
        .into_iter()
        .map(|value| convert_flash_value(value, name))
        .collect()
}

pub(crate) fn flash_values_to_f64<P: SpectrumFloat>(values: &[P]) -> Vec<f64> {
    values.iter().map(|value| value.to_f64()).collect()
}

fn compare_indexed_values<P: SpectrumFloat>(values: &[P], left: u32, right: u32) -> Ordering {
    values[left as usize]
        .to_f64()
        .total_cmp(&values[right as usize].to_f64())
        .then_with(|| left.cmp(&right))
}

fn sort_permutation_by_values<P>(perm: &mut [u32], values: &[P], backend: SortBackend)
where
    P: SpectrumFloat + Sync,
{
    match backend {
        SortBackend::Sequential => {
            perm.sort_unstable_by(|&left, &right| compare_indexed_values(values, left, right));
        }
        #[cfg(feature = "rayon")]
        SortBackend::Parallel => {
            perm.par_sort_unstable_by(|&left, &right| compare_indexed_values(values, left, right));
        }
    }
}

fn sort_mz_values<P: SpectrumFloat>(values: &mut [P]) {
    values.sort_unstable_by(|left, right| left.to_f64().total_cmp(&right.to_f64()));
}

struct SearchBitVec {
    words: Vec<usize>,
    len: usize,
}

impl SearchBitVec {
    const WORD_BITS: usize = usize::BITS as usize;

    fn new() -> Self {
        Self {
            words: Vec::new(),
            len: 0,
        }
    }

    fn zeros(len: usize) -> Self {
        Self {
            words: alloc::vec![0; len.div_ceil(Self::WORD_BITS)],
            len,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.len);
        let word = self.words[index / Self::WORD_BITS];
        let mask = 1usize << (index % Self::WORD_BITS);
        word & mask != 0
    }

    #[inline]
    fn set(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.len);
        let word = &mut self.words[index / Self::WORD_BITS];
        let mask = 1usize << (index % Self::WORD_BITS);
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    #[cfg(feature = "mem_size")]
    fn stored_bytes(&self, flags: mem_dbg::SizeFlags) -> usize {
        let words = if flags.contains(mem_dbg::SizeFlags::CAPACITY) {
            self.words.capacity()
        } else {
            self.words.len()
        };
        words * core::mem::size_of::<usize>()
    }
}

#[cfg(feature = "mem_size")]
impl mem_dbg::FlatType for SearchBitVec {
    type Flat = mem_dbg::False;
}

#[cfg(feature = "mem_size")]
impl mem_dbg::MemSize for SearchBitVec {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        _refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        core::mem::size_of::<Self>() + self.stored_bytes(flags)
    }
}

#[cfg(feature = "mem_dbg")]
impl mem_dbg::MemDbgImpl for SearchBitVec {}

// ---------------------------------------------------------------------------
// FlashKernel — scoring kernel abstraction
// ---------------------------------------------------------------------------

/// Scoring kernel abstraction for the Flash index.
///
/// Each kernel defines how to prepare per-peak scoring data, compute per-spectrum
/// metadata (e.g. a norm), score a single matched pair, and finalize the
/// accumulated raw score into a `[0, 1]` similarity.
pub(crate) trait FlashKernel {
    /// Per-spectrum metadata (norm for cosine, `()` for entropy).
    type SpectrumMeta: Copy + Default;

    /// Compute per-spectrum metadata from all prepared peak data values.
    fn spectrum_meta<P: SpectrumFloat>(peak_data: &[P]) -> Self::SpectrumMeta;

    /// Score contribution of a single matched pair.
    fn pair_score(query: f64, library: f64) -> f64;

    /// Finalize accumulated raw score into `[0, 1]`.
    fn finalize(
        raw: f64,
        n_matches: usize,
        query_meta: &Self::SpectrumMeta,
        lib_meta: &Self::SpectrumMeta,
    ) -> f64;
}

// ---------------------------------------------------------------------------
// FlashSearchResult
// ---------------------------------------------------------------------------

/// A single search result from a Flash index query.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlashSearchResult {
    /// Index of the library spectrum (0-based, insertion order).
    pub spectrum_id: u32,
    /// Similarity score in `[0, 1]`.
    pub score: f64,
    /// Number of matched peak pairs.
    pub n_matches: usize,
}

/// Per-query diagnostic counters for Flash index searches.
///
/// These counters are intended for benchmarks and profiling. They describe the
/// most recent search run with a [`SearchState`] and are reset by public search
/// entry points before executing a query.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlashSearchDiagnostics {
    /// Product-ion postings visited while scanning m/z windows.
    pub product_postings_visited: usize,
    /// Threshold-prefix postings visited while intersecting candidate sets.
    pub prefix_postings_visited: usize,
    /// Unique primary candidates marked for exact scoring.
    pub candidates_marked: usize,
    /// Unique secondary candidates retained after prefix intersection.
    pub secondary_candidates_marked: usize,
    /// Candidates passed to exact scoring.
    pub candidates_rescored: usize,
    /// Results emitted after threshold/top-k filtering.
    pub results_emitted: usize,
    /// Spectrum blocks considered by a threshold upper-bound filter.
    pub spectrum_blocks_evaluated: usize,
    /// Spectrum blocks whose upper bound was high enough to keep.
    pub spectrum_blocks_allowed: usize,
    /// Spectrum blocks rejected before candidate marking.
    pub spectrum_blocks_pruned: usize,
}

pub(crate) fn compare_search_results_by_rank(
    left: &FlashSearchResult,
    right: &FlashSearchResult,
) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| right.n_matches.cmp(&left.n_matches))
        .then_with(|| left.spectrum_id.cmp(&right.spectrum_id))
}

/// Reusable scratch space for top-k Flash index searches.
///
/// Use one `TopKSearchState` per worker together with [`SearchState`] when
/// running many top-k queries, so the bounded candidate buffer can be reused
/// instead of allocated for every query.
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
/// let mut search_state = index.new_search_state();
/// let mut top_k_state = TopKSearchState::new();
/// let mut hits = Vec::new();
///
/// index
///     .for_each_top_k_threshold_with_state(
///         &spectra[0],
///         2,
///         0.8,
///         &mut search_state,
///         &mut top_k_state,
///         |hit| hits.push(hit),
///     )
///     .unwrap();
///
/// assert_eq!(hits[0].spectrum_id, 0);
/// assert!(hits.iter().any(|hit| hit.spectrum_id == 1));
/// ```
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Default)]
pub struct TopKSearchState {
    results: Vec<FlashSearchResult>,
}

impl TopKSearchState {
    /// Create an empty top-k scratch state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) struct TopKSearchResults<'a> {
    k: usize,
    min_score: f64,
    results: &'a mut Vec<FlashSearchResult>,
    worst_index: Option<usize>,
    pruning_score: f64,
}

impl<'a> TopKSearchResults<'a> {
    /// Prepare the bounded result buffer for one query.
    pub(crate) fn new(k: usize, min_score: f64, state: &'a mut TopKSearchState) -> Self {
        state.results.clear();
        if state.results.capacity() < k {
            state.results.reserve(k - state.results.capacity());
        }
        Self {
            k,
            min_score,
            results: &mut state.results,
            worst_index: None,
            pruning_score: min_score,
        }
    }

    /// Insert a result if it is good enough for the current top-k set.
    #[inline]
    pub(crate) fn push(&mut self, result: FlashSearchResult) -> bool {
        if self.k == 0 || result.score < self.min_score {
            return false;
        }
        if self.results.len() < self.k {
            self.results.push(result);
            if self.results.len() == self.k {
                self.refresh_worst();
            }
            return true;
        }

        let worst_index = self
            .worst_index
            .expect("full top-k buffer should have a cached worst index");
        let worst_result = self.results[worst_index];

        if compare_search_results_by_rank(&result, &worst_result).is_lt() {
            self.results[worst_index] = result;
            self.refresh_worst();
            return true;
        }
        false
    }

    /// Return the score that a future candidate must reach to matter.
    #[inline]
    pub(crate) fn pruning_score(&self) -> f64 {
        self.pruning_score
    }

    /// Return the number of results currently retained.
    pub(crate) fn len(&self) -> usize {
        self.results.len()
    }

    /// Sort selected results into the public deterministic rank order.
    fn finish(&mut self) {
        self.results.sort_by(compare_search_results_by_rank);
    }

    /// Refresh the cached replacement position and pruning score.
    fn refresh_worst(&mut self) {
        if self.results.len() < self.k {
            self.worst_index = None;
            self.pruning_score = self.min_score;
            return;
        }

        let Some((worst_index, worst_result)) = self
            .results
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| compare_search_results_by_rank(left, right))
        else {
            self.worst_index = None;
            self.pruning_score = self.min_score;
            return;
        };

        self.worst_index = Some(worst_index);
        self.pruning_score = worst_result.score.max(self.min_score);
    }

    /// Emit selected results after final ranking.
    pub(crate) fn emit<Emit>(mut self, mut emit: Emit)
    where
        Emit: FnMut(FlashSearchResult),
    {
        self.finish();
        for &result in self.results.iter() {
            emit(result);
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DirectThresholdSearch<'a, K: FlashKernel, Q: SpectrumFloat> {
    pub(crate) query_mz: &'a [Q],
    pub(crate) query_data: &'a [Q],
    pub(crate) query_meta: &'a K::SpectrumMeta,
    pub(crate) score_threshold: f64,
}

/// Library-side prefix postings for threshold-specialized direct searches.
///
/// A fixed-threshold index chooses how each spectrum's prefix is computed,
/// then this shared structure stores those prefix peaks in a sorted m/z index
/// and in per-spectrum order for indexed-query graph construction.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct ThresholdPrefixPostings<P: SpectrumFloat = f64> {
    spectrum_prefix_offsets: Vec<u32>,
    spectrum_prefix_mz: Vec<P>,
}

impl<P: SpectrumFloat> ThresholdPrefixPostings<P> {
    pub(crate) fn build(
        spectra: &[PreparedFlashSpectrum<P>],
        mut prefix_indices: impl FnMut(&[P]) -> Vec<usize>,
    ) -> Result<Self, SimilarityComputationError> {
        let mut spectrum_prefix_offsets = Vec::with_capacity(spectra.len() + 1);
        let mut spectrum_prefix_mz = Vec::new();
        spectrum_prefix_offsets.push(0);

        for (spec_id, spectrum) in spectra.iter().enumerate() {
            let _: u32 =
                u32::try_from(spec_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            let mut prefix_mz: Vec<P> = prefix_indices(&spectrum.data)
                .into_iter()
                .map(|peak_index| spectrum.mz[peak_index])
                .collect();
            sort_mz_values(&mut prefix_mz);

            for mz in prefix_mz {
                spectrum_prefix_mz.push(mz);
            }
            let offset = u32::try_from(spectrum_prefix_mz.len())
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
            spectrum_prefix_offsets.push(offset);
        }

        Ok(Self {
            spectrum_prefix_offsets,
            spectrum_prefix_mz,
        })
    }

    #[inline]
    pub(crate) fn n_prefix_peaks(&self) -> usize {
        self.spectrum_prefix_mz.len()
    }

    pub(crate) fn spectrum_prefix_mz(&self, spec_id: u32) -> &[P] {
        let prefix_start = self.spectrum_prefix_offsets[spec_id as usize] as usize;
        let prefix_end = self.spectrum_prefix_offsets[spec_id as usize + 1] as usize;
        &self.spectrum_prefix_mz[prefix_start..prefix_end]
    }
}

/// Product-ion postings partitioned by contiguous spectrum-id blocks.
///
/// The global product index is optimal when every spectrum is eligible. Once a
/// query-level block bound has selected a tiny subset of spectrum blocks, this
/// layout lets candidate generation binary-search only those block-local peak
/// lists and skip postings from every rejected block.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct SpectrumBlockProductIndex<P: SpectrumFloat = f64> {
    block_offsets: Vec<u32>,
    mz: Vec<P>,
    data: Vec<P>,
    spec_id: Vec<u32>,
}

impl<P: SpectrumFloat> SpectrumBlockProductIndex<P> {
    pub(crate) fn build(
        spectra: &[PreparedFlashSpectrum<P>],
        block_size: usize,
    ) -> Result<Self, SimilarityComputationError> {
        let block_size = block_size.max(1);
        let n_blocks = spectra.len().div_ceil(block_size);
        let total_peaks: usize = spectra.iter().map(|spectrum| spectrum.mz.len()).sum();
        let _: u32 =
            u32::try_from(total_peaks).map_err(|_| SimilarityComputationError::IndexOverflow)?;

        let mut entries_by_block: Vec<Vec<(P, P, u32)>> =
            (0..n_blocks).map(|_| Vec::new()).collect();
        for (spec_id, spectrum) in spectra.iter().enumerate() {
            let spec_id_u32 =
                u32::try_from(spec_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            let block_id = spec_id / block_size;
            for (&mz, &data) in spectrum.mz.iter().zip(spectrum.data.iter()) {
                entries_by_block[block_id].push((mz, data, spec_id_u32));
            }
        }

        let mut block_offsets = Vec::with_capacity(n_blocks + 1);
        let mut mz = Vec::with_capacity(total_peaks);
        let mut data = Vec::with_capacity(total_peaks);
        let mut spec_id = Vec::with_capacity(total_peaks);
        block_offsets.push(0);

        for entries in &mut entries_by_block {
            entries.sort_unstable_by(|left, right| {
                left.0
                    .to_f64()
                    .total_cmp(&right.0.to_f64())
                    .then_with(|| left.2.cmp(&right.2))
            });
            for &(entry_mz, entry_data, entry_spec_id) in entries.iter() {
                mz.push(entry_mz);
                data.push(entry_data);
                spec_id.push(entry_spec_id);
            }
            let offset =
                u32::try_from(mz.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            block_offsets.push(offset);
        }

        Ok(Self {
            block_offsets,
            mz,
            data,
            spec_id,
        })
    }

    pub(crate) fn for_each_peak_in_window<Q: SpectrumFloat>(
        &self,
        block_id: u32,
        mz: Q,
        tolerance: f64,
        mut emit: impl FnMut(u32, P),
    ) -> usize {
        let block_index = block_id as usize;
        if block_index + 1 >= self.block_offsets.len() {
            return 0;
        }

        let start = self.block_offsets[block_index] as usize;
        let end = self.block_offsets[block_index + 1] as usize;
        let block_mz = &self.mz[start..end];
        let query_mz = mz.to_f64();
        let lo = query_mz - tolerance;
        let hi = query_mz + tolerance;
        let first = block_mz.partition_point(|&value| value.to_f64() < lo);

        let mut visited = 0usize;
        for (relative_index, &entry_mz) in block_mz.iter().enumerate().skip(first) {
            let product_mz = entry_mz.to_f64();
            if product_mz > hi {
                break;
            }
            visited = visited.saturating_add(1);
            emit(
                self.spec_id[start + relative_index],
                self.data[start + relative_index],
            );
        }

        visited
    }
}

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Clone, Copy)]
struct SpectrumBlockUpperBound {
    block_id: u32,
    max_value: f64,
}

/// Sparse spectrum-block upper bounds over coarse m/z bins.
///
/// This is a shared pruning-only index. Each m/z bin stores, per contiguous
/// spectrum-id block, the maximum metric-specific peak value observed in that
/// block. Query code supplies the metric-specific function that turns a query
/// peak and a stored block maximum into a safe score upper-bound contribution.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct SpectrumBlockUpperBoundIndex {
    bin_width: f64,
    min_bin: i64,
    bins: Vec<Vec<SpectrumBlockUpperBound>>,
    n_blocks: usize,
}

impl SpectrumBlockUpperBoundIndex {
    pub(crate) fn build<P, Values>(
        spectra: &[PreparedFlashSpectrum<P>],
        tolerance: f64,
        block_size: usize,
        mut peak_values: Values,
    ) -> Result<Self, SimilarityComputationError>
    where
        P: SpectrumFloat,
        Values:
            FnMut(u32, &PreparedFlashSpectrum<P>) -> Result<Vec<f64>, SimilarityComputationError>,
    {
        let block_size = block_size.max(1);
        let n_blocks = spectra.len().div_ceil(block_size);
        let bin_width = if tolerance > 0.0 {
            2.0 * tolerance
        } else {
            1.0
        };

        let mut min_bin = i64::MAX;
        let mut max_bin = i64::MIN;
        for spectrum in spectra {
            for &mz in &spectrum.mz {
                let bin = Self::bin_id(mz.to_f64(), bin_width);
                min_bin = min_bin.min(bin);
                max_bin = max_bin.max(bin);
            }
        }

        if min_bin == i64::MAX {
            return Ok(Self {
                bin_width,
                min_bin: 0,
                bins: Vec::new(),
                n_blocks,
            });
        }

        let n_bins = usize::try_from(max_bin - min_bin + 1)
            .map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let total_peaks = spectra.iter().map(|spectrum| spectrum.mz.len()).sum();
        let mut entries: Vec<(usize, u32, f64)> = Vec::with_capacity(total_peaks);

        for (spec_id, spectrum) in spectra.iter().enumerate() {
            let spec_id_u32 =
                u32::try_from(spec_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            let values = peak_values(spec_id_u32, spectrum)?;
            debug_assert_eq!(values.len(), spectrum.mz.len());

            let block_id = u32::try_from(spec_id / block_size)
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
            for (&mz, &value) in spectrum.mz.iter().zip(values.iter()) {
                if !value.is_finite() {
                    return Err(SimilarityComputationError::NonFiniteValue(
                        "block_upper_bound",
                    ));
                }
                if value <= 0.0 {
                    continue;
                }

                let bin_index = usize::try_from(Self::bin_id(mz.to_f64(), bin_width) - min_bin)
                    .map_err(|_| SimilarityComputationError::IndexOverflow)?;
                entries.push((bin_index, block_id, value));
            }
        }

        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.total_cmp(&right.2))
        });

        let mut bins: Vec<Vec<SpectrumBlockUpperBound>> = (0..n_bins).map(|_| Vec::new()).collect();
        for (bin_index, block_id, value) in entries {
            if let Some(last) = bins[bin_index].last_mut()
                && last.block_id == block_id
            {
                last.max_value = last.max_value.max(value);
                continue;
            }
            bins[bin_index].push(SpectrumBlockUpperBound {
                block_id,
                max_value: value,
            });
        }

        Ok(Self {
            bin_width,
            min_bin,
            bins,
            n_blocks,
        })
    }

    pub(crate) fn prepare_allowed_blocks<Q, Contribution>(
        &self,
        query_mz: &[Q],
        tolerance: f64,
        minimum_bound: f64,
        state: &mut SearchState,
        mut contribution: Contribution,
    ) where
        Q: SpectrumFloat,
        Contribution: FnMut(usize, f64) -> f64,
    {
        state.prepare_spectrum_block_scratch(self.n_blocks);
        if self.n_blocks == 0 || self.bins.is_empty() {
            state.add_spectrum_block_filter_stats(self.n_blocks, 0);
            return;
        }

        for (query_index, &mz) in query_mz.iter().enumerate() {
            for bin_index in self.bin_indices_for_window(mz.to_f64(), tolerance) {
                for entry in &self.bins[bin_index] {
                    state.add_spectrum_block_upper_bound(
                        entry.block_id,
                        contribution(query_index, entry.max_value),
                    );
                }
            }
        }

        let allowed_blocks = state.mark_allowed_spectrum_blocks(minimum_bound);
        state.add_spectrum_block_filter_stats(self.n_blocks, allowed_blocks);
    }

    fn bin_indices_for_window(&self, mz: f64, tolerance: f64) -> impl Iterator<Item = usize> + '_ {
        let lo_bin = Self::bin_id(mz - tolerance, self.bin_width);
        let hi_bin = Self::bin_id(mz + tolerance, self.bin_width);
        let start = lo_bin.max(self.min_bin);
        let end = hi_bin.min(self.min_bin + self.bins.len() as i64 - 1);

        let (start, end) = if start <= end {
            (
                (start - self.min_bin) as usize,
                (end - self.min_bin) as usize,
            )
        } else {
            (1, 0)
        };

        start..=end
    }

    #[inline]
    fn bin_id(mz: f64, bin_width: f64) -> i64 {
        (mz / bin_width).floor() as i64
    }
}

/// Prefix indices for score bounds that can be represented as an L2 suffix
/// norm over per-peak weights.
pub(crate) fn l2_threshold_prefix_indices<P: SpectrumFloat>(
    peak_weights: &[P],
    score_threshold: f64,
) -> Vec<usize> {
    if peak_weights.is_empty() || score_threshold > 1.0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..peak_weights.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        peak_weights[right]
            .to_f64()
            .abs()
            .total_cmp(&peak_weights[left].to_f64().abs())
            .then_with(|| left.cmp(&right))
    });

    if score_threshold <= 0.0 {
        return order;
    }

    let norm = peak_weights
        .iter()
        .map(|&value| {
            let value = value.to_f64();
            value * value
        })
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Vec::new();
    }

    let mut suffix_norm = alloc::vec![0.0_f64; order.len() + 1];
    for order_index in (0..order.len()).rev() {
        let peak_index = order[order_index];
        let peak_weight = peak_weights[peak_index].to_f64();
        suffix_norm[order_index] = suffix_norm[order_index + 1] + peak_weight * peak_weight;
    }
    for value in &mut suffix_norm {
        *value = value.sqrt();
    }

    let target_norm = score_threshold * norm;
    let prefix_len = suffix_norm
        .iter()
        .position(|&remaining_norm| remaining_norm < target_norm)
        .unwrap_or(order.len());
    order.truncate(prefix_len);
    order
}

// ---------------------------------------------------------------------------
// DenseAccumulator
// ---------------------------------------------------------------------------

/// Per-query score accumulator. Uses dense arrays (one slot per library
/// spectrum) with a `touched` list for efficient reset.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
struct DenseAccumulator {
    scores: Vec<f64>,
    counts: Vec<u32>,
    touched: Vec<u32>,
}

impl DenseAccumulator {
    fn new() -> Self {
        Self {
            scores: Vec::new(),
            counts: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, n_spectra: usize) {
        if self.scores.len() == n_spectra {
            return;
        }

        self.scores.clear();
        self.counts.clear();
        self.touched.clear();
        self.scores.resize(n_spectra, 0.0);
        self.counts.resize(n_spectra, 0);
    }

    #[inline]
    fn accumulate(&mut self, spec_id: u32, score: f64) {
        let idx = spec_id as usize;
        if self.counts[idx] == 0 {
            self.touched.push(spec_id);
        }
        self.scores[idx] += score;
        self.counts[idx] = self.counts[idx].saturating_add(1);
    }

    #[inline]
    fn touched_len(&self) -> usize {
        self.touched.len()
    }

    /// Replace a previously accumulated score with a better one, without
    /// changing the match count.
    #[inline]
    fn upgrade(&mut self, spec_id: u32, old_score: f64, new_score: f64) {
        self.scores[spec_id as usize] += new_score - old_score;
    }

    /// Drain the accumulator, calling `emit` for each spectrum that received
    /// at least one match. Resets all touched slots for reuse.
    fn drain(&mut self, mut emit: impl FnMut(u32, f64, u32)) {
        for &id in &self.touched {
            let idx = id as usize;
            emit(id, self.scores[idx], self.counts[idx]);
            self.scores[idx] = 0.0;
            self.counts[idx] = 0;
        }
        self.touched.clear();
    }
}

// ---------------------------------------------------------------------------
// SearchState — reusable per-query scratch space
// ---------------------------------------------------------------------------

/// Reusable scratch space for Flash index searches.
///
/// Create one `SearchState` per worker and pass it to repeated search calls to
/// reuse scratch buffers. Large buffers are allocated lazily by the search mode
/// that needs them, so direct threshold searches do not pay for modified-search
/// scratch.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct SearchState {
    acc: DenseAccumulator,
    matched_products: SearchBitVec,
    direct_scores: Vec<f64>,
    candidate_spectra: SearchBitVec,
    candidate_touched: Vec<u32>,
    block_upper_bounds: Vec<f64>,
    block_upper_bound_touched: Vec<u32>,
    allowed_spectrum_blocks: SearchBitVec,
    allowed_block_touched: Vec<u32>,
    query_order: Vec<usize>,
    query_suffix_bound: Vec<f64>,
    diagnostics: FlashSearchDiagnostics,
}

impl SearchState {
    /// Create a new `SearchState` sized for the given index.
    fn new(n_spectra: usize, n_products: usize) -> Self {
        let _ = (n_spectra, n_products);
        Self {
            acc: DenseAccumulator::new(),
            matched_products: SearchBitVec::new(),
            direct_scores: Vec::new(),
            candidate_spectra: SearchBitVec::new(),
            candidate_touched: Vec::new(),
            block_upper_bounds: Vec::new(),
            block_upper_bound_touched: Vec::new(),
            allowed_spectrum_blocks: SearchBitVec::new(),
            allowed_block_touched: Vec::new(),
            query_order: Vec::new(),
            query_suffix_bound: Vec::new(),
            diagnostics: FlashSearchDiagnostics::default(),
        }
    }

    /// Return diagnostics for the most recent query executed with this state.
    #[inline]
    pub fn diagnostics(&self) -> FlashSearchDiagnostics {
        self.diagnostics
    }

    /// Reset query diagnostics.
    #[inline]
    pub fn reset_diagnostics(&mut self) {
        self.diagnostics = FlashSearchDiagnostics::default();
    }

    #[inline]
    pub(crate) fn add_product_postings_visited(&mut self, count: usize) {
        self.diagnostics.product_postings_visited = self
            .diagnostics
            .product_postings_visited
            .saturating_add(count);
    }

    #[inline]
    pub(crate) fn add_candidates_rescored(&mut self, count: usize) {
        self.diagnostics.candidates_rescored =
            self.diagnostics.candidates_rescored.saturating_add(count);
    }

    #[inline]
    pub(crate) fn add_candidates_marked(&mut self, count: usize) {
        self.diagnostics.candidates_marked =
            self.diagnostics.candidates_marked.saturating_add(count);
    }

    #[inline]
    pub(crate) fn add_results_emitted(&mut self, count: usize) {
        self.diagnostics.results_emitted = self.diagnostics.results_emitted.saturating_add(count);
    }

    /// Records how many spectrum blocks were kept by a query-level upper-bound filter.
    #[inline]
    pub(crate) fn add_spectrum_block_filter_stats(
        &mut self,
        blocks_evaluated: usize,
        blocks_allowed: usize,
    ) {
        self.diagnostics.spectrum_blocks_evaluated = self
            .diagnostics
            .spectrum_blocks_evaluated
            .saturating_add(blocks_evaluated);
        self.diagnostics.spectrum_blocks_allowed = self
            .diagnostics
            .spectrum_blocks_allowed
            .saturating_add(blocks_allowed);
        self.diagnostics.spectrum_blocks_pruned = self
            .diagnostics
            .spectrum_blocks_pruned
            .saturating_add(blocks_evaluated.saturating_sub(blocks_allowed));
    }

    pub(crate) fn ensure_candidate_capacity(&mut self, n_spectra: usize) {
        if self.candidate_spectra.len() == n_spectra {
            return;
        }

        self.candidate_spectra = SearchBitVec::zeros(n_spectra);
        self.candidate_touched.clear();
    }

    fn ensure_modified_capacity(&mut self, n_products: usize) {
        if self.matched_products.len() == n_products && self.direct_scores.len() == n_products {
            return;
        }

        self.matched_products = SearchBitVec::zeros(n_products);
        self.direct_scores.clear();
        self.direct_scores.resize(n_products, 0.0);
    }

    pub(crate) fn prepare_threshold_order<P: SpectrumFloat>(&mut self, query_data: &[P]) {
        self.query_order.clear();
        self.query_order.extend(0..query_data.len());
        self.query_order.sort_unstable_by(|&left, &right| {
            query_data[right]
                .to_f64()
                .abs()
                .total_cmp(&query_data[left].to_f64().abs())
                .then_with(|| left.cmp(&right))
        });

        self.query_suffix_bound.clear();
        self.query_suffix_bound.resize(query_data.len() + 1, 0.0);
        for order_index in (0..self.query_order.len()).rev() {
            let query_index = self.query_order[order_index];
            let query_value = query_data[query_index].to_f64();
            self.query_suffix_bound[order_index] =
                self.query_suffix_bound[order_index + 1] + query_value * query_value;
        }
        for value in &mut self.query_suffix_bound {
            *value = value.sqrt();
        }
    }

    pub(crate) fn prepare_additive_threshold_order<P: SpectrumFloat>(
        &mut self,
        query_data: &[P],
        mut upper_bound: impl FnMut(f64) -> f64,
    ) {
        self.query_order.clear();
        self.query_order.extend(0..query_data.len());
        self.query_order.sort_unstable_by(|&left, &right| {
            upper_bound(query_data[right].to_f64())
                .total_cmp(&upper_bound(query_data[left].to_f64()))
                .then_with(|| left.cmp(&right))
        });

        self.query_suffix_bound.clear();
        self.query_suffix_bound.resize(query_data.len() + 1, 0.0);
        for order_index in (0..self.query_order.len()).rev() {
            let query_index = self.query_order[order_index];
            self.query_suffix_bound[order_index] = self.query_suffix_bound[order_index + 1]
                + upper_bound(query_data[query_index].to_f64());
        }
    }

    pub(crate) fn threshold_prefix_len_by_target(&self, target: f64) -> usize {
        self.query_suffix_bound
            .iter()
            .position(|&remaining_bound| remaining_bound < target)
            .unwrap_or(self.query_order.len())
    }

    pub(crate) fn query_order(&self) -> &[usize] {
        &self.query_order
    }

    pub(crate) fn query_suffix_bound_at(&self, index: usize) -> f64 {
        self.query_suffix_bound[index]
    }

    #[inline]
    pub(crate) fn mark_candidate(&mut self, spec_id: u32) {
        let idx = spec_id as usize;
        if !self.candidate_spectra.get(idx) {
            self.candidate_spectra.set(idx, true);
            self.candidate_touched.push(spec_id);
            self.diagnostics.candidates_marked =
                self.diagnostics.candidates_marked.saturating_add(1);
        }
    }

    #[inline]
    pub(crate) fn is_candidate(&self, spec_id: u32) -> bool {
        self.candidate_spectra.get(spec_id as usize)
    }

    pub(crate) fn candidate_touched(&self) -> &[u32] {
        &self.candidate_touched
    }

    pub(crate) fn reset_candidates(&mut self) {
        for &spec_id in &self.candidate_touched {
            self.candidate_spectra.set(spec_id as usize, false);
        }
        self.candidate_touched.clear();
    }

    /// Ensures that block-pruning scratch space is sized for the current index.
    pub(crate) fn prepare_spectrum_block_scratch(&mut self, n_blocks: usize) {
        if self.block_upper_bounds.len() != n_blocks {
            self.block_upper_bounds.clear();
            self.block_upper_bounds.resize(n_blocks, 0.0);
            self.block_upper_bound_touched.clear();
            self.allowed_spectrum_blocks = SearchBitVec::zeros(n_blocks);
            self.allowed_block_touched.clear();
            return;
        }

        self.block_upper_bound_touched.clear();
        self.reset_allowed_spectrum_blocks();
    }

    /// Adds one query contribution to a block-level maximum possible score.
    #[inline]
    pub(crate) fn add_spectrum_block_upper_bound(&mut self, block_id: u32, contribution: f64) {
        if contribution <= 0.0 {
            return;
        }

        let block_index = block_id as usize;
        if self.block_upper_bounds[block_index] == 0.0 {
            self.block_upper_bound_touched.push(block_id);
        }
        self.block_upper_bounds[block_index] += contribution;
    }

    /// Marks blocks whose accumulated bound can still reach `minimum_score`.
    pub(crate) fn mark_allowed_spectrum_blocks(&mut self, minimum_score: f64) -> usize {
        const BOUND_EPSILON: f64 = 1e-12;

        let mut allowed_blocks = 0usize;
        for &block_id in &self.block_upper_bound_touched {
            let block_index = block_id as usize;
            if self.block_upper_bounds[block_index] + BOUND_EPSILON >= minimum_score
                && !self.allowed_spectrum_blocks.get(block_index)
            {
                self.allowed_spectrum_blocks.set(block_index, true);
                self.allowed_block_touched.push(block_id);
                allowed_blocks += 1;
            }
            self.block_upper_bounds[block_index] = 0.0;
        }
        self.block_upper_bound_touched.clear();
        allowed_blocks
    }

    /// Returns the number of spectrum blocks allowed by the current block filter.
    #[inline]
    pub(crate) fn n_allowed_spectrum_blocks(&self) -> usize {
        self.allowed_block_touched.len()
    }

    /// Returns the `index`th allowed spectrum block id.
    #[inline]
    pub(crate) fn allowed_spectrum_block_id(&self, index: usize) -> u32 {
        self.allowed_block_touched[index]
    }

    /// Clears the current query's allowed block bitset.
    pub(crate) fn reset_allowed_spectrum_blocks(&mut self) {
        for &block_id in &self.allowed_block_touched {
            self.allowed_spectrum_blocks.set(block_id as usize, false);
        }
        self.allowed_block_touched.clear();
    }
}

// ---------------------------------------------------------------------------
// FlashIndex<K> — the inverted m/z index
// ---------------------------------------------------------------------------

/// Inverted m/z index shared by all Flash search variants.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct FlashIndex<K: FlashKernel, P: SpectrumFloat = f64> {
    // Product ion index (sorted by m/z).
    pub(crate) product_mz: Vec<P>,
    product_spec_id: Vec<u32>,
    pub(crate) product_data: Vec<P>,

    // Product peaks in per-spectrum order for exact candidate rescoring.
    spectrum_offsets: Vec<u32>,
    spectrum_mz: Vec<P>,
    spectrum_data: Vec<P>,

    // Neutral loss index (sorted by neutral loss value).
    nl_value: Vec<P>,
    nl_spec_id: Vec<u32>,
    nl_data: Vec<P>,
    /// Maps neutral-loss entry → product entry for anti-double-counting.
    nl_to_product: Vec<u32>,

    // Per-spectrum metadata.
    spectrum_meta: Vec<K::SpectrumMeta>,
    spectrum_id_map: SpectrumIdMap,
    pub(crate) n_spectra: u32,

    // Config.
    pub(crate) tolerance: f64,
}

impl<K: FlashKernel, P: SpectrumFloat + Sync> FlashIndex<K, P> {
    /// Build the index from prepared spectrum data.
    ///
    /// Each element of `spectra` is `(precursor_mz, mz_values, peak_data_values)`.
    /// The caller is responsible for preparing peak data via the kernel's
    /// scoring function and validating well-separated preconditions.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityComputationError::IndexOverflow`] if the number of
    /// spectra or total peaks exceeds `u32::MAX`.
    pub(crate) fn build(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
    ) -> Result<Self, SimilarityComputationError> {
        Self::build_with_sort(
            tolerance,
            spectra,
            SpectrumIdMap::identity(),
            SortBackend::Sequential,
        )
    }

    pub(crate) fn build_with_spectrum_id_map(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
        spectrum_id_map: SpectrumIdMap,
    ) -> Result<Self, SimilarityComputationError> {
        Self::build_with_sort(tolerance, spectra, spectrum_id_map, SortBackend::Sequential)
    }

    /// Build the index from prepared spectrum data using Rayon-backed sorting.
    #[cfg(feature = "rayon")]
    pub(crate) fn build_parallel(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
    ) -> Result<Self, SimilarityComputationError> {
        Self::build_with_sort(
            tolerance,
            spectra,
            SpectrumIdMap::identity(),
            SortBackend::Parallel,
        )
    }

    #[cfg(all(feature = "rayon", feature = "experimental_reordered_index"))]
    pub(crate) fn build_parallel_with_spectrum_id_map(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
        spectrum_id_map: SpectrumIdMap,
    ) -> Result<Self, SimilarityComputationError> {
        Self::build_with_sort(tolerance, spectra, spectrum_id_map, SortBackend::Parallel)
    }

    fn build_with_sort(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
        spectrum_id_map: SpectrumIdMap,
        sort_backend: SortBackend,
    ) -> Result<Self, SimilarityComputationError> {
        let n_spectra: u32 =
            u32::try_from(spectra.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let total_peaks: usize = spectra.iter().map(|spectrum| spectrum.mz.len()).sum();
        let _: u32 =
            u32::try_from(total_peaks).map_err(|_| SimilarityComputationError::IndexOverflow)?;

        let mut spectrum_meta_vec: Vec<K::SpectrumMeta> = Vec::with_capacity(spectra.len());

        // Collect sort-key indices. We sort a permutation array by m/z rather
        // than sorting the full (u32, PeakEntry) tuples to reduce sort
        // bandwidth.
        let mut peak_mz_flat: Vec<P> = Vec::with_capacity(total_peaks);
        let mut peak_spec_id_flat: Vec<u32> = Vec::with_capacity(total_peaks);
        let mut peak_data_flat: Vec<P> = Vec::with_capacity(total_peaks);
        let mut peak_nl_flat: Vec<P> = Vec::with_capacity(total_peaks);
        let mut spectrum_offsets: Vec<u32> = Vec::with_capacity(spectra.len() + 1);
        let mut spectrum_mz: Vec<P> = Vec::with_capacity(total_peaks);
        let mut spectrum_data: Vec<P> = Vec::with_capacity(total_peaks);

        for (spec_id, spectrum) in spectra.iter().enumerate() {
            let spec_id = spec_id as u32; // safe: checked above
            spectrum_meta_vec.push(K::spectrum_meta(&spectrum.data));
            spectrum_offsets.push(spectrum_mz.len() as u32);

            for (&mz, &data) in spectrum.mz.iter().zip(spectrum.data.iter()) {
                peak_mz_flat.push(mz);
                peak_spec_id_flat.push(spec_id);
                peak_data_flat.push(data);
                peak_nl_flat.push(convert_flash_value(
                    spectrum.precursor_mz.to_f64() - mz.to_f64(),
                    "neutral_loss",
                )?);
                spectrum_mz.push(mz);
                spectrum_data.push(data);
            }
        }
        spectrum_offsets.push(spectrum_mz.len() as u32);

        // Build a permutation array and sort it by m/z.
        let mut product_perm: Vec<u32> = (0..total_peaks as u32).collect();
        sort_permutation_by_values(&mut product_perm, &peak_mz_flat, sort_backend);

        // Build old insertion index → new sorted index mapping.
        let mut old_to_new = alloc::vec![0u32; total_peaks];
        for (new_idx, &old_idx) in product_perm.iter().enumerate() {
            old_to_new[old_idx as usize] = new_idx as u32;
        }

        // Scatter into sorted product arrays.
        let mut product_mz = Vec::with_capacity(total_peaks);
        let mut product_spec_id = Vec::with_capacity(total_peaks);
        let mut product_data = Vec::with_capacity(total_peaks);
        for &old_idx in &product_perm {
            let i = old_idx as usize;
            product_mz.push(peak_mz_flat[i]);
            product_spec_id.push(peak_spec_id_flat[i]);
            product_data.push(peak_data_flat[i]);
        }

        // Build a permutation for NL entries sorted by neutral loss value.
        let mut nl_perm: Vec<u32> = (0..total_peaks as u32).collect();
        sort_permutation_by_values(&mut nl_perm, &peak_nl_flat, sort_backend);

        // Scatter into sorted NL arrays, remapping product index.
        let mut nl_value = Vec::with_capacity(total_peaks);
        let mut nl_spec_id = Vec::with_capacity(total_peaks);
        let mut nl_data = Vec::with_capacity(total_peaks);
        let mut nl_to_product = Vec::with_capacity(total_peaks);
        for &old_idx in &nl_perm {
            let i = old_idx as usize;
            nl_value.push(peak_nl_flat[i]);
            nl_spec_id.push(peak_spec_id_flat[i]);
            nl_data.push(peak_data_flat[i]);
            nl_to_product.push(old_to_new[i]);
        }

        Ok(FlashIndex {
            product_mz,
            product_spec_id,
            product_data,
            spectrum_offsets,
            spectrum_mz,
            spectrum_data,
            nl_value,
            nl_spec_id,
            nl_data,
            nl_to_product,
            spectrum_meta: spectrum_meta_vec,
            spectrum_id_map,
            n_spectra,
            tolerance,
        })
    }

    /// Create a [`SearchState`] sized for this index, suitable for reuse
    /// across multiple queries.
    pub(crate) fn new_search_state(&self) -> SearchState {
        SearchState::new(self.n_spectra as usize, self.product_mz.len())
    }

    pub(crate) fn spectrum_slices(&self, spec_id: u32) -> (&[P], &[P]) {
        let offset_start = self.spectrum_offsets[spec_id as usize] as usize;
        let offset_end = self.spectrum_offsets[spec_id as usize + 1] as usize;
        (
            &self.spectrum_mz[offset_start..offset_end],
            &self.spectrum_data[offset_start..offset_end],
        )
    }

    pub(crate) fn spectrum_meta(&self, spec_id: u32) -> &K::SpectrumMeta {
        &self.spectrum_meta[spec_id as usize]
    }

    #[inline]
    pub(crate) fn public_spectrum_id(&self, internal_id: u32) -> u32 {
        self.spectrum_id_map.internal_to_public(internal_id)
    }

    #[inline]
    pub(crate) fn internal_spectrum_id(
        &self,
        public_id: u32,
    ) -> Result<u32, SimilarityComputationError> {
        self.spectrum_id_map
            .public_to_internal(public_id)
            .filter(|&internal_id| internal_id < self.n_spectra)
            .ok_or(SimilarityComputationError::IndexOverflow)
    }

    pub(crate) fn for_each_product_spectrum_in_window<Q: SpectrumFloat>(
        &self,
        mz: Q,
        mut emit: impl FnMut(u32),
    ) -> usize {
        let mz = mz.to_f64();
        let lo = mz - self.tolerance;
        let hi = mz + self.tolerance;
        let start = self.product_mz.partition_point(|&v| v.to_f64() < lo);

        let mut visited = 0usize;
        for idx in start..self.product_mz.len() {
            let product_mz = self.product_mz[idx].to_f64();
            if product_mz > hi {
                break;
            }
            visited += 1;
            if product_mz < mz - self.tolerance || product_mz > mz + self.tolerance {
                continue;
            }
            emit(self.product_spec_id[idx]);
        }
        visited
    }

    pub(crate) fn mark_candidates_from_query_prefix_indices(
        &self,
        query_mz: &[impl SpectrumFloat],
        query_prefix_indices: &[usize],
        state: &mut SearchState,
    ) {
        state.ensure_candidate_capacity(self.n_spectra as usize);
        for &query_index in query_prefix_indices {
            let visited =
                self.for_each_product_spectrum_in_window(query_mz[query_index], |spec_id| {
                    state.mark_candidate(spec_id);
                });
            state.add_product_postings_visited(visited);
        }
    }

    pub(crate) fn mark_candidates_from_query_order_prefix(
        &self,
        query_mz: &[impl SpectrumFloat],
        prefix_len: usize,
        state: &mut SearchState,
    ) {
        state.ensure_candidate_capacity(self.n_spectra as usize);
        let prefix_len = prefix_len.min(state.query_order.len());
        for order_position in 0..prefix_len {
            let query_index = state.query_order[order_position];
            let visited =
                self.for_each_product_spectrum_in_window(query_mz[query_index], |spec_id| {
                    state.mark_candidate(spec_id);
                });
            state.add_product_postings_visited(visited);
        }
    }

    pub(crate) fn for_each_allowed_block_raw_score<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        block_products: &SpectrumBlockProductIndex<P>,
        state: &mut SearchState,
        mut emit: impl FnMut(u32, f64, usize),
    ) {
        state.acc.ensure_capacity(self.n_spectra as usize);

        let mut product_postings_visited = 0usize;
        for (query_index, &mz) in query_mz.iter().enumerate() {
            for block_index in 0..state.n_allowed_spectrum_blocks() {
                let block_id = state.allowed_spectrum_block_id(block_index);
                product_postings_visited = product_postings_visited.saturating_add(
                    block_products.for_each_peak_in_window(
                        block_id,
                        mz,
                        self.tolerance,
                        |spec_id, library_data| {
                            let score = K::pair_score(
                                query_data[query_index].to_f64(),
                                library_data.to_f64(),
                            );
                            if score != 0.0 {
                                state.acc.accumulate(spec_id, score);
                            }
                        },
                    ),
                );
            }
        }

        state.add_product_postings_visited(product_postings_visited);
        state.add_candidates_marked(state.acc.touched_len());
        state.acc.drain(|spec_id, raw, count| {
            emit(spec_id, raw, count as usize);
        });
        state.reset_allowed_spectrum_blocks();
    }

    /// Visit candidates from the currently allowed spectrum blocks in query
    /// suffix-bound order.
    ///
    /// The caller must prepare `state.query_order` and `state.query_suffix_bound`
    /// in the metric's bound units before calling this method. Each spectrum is
    /// passed to `visit_candidate` at most once, where the caller can exact-score
    /// it and return the updated suffix stop bound derived from the current top-k
    /// floor. Once the remaining query suffix falls below that bound, unseen
    /// spectra cannot enter the top-k set and the block scan stops.
    pub(crate) fn score_allowed_block_candidates_by_query_order<Q, Visit>(
        &self,
        query_mz: &[Q],
        block_products: &SpectrumBlockProductIndex<P>,
        state: &mut SearchState,
        mut suffix_stop_bound: f64,
        mut visit_candidate: Visit,
    ) where
        Q: SpectrumFloat,
        Visit: FnMut(u32) -> f64,
    {
        state.ensure_candidate_capacity(self.n_spectra as usize);

        let mut product_postings_visited = 0usize;
        let mut candidates_rescored = 0usize;
        let query_order_len = state.query_order().len();
        for order_position in 0..query_order_len {
            let query_index = state.query_order()[order_position];
            let query_mz = query_mz[query_index];

            for block_index in 0..state.n_allowed_spectrum_blocks() {
                let block_id = state.allowed_spectrum_block_id(block_index);
                product_postings_visited = product_postings_visited.saturating_add(
                    block_products.for_each_peak_in_window(
                        block_id,
                        query_mz,
                        self.tolerance,
                        |spec_id, _| {
                            if state.is_candidate(spec_id) {
                                return;
                            }
                            state.mark_candidate(spec_id);
                            candidates_rescored = candidates_rescored.saturating_add(1);
                            suffix_stop_bound = visit_candidate(spec_id);
                        },
                    ),
                );
            }

            if state.query_suffix_bound_at(order_position + 1) < suffix_stop_bound {
                break;
            }
        }

        state.add_product_postings_visited(product_postings_visited);
        state.add_candidates_rescored(candidates_rescored);
        state.reset_candidates();
        state.reset_allowed_spectrum_blocks();
    }

    /// Direct (unshifted) search: for each query peak, binary-search the
    /// product index and accumulate scores.
    pub(crate) fn search_direct<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
    ) -> Vec<FlashSearchResult> {
        let mut state = self.new_search_state();
        self.search_direct_with_state(query_mz, query_data, query_meta, &mut state)
    }

    /// Direct search using a caller-provided [`SearchState`] to avoid
    /// per-query allocation.
    pub(crate) fn search_direct_with_state<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        state: &mut SearchState,
    ) -> Vec<FlashSearchResult> {
        let mut results = Vec::new();
        self.for_each_direct_with_state(query_mz, query_data, query_meta, state, |result| {
            results.push(result);
        });
        results
    }

    pub(crate) fn for_each_direct_with_state<Q: SpectrumFloat, Emit>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        state: &mut SearchState,
        mut emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        state.reset_diagnostics();
        if self.n_spectra == 0 || query_mz.is_empty() {
            return;
        }

        state.acc.ensure_capacity(self.n_spectra as usize);
        let mut product_postings_visited = 0usize;

        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            let qmz = qmz.to_f64();
            let lo = qmz - self.tolerance;
            let hi = qmz + self.tolerance;

            // Binary search for the start of the tolerance window.
            let start = self.product_mz.partition_point(|&v| v.to_f64() < lo);

            for idx in start..self.product_mz.len() {
                let product_mz = self.product_mz[idx].to_f64();
                if product_mz > hi {
                    break;
                }
                product_postings_visited = product_postings_visited.saturating_add(1);
                // Guard against FP inconsistency between window arithmetic
                // (qmz ± tol) and the canonical |diff| ≤ tol check used by
                // the linear oracle.
                if product_mz < qmz - self.tolerance || product_mz > qmz + self.tolerance {
                    continue;
                }
                let score =
                    K::pair_score(query_data[q_idx].to_f64(), self.product_data[idx].to_f64());
                state.acc.accumulate(self.product_spec_id[idx], score);
            }
        }

        state.add_product_postings_visited(product_postings_visited);
        let mut results_emitted = 0usize;
        state.acc.drain(|spec_id, raw, count| {
            let score = K::finalize(
                raw,
                count as usize,
                query_meta,
                &self.spectrum_meta[spec_id as usize],
            );
            if score > 0.0 {
                results_emitted = results_emitted.saturating_add(1);
                emit(FlashSearchResult {
                    spectrum_id: self.public_spectrum_id(spec_id),
                    score,
                    n_matches: count as usize,
                });
            }
        });
        state.add_results_emitted(results_emitted);
    }

    pub(crate) fn direct_score_for_spectrum<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        spec_id: u32,
    ) -> (f64, usize) {
        let offset_start = self.spectrum_offsets[spec_id as usize] as usize;
        let offset_end = self.spectrum_offsets[spec_id as usize + 1] as usize;
        let library_mz = &self.spectrum_mz[offset_start..offset_end];
        let library_data = &self.spectrum_data[offset_start..offset_end];

        let mut raw = 0.0_f64;
        let mut n_matches = 0usize;
        let mut library_index = 0usize;

        for (query_index, &qmz) in query_mz.iter().enumerate() {
            let qmz = qmz.to_f64();
            while library_index < library_mz.len()
                && library_mz[library_index].to_f64() < qmz - self.tolerance
            {
                library_index += 1;
            }
            if library_index < library_mz.len()
                && library_mz[library_index].to_f64() >= qmz - self.tolerance
                && library_mz[library_index].to_f64() <= qmz + self.tolerance
            {
                let score = K::pair_score(
                    query_data[query_index].to_f64(),
                    library_data[library_index].to_f64(),
                );
                if score != 0.0 {
                    raw += score;
                    n_matches += 1;
                }
                library_index += 1;
            }
        }

        (raw, n_matches)
    }

    pub(crate) fn emit_exact_primary_candidates<Q, Emit, TargetRaw, LibraryBound>(
        &self,
        search: DirectThresholdSearch<'_, K, Q>,
        state: &mut SearchState,
        emit: &mut Emit,
        target_raw_score: &mut TargetRaw,
        library_bound: &mut LibraryBound,
    ) where
        Q: SpectrumFloat,
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        let mut candidates_rescored = 0usize;
        let mut results_emitted = 0usize;
        for &spec_id in state.candidate_touched() {
            candidates_rescored = candidates_rescored.saturating_add(1);
            if self.emit_exact_threshold_candidate(
                &search,
                spec_id,
                emit,
                target_raw_score,
                library_bound,
            ) {
                results_emitted = results_emitted.saturating_add(1);
            }
        }

        state.add_candidates_rescored(candidates_rescored);
        state.add_results_emitted(results_emitted);
        state.reset_candidates();
    }

    fn emit_exact_threshold_candidate<Q, Emit, TargetRaw, LibraryBound>(
        &self,
        search: &DirectThresholdSearch<'_, K, Q>,
        spec_id: u32,
        emit: &mut Emit,
        target_raw_score: &mut TargetRaw,
        library_bound: &mut LibraryBound,
    ) -> bool
    where
        Q: SpectrumFloat,
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        let lib_meta = &self.spectrum_meta[spec_id as usize];
        if library_bound(lib_meta) == 0.0 {
            return false;
        }

        let (raw, count) =
            self.direct_score_for_spectrum(search.query_mz, search.query_data, spec_id);
        if raw < target_raw_score(lib_meta) {
            return false;
        }

        let score = K::finalize(raw, count, search.query_meta, lib_meta);
        if score > 0.0 && score >= search.score_threshold {
            emit(FlashSearchResult {
                spectrum_id: self.public_spectrum_id(spec_id),
                score,
                n_matches: count,
            });
            return true;
        }
        false
    }

    pub(crate) fn for_each_direct_threshold_with_state<Q, Emit, TargetRaw, LibraryBound>(
        &self,
        search: DirectThresholdSearch<'_, K, Q>,
        state: &mut SearchState,
        mut emit: Emit,
        mut target_raw_score: TargetRaw,
        mut library_bound: LibraryBound,
    ) where
        Q: SpectrumFloat,
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        state.reset_diagnostics();
        if self.n_spectra == 0 || search.query_mz.is_empty() {
            return;
        }

        if search.score_threshold < PREFIX_PRUNING_MIN_THRESHOLD {
            self.for_each_direct_with_state(
                search.query_mz,
                search.query_data,
                search.query_meta,
                state,
                |result| {
                    if result.score >= search.score_threshold {
                        emit(result);
                    }
                },
            );
            return;
        }

        state.prepare_threshold_order(search.query_data);
        let target_query_norm = state.query_suffix_bound[0] * search.score_threshold;
        let prefix_len = state
            .query_suffix_bound
            .iter()
            .position(|&remaining_norm| remaining_norm < target_query_norm)
            .unwrap_or(state.query_order.len());

        if prefix_len * 2 > state.query_order.len() {
            self.for_each_direct_with_state(
                search.query_mz,
                search.query_data,
                search.query_meta,
                state,
                |result| {
                    if result.score >= search.score_threshold {
                        emit(result);
                    }
                },
            );
            return;
        }

        self.mark_candidates_from_query_order_prefix(search.query_mz, prefix_len, state);
        self.emit_exact_primary_candidates(
            search,
            state,
            &mut emit,
            &mut target_raw_score,
            &mut library_bound,
        );
    }

    /// Modified (direct + shifted) search with anti-double-counting.
    ///
    /// Phase 1: direct matches (same as direct search), marking matched
    /// product entries in a bit set.
    ///
    /// Phase 2: for each query peak, compute `neutral_loss = query_precursor -
    /// query_mz`, binary-search the neutral-loss index, skip entries whose
    /// product counterpart was already matched in phase 1, accumulate the rest.
    pub(crate) fn search_modified<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: f64,
    ) -> Vec<FlashSearchResult> {
        let mut state = self.new_search_state();
        self.search_modified_with_state(
            query_mz,
            query_data,
            query_meta,
            query_precursor_mz,
            &mut state,
        )
    }

    /// Modified search using a caller-provided [`SearchState`] to avoid
    /// per-query allocation.
    pub(crate) fn search_modified_with_state<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: f64,
        state: &mut SearchState,
    ) -> Vec<FlashSearchResult> {
        state.reset_diagnostics();
        if self.n_spectra == 0 || query_mz.is_empty() {
            return Vec::new();
        }

        state.acc.ensure_capacity(self.n_spectra as usize);
        state.ensure_modified_capacity(self.product_mz.len());

        // Destructure state to avoid borrow conflicts between fields.
        let SearchState {
            acc,
            matched_products,
            direct_scores,
            ..
        } = state;

        // Track which product indices we set so we can reset them efficiently.
        let mut set_indices: Vec<usize> = Vec::new();

        // Phase 1: direct matches.
        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            let qmz = qmz.to_f64();
            let lo = qmz - self.tolerance;
            let hi = qmz + self.tolerance;
            let start = self.product_mz.partition_point(|&v| v.to_f64() < lo);

            let mut idx = start;
            while idx < self.product_mz.len() {
                let product_mz = self.product_mz[idx].to_f64();
                if product_mz > hi {
                    break;
                }
                if product_mz < qmz - self.tolerance || product_mz > qmz + self.tolerance {
                    idx += 1;
                    continue;
                }
                let score =
                    K::pair_score(query_data[q_idx].to_f64(), self.product_data[idx].to_f64());
                acc.accumulate(self.product_spec_id[idx], score);
                matched_products.set(idx, true);
                direct_scores[idx] = score;
                set_indices.push(idx);
                idx += 1;
            }
        }

        // Phase 2: shifted (neutral loss) matches.
        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            let query_nl = query_precursor_mz - qmz.to_f64();
            let lo = query_nl - self.tolerance;
            let hi = query_nl + self.tolerance;
            let start = self.nl_value.partition_point(|&v| v.to_f64() < lo);

            for idx in start..self.nl_value.len() {
                let nl_value = self.nl_value[idx].to_f64();
                if nl_value > hi {
                    break;
                }
                if nl_value < query_nl - self.tolerance || nl_value > query_nl + self.tolerance {
                    continue;
                }
                let product_idx = self.nl_to_product[idx] as usize;
                let nl_score =
                    K::pair_score(query_data[q_idx].to_f64(), self.nl_data[idx].to_f64());
                if matched_products.get(product_idx) {
                    // Library peak already matched directly. Upgrade if NL
                    // gives a better pair score.
                    if nl_score > direct_scores[product_idx] {
                        acc.upgrade(self.nl_spec_id[idx], direct_scores[product_idx], nl_score);
                    }
                    continue;
                }
                acc.accumulate(self.nl_spec_id[idx], nl_score);
            }
        }

        // Reset the bitvec and direct scores for reuse.
        for &idx in &set_indices {
            matched_products.set(idx, false);
            direct_scores[idx] = 0.0;
        }

        let mut results = Vec::with_capacity(acc.touched.len());
        acc.drain(|spec_id, raw, count| {
            let score = K::finalize(
                raw,
                count as usize,
                query_meta,
                &self.spectrum_meta[spec_id as usize],
            );
            if score > 0.0 {
                results.push(FlashSearchResult {
                    spectrum_id: self.public_spectrum_id(spec_id),
                    score,
                    n_matches: count as usize,
                });
            }
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;

    #[derive(Clone, Copy, Default)]
    struct TestKernel;

    impl FlashKernel for TestKernel {
        type SpectrumMeta = ();

        fn spectrum_meta<P: SpectrumFloat>(_peak_data: &[P]) -> Self::SpectrumMeta {}

        fn pair_score(query: f64, library: f64) -> f64 {
            query * library
        }

        fn finalize(
            raw: f64,
            _n_matches: usize,
            _query_meta: &Self::SpectrumMeta,
            _lib_meta: &Self::SpectrumMeta,
        ) -> f64 {
            raw
        }
    }

    fn prepared(precursor_mz: f64, mz: Vec<f64>, data: Vec<f64>) -> PreparedFlashSpectrum<f64> {
        PreparedFlashSpectrum {
            precursor_mz,
            mz,
            data,
        }
    }

    fn build_test_index(
        spectra: PreparedFlashSpectra<f64>,
        tolerance: f64,
    ) -> FlashIndex<TestKernel> {
        FlashIndex::<TestKernel>::build(tolerance, spectra).expect("test index should build")
    }

    #[test]
    fn dense_accumulator_drains_and_resets_touched_slots() {
        let mut acc = DenseAccumulator::new();
        acc.ensure_capacity(4);
        acc.accumulate(2, 1.5);
        acc.accumulate(2, 0.5);
        acc.accumulate(1, 2.0);

        let mut emitted = Vec::new();
        acc.drain(|id, score, count| emitted.push((id, score, count)));

        emitted.sort_by_key(|&(id, _, _)| id);
        assert_eq!(emitted, vec![(1, 2.0, 1), (2, 2.0, 2)]);
        assert!(acc.touched.is_empty());
        assert_eq!(acc.scores, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(acc.counts, vec![0, 0, 0, 0]);

        let mut second = Vec::new();
        acc.drain(|id, score, count| second.push((id, score, count)));
        assert!(second.is_empty());
    }

    #[test]
    fn dense_accumulator_upgrade_replaces_score_without_changing_count() {
        let mut acc = DenseAccumulator::new();
        acc.ensure_capacity(2);
        acc.accumulate(0, 1.0);
        acc.upgrade(0, 1.0, 3.5);

        let mut emitted = Vec::new();
        acc.drain(|id, score, count| emitted.push((id, score, count)));
        assert_eq!(emitted, vec![(0, 3.5, 1)]);
    }

    #[test]
    fn build_sorts_products_and_lazily_initializes_search_state() {
        let index = build_test_index(
            vec![
                prepared(210.0, vec![110.0, 100.0], vec![4.0, 1.0]),
                prepared(205.0, vec![90.0], vec![2.0]),
            ],
            0.1,
        );

        assert_eq!(index.product_mz, vec![90.0, 100.0, 110.0]);
        assert_eq!(index.product_data, vec![2.0, 1.0, 4.0]);
        assert_eq!(index.n_spectra, 2);

        let state = index.new_search_state();
        assert!(state.matched_products.is_empty());
        assert!(state.direct_scores.is_empty());
        assert!(state.acc.scores.is_empty());

        let mut direct_state = index.new_search_state();
        let _ = index.search_direct_with_state(&[100.0], &[1.0], &(), &mut direct_state);
        assert_eq!(direct_state.acc.scores.len(), 2);
        assert!(direct_state.matched_products.is_empty());
        assert!(direct_state.direct_scores.is_empty());

        let mut modified_state = index.new_search_state();
        let _ = index.search_modified_with_state(&[100.0], &[1.0], &(), 200.0, &mut modified_state);
        assert_eq!(modified_state.acc.scores.len(), 2);
        assert_eq!(modified_state.matched_products.len(), 3);
        assert_eq!(modified_state.direct_scores.len(), 3);
    }

    #[test]
    fn l2_threshold_prefix_indices_stop_when_suffix_bound_is_below_threshold() {
        let prefix = l2_threshold_prefix_indices(&[1.0, 4.0, 0.5], 0.9);
        assert_eq!(prefix, vec![1]);

        let all = l2_threshold_prefix_indices(&[1.0, 4.0, 0.5], 0.0);
        assert_eq!(all, vec![1, 0, 2]);

        let empty = l2_threshold_prefix_indices(&[1.0, 4.0, 0.5], 1.1);
        assert!(empty.is_empty());
    }

    #[test]
    fn threshold_prefix_postings_store_sorted_per_spectrum_prefixes() {
        let spectra = vec![
            prepared(200.0, vec![100.0, 150.0], vec![4.0, 1.0]),
            prepared(300.0, vec![100.05, 250.0], vec![1.0, 5.0]),
        ];
        let prefixes =
            ThresholdPrefixPostings::build(&spectra, |data| l2_threshold_prefix_indices(data, 0.9))
                .expect("prefix postings should build");
        assert_eq!(prefixes.n_prefix_peaks(), 2);
        assert_eq!(prefixes.spectrum_prefix_mz(0), &[100.0]);
        assert_eq!(prefixes.spectrum_prefix_mz(1), &[250.0]);

        let sorted_prefixes = ThresholdPrefixPostings::build(
            &[prepared(200.0, vec![100.0, 150.0], vec![1.0, 4.0])],
            |_| vec![1, 0],
        )
        .expect("prefix postings should build");
        assert_eq!(sorted_prefixes.spectrum_prefix_mz(0), &[100.0, 150.0]);
    }

    #[test]
    fn spectrum_block_upper_bound_index_filters_candidate_blocks() {
        let spectra = vec![
            prepared(500.0, vec![100.0], vec![1.0]),
            prepared(500.0, vec![100.0], vec![0.8]),
            prepared(500.0, vec![100.0], vec![0.1]),
        ];
        let block_index = SpectrumBlockUpperBoundIndex::build(&spectra, 0.1, 2, |_, spectrum| {
            Ok(spectrum.data.iter().map(|&value| value.to_f64()).collect())
        })
        .expect("block index should build");
        let block_products = SpectrumBlockProductIndex::build(&spectra, 2)
            .expect("block product index should build");
        let index = build_test_index(spectra, 0.1);
        let mut state = index.new_search_state();

        block_index
            .prepare_allowed_blocks(&[100.0], 0.1, 0.5, &mut state, |_, max_value| max_value);
        let mut scored = Vec::new();
        index.for_each_allowed_block_raw_score(
            &[100.0],
            &[2.0],
            &block_products,
            &mut state,
            |spec_id, raw, count| scored.push((spec_id, raw, count)),
        );

        scored.sort_by_key(|&(spec_id, _, _)| spec_id);
        assert_eq!(scored, vec![(0, 2.0, 1), (1, 1.6, 1)]);
        assert_eq!(state.diagnostics.spectrum_blocks_evaluated, 2);
        assert_eq!(state.diagnostics.spectrum_blocks_allowed, 1);
        assert_eq!(state.diagnostics.spectrum_blocks_pruned, 1);
        assert_eq!(state.diagnostics.product_postings_visited, 2);
        assert_eq!(state.diagnostics.candidates_marked, 2);
    }

    #[test]
    fn search_direct_filters_zero_scores_and_handles_empty_inputs() {
        let empty_index = build_test_index(vec![], 0.1);
        assert!(empty_index.search_direct(&[100.0], &[1.0], &()).is_empty());

        let zero_score_index = build_test_index(vec![prepared(200.0, vec![100.0], vec![0.0])], 0.1);
        let mut state = zero_score_index.new_search_state();
        let zero_results =
            zero_score_index.search_direct_with_state(&[100.0], &[5.0], &(), &mut state);
        assert!(zero_results.is_empty());

        let repeated_empty = zero_score_index.search_direct_with_state(
            &[] as &[f64],
            &[] as &[f64],
            &(),
            &mut state,
        );
        assert!(repeated_empty.is_empty());
    }

    #[test]
    fn modified_search_upgrades_direct_matches_and_reuses_state_cleanly() {
        let index = build_test_index(vec![prepared(200.0, vec![100.0], vec![1.0])], 0.1);
        let mut state = index.new_search_state();

        let upgraded =
            index.search_modified_with_state(&[100.0, 110.0], &[1.0, 5.0], &(), 210.0, &mut state);
        assert_eq!(
            upgraded,
            vec![FlashSearchResult {
                spectrum_id: 0,
                score: 5.0,
                n_matches: 1,
            }]
        );

        let shifted_only =
            index.search_modified_with_state(&[110.0], &[3.0], &(), 210.0, &mut state);
        assert_eq!(
            shifted_only,
            vec![FlashSearchResult {
                spectrum_id: 0,
                score: 3.0,
                n_matches: 1,
            }]
        );

        let no_match = index.search_modified_with_state(&[150.0], &[2.0], &(), 210.0, &mut state);
        assert!(no_match.is_empty());

        let repeated =
            index.search_modified_with_state(&[100.0, 110.0], &[1.0, 5.0], &(), 210.0, &mut state);
        assert_eq!(repeated, upgraded);
    }
}
