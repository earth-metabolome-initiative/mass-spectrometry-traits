//! Shared internals for the Flash inverted m/z index.
//!
//! All items are `pub(crate)` — individual variant modules (`flash_cosine_index`,
//! `flash_entropy_index`) expose only their public wrappers.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::{Deref, DerefMut};

use bitvec::prelude::*;

use super::similarity_errors::SimilarityComputationError;

const PREFIX_PRUNING_MIN_THRESHOLD: f64 = 0.85;

struct SearchBitVec(BitVec);

impl SearchBitVec {
    fn new() -> Self {
        Self(BitVec::new())
    }

    fn zeros(len: usize) -> Self {
        Self(bitvec![0; len])
    }

    #[cfg(feature = "mem_size")]
    fn stored_bytes(&self, flags: mem_dbg::SizeFlags) -> usize {
        let bits = if flags.contains(mem_dbg::SizeFlags::CAPACITY) {
            self.0.capacity()
        } else {
            self.0.len()
        };
        bits.div_ceil(usize::BITS as usize) * core::mem::size_of::<usize>()
    }
}

impl Deref for SearchBitVec {
    type Target = BitVec;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SearchBitVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
    fn spectrum_meta(peak_data: &[f64]) -> Self::SpectrumMeta;

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
/// let index = FlashCosineIndex::new(0.0, 1.0, 0.1, &spectra).unwrap();
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
        }
    }

    /// Insert a result if it is good enough for the current top-k set.
    #[inline]
    pub(crate) fn push(&mut self, result: FlashSearchResult) {
        if self.k == 0 || result.score < self.min_score {
            return;
        }
        if self.results.len() < self.k {
            self.results.push(result);
            return;
        }

        let Some((worst_index, worst_result)) = self
            .results
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| compare_search_results_by_rank(left, right))
        else {
            return;
        };

        if compare_search_results_by_rank(&result, worst_result).is_lt() {
            self.results[worst_index] = result;
        }
    }

    /// Return the score that a future candidate must reach to matter.
    pub(crate) fn pruning_score(&self) -> f64 {
        if self.results.len() < self.k {
            return self.min_score;
        }

        let Some(worst_result) = self
            .results
            .iter()
            .max_by(|left, right| compare_search_results_by_rank(left, right))
        else {
            return self.min_score;
        };
        worst_result.score.max(self.min_score)
    }

    /// Sort selected results into the public deterministic rank order.
    fn finish(&mut self) {
        self.results.sort_by(compare_search_results_by_rank);
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
pub(crate) struct DirectThresholdSearch<'a, K: FlashKernel> {
    pub(crate) query_mz: &'a [f64],
    pub(crate) query_data: &'a [f64],
    pub(crate) query_meta: &'a K::SpectrumMeta,
    pub(crate) score_threshold: f64,
}

/// Library-side prefix postings for threshold-specialized direct searches.
///
/// A threshold-aware wrapper chooses how each spectrum's prefix is computed,
/// then this shared structure stores those prefix peaks in a sorted m/z index
/// and in per-spectrum order for indexed-query graph construction.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct ThresholdPrefixPostings {
    prefix_mz: Vec<f64>,
    prefix_spec_id: Vec<u32>,
    spectrum_prefix_offsets: Vec<u32>,
    spectrum_prefix_mz: Vec<f64>,
}

impl ThresholdPrefixPostings {
    pub(crate) fn build(
        spectra: &[(f64, Vec<f64>, Vec<f64>)],
        mut prefix_indices: impl FnMut(&[f64]) -> Vec<usize>,
    ) -> Result<Self, SimilarityComputationError> {
        let mut prefix_entries: Vec<(f64, u32)> = Vec::new();
        let mut spectrum_prefix_offsets = Vec::with_capacity(spectra.len() + 1);
        let mut spectrum_prefix_mz = Vec::new();
        spectrum_prefix_offsets.push(0);

        for (spec_id, (_, mz_vals, data_vals)) in spectra.iter().enumerate() {
            let spec_id_u32 =
                u32::try_from(spec_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            for peak_index in prefix_indices(data_vals) {
                let mz = mz_vals[peak_index];
                prefix_entries.push((mz, spec_id_u32));
                spectrum_prefix_mz.push(mz);
            }
            let offset = u32::try_from(spectrum_prefix_mz.len())
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
            spectrum_prefix_offsets.push(offset);
        }

        prefix_entries.sort_unstable_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });

        let mut prefix_mz = Vec::with_capacity(prefix_entries.len());
        let mut prefix_spec_id = Vec::with_capacity(prefix_entries.len());
        for (mz, spec_id) in prefix_entries {
            prefix_mz.push(mz);
            prefix_spec_id.push(spec_id);
        }

        Ok(Self {
            prefix_mz,
            prefix_spec_id,
            spectrum_prefix_offsets,
            spectrum_prefix_mz,
        })
    }

    #[inline]
    pub(crate) fn n_prefix_peaks(&self) -> usize {
        self.prefix_mz.len()
    }

    pub(crate) fn spectrum_prefix_mz(&self, spec_id: u32) -> &[f64] {
        let prefix_start = self.spectrum_prefix_offsets[spec_id as usize] as usize;
        let prefix_end = self.spectrum_prefix_offsets[spec_id as usize + 1] as usize;
        &self.spectrum_prefix_mz[prefix_start..prefix_end]
    }

    fn for_each_spectrum_in_window(&self, mz: f64, tolerance: f64, mut emit: impl FnMut(u32)) {
        let lo = mz - tolerance;
        let hi = mz + tolerance;
        let start = self.prefix_mz.partition_point(|&value| value < lo);

        for idx in start..self.prefix_mz.len() {
            if self.prefix_mz[idx] > hi {
                break;
            }
            if self.prefix_mz[idx] < mz - tolerance || self.prefix_mz[idx] > mz + tolerance {
                continue;
            }

            emit(self.prefix_spec_id[idx]);
        }
    }
}

/// Prefix indices for score bounds that can be represented as an L2 suffix
/// norm over per-peak weights.
pub(crate) fn l2_threshold_prefix_indices(
    peak_weights: &[f64],
    score_threshold: f64,
) -> Vec<usize> {
    if peak_weights.is_empty() || score_threshold > 1.0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..peak_weights.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        peak_weights[right]
            .abs()
            .total_cmp(&peak_weights[left].abs())
            .then_with(|| left.cmp(&right))
    });

    if score_threshold <= 0.0 {
        return order;
    }

    let norm = peak_weights
        .iter()
        .map(|&value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Vec::new();
    }

    let mut suffix_norm = alloc::vec![0.0_f64; order.len() + 1];
    for order_index in (0..order.len()).rev() {
        let peak_index = order[order_index];
        suffix_norm[order_index] =
            suffix_norm[order_index + 1] + peak_weights[peak_index] * peak_weights[peak_index];
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
    secondary_candidate_spectra: SearchBitVec,
    secondary_candidate_touched: Vec<u32>,
    query_order: Vec<usize>,
    query_suffix_norm: Vec<f64>,
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
            secondary_candidate_spectra: SearchBitVec::new(),
            secondary_candidate_touched: Vec::new(),
            query_order: Vec::new(),
            query_suffix_norm: Vec::new(),
        }
    }

    pub(crate) fn ensure_candidate_capacity(&mut self, n_spectra: usize) {
        if self.candidate_spectra.len() == n_spectra {
            return;
        }

        self.candidate_spectra = SearchBitVec::zeros(n_spectra);
        self.candidate_touched.clear();
    }

    pub(crate) fn ensure_secondary_candidate_capacity(&mut self, n_spectra: usize) {
        if self.secondary_candidate_spectra.len() == n_spectra {
            return;
        }

        self.secondary_candidate_spectra = SearchBitVec::zeros(n_spectra);
        self.secondary_candidate_touched.clear();
    }

    fn ensure_modified_capacity(&mut self, n_products: usize) {
        if self.matched_products.len() == n_products && self.direct_scores.len() == n_products {
            return;
        }

        self.matched_products = SearchBitVec::zeros(n_products);
        self.direct_scores.clear();
        self.direct_scores.resize(n_products, 0.0);
    }

    pub(crate) fn prepare_threshold_order(&mut self, query_data: &[f64]) {
        self.query_order.clear();
        self.query_order.extend(0..query_data.len());
        self.query_order.sort_unstable_by(|&left, &right| {
            query_data[right]
                .abs()
                .total_cmp(&query_data[left].abs())
                .then_with(|| left.cmp(&right))
        });

        self.query_suffix_norm.clear();
        self.query_suffix_norm.resize(query_data.len() + 1, 0.0);
        for order_index in (0..self.query_order.len()).rev() {
            let query_index = self.query_order[order_index];
            self.query_suffix_norm[order_index] = self.query_suffix_norm[order_index + 1]
                + query_data[query_index] * query_data[query_index];
        }
        for value in &mut self.query_suffix_norm {
            *value = value.sqrt();
        }
    }

    pub(crate) fn threshold_prefix_len(&self, score_threshold: f64) -> usize {
        let Some(&query_norm) = self.query_suffix_norm.first() else {
            return 0;
        };
        let target_query_norm = query_norm * score_threshold;
        self.query_suffix_norm
            .iter()
            .position(|&remaining_norm| remaining_norm < target_query_norm)
            .unwrap_or(self.query_order.len())
    }

    pub(crate) fn query_order(&self) -> &[usize] {
        &self.query_order
    }

    pub(crate) fn query_suffix_norm_at(&self, index: usize) -> f64 {
        self.query_suffix_norm[index]
    }

    #[inline]
    pub(crate) fn mark_candidate(&mut self, spec_id: u32) {
        let idx = spec_id as usize;
        if !self.candidate_spectra[idx] {
            self.candidate_spectra.set(idx, true);
            self.candidate_touched.push(spec_id);
        }
    }

    #[inline]
    pub(crate) fn is_candidate(&self, spec_id: u32) -> bool {
        self.candidate_spectra[spec_id as usize]
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

    #[inline]
    pub(crate) fn mark_secondary_candidate(&mut self, spec_id: u32) {
        let idx = spec_id as usize;
        if !self.secondary_candidate_spectra[idx] {
            self.secondary_candidate_spectra.set(idx, true);
            self.secondary_candidate_touched.push(spec_id);
        }
    }

    pub(crate) fn secondary_candidate_touched(&self) -> &[u32] {
        &self.secondary_candidate_touched
    }

    pub(crate) fn reset_secondary_candidates(&mut self) {
        for &spec_id in &self.secondary_candidate_touched {
            self.secondary_candidate_spectra
                .set(spec_id as usize, false);
        }
        self.secondary_candidate_touched.clear();
    }
}

// ---------------------------------------------------------------------------
// FlashIndex<K> — the inverted m/z index
// ---------------------------------------------------------------------------

/// Inverted m/z index shared by all Flash search variants.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct FlashIndex<K: FlashKernel> {
    // Product ion index (sorted by m/z).
    pub(crate) product_mz: Vec<f64>,
    product_spec_id: Vec<u32>,
    pub(crate) product_data: Vec<f64>,

    // Product peaks in per-spectrum order for exact candidate rescoring.
    spectrum_offsets: Vec<u32>,
    spectrum_mz: Vec<f64>,
    spectrum_data: Vec<f64>,

    // Neutral loss index (sorted by neutral loss value).
    nl_value: Vec<f64>,
    nl_spec_id: Vec<u32>,
    nl_data: Vec<f64>,
    /// Maps neutral-loss entry → product entry for anti-double-counting.
    nl_to_product: Vec<u32>,

    // Per-spectrum metadata.
    spectrum_meta: Vec<K::SpectrumMeta>,
    pub(crate) n_spectra: u32,

    // Config.
    pub(crate) tolerance: f64,
}

impl<K: FlashKernel> FlashIndex<K> {
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
        spectra: Vec<(f64, Vec<f64>, Vec<f64>)>,
    ) -> Result<Self, SimilarityComputationError> {
        let n_spectra: u32 =
            u32::try_from(spectra.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let total_peaks: usize = spectra.iter().map(|(_, mz, _)| mz.len()).sum();
        let _: u32 =
            u32::try_from(total_peaks).map_err(|_| SimilarityComputationError::IndexOverflow)?;

        let mut spectrum_meta_vec: Vec<K::SpectrumMeta> = Vec::with_capacity(spectra.len());

        // Collect sort-key indices. We sort a permutation array by m/z rather
        // than sorting the full (u32, PeakEntry) tuples to reduce sort
        // bandwidth.
        let mut peak_mz_flat: Vec<f64> = Vec::with_capacity(total_peaks);
        let mut peak_spec_id_flat: Vec<u32> = Vec::with_capacity(total_peaks);
        let mut peak_data_flat: Vec<f64> = Vec::with_capacity(total_peaks);
        let mut peak_nl_flat: Vec<f64> = Vec::with_capacity(total_peaks);
        let mut spectrum_offsets: Vec<u32> = Vec::with_capacity(spectra.len() + 1);
        let mut spectrum_mz: Vec<f64> = Vec::with_capacity(total_peaks);
        let mut spectrum_data: Vec<f64> = Vec::with_capacity(total_peaks);

        for (spec_id, (prec_mz, mz_vals, data_vals)) in spectra.iter().enumerate() {
            let spec_id = spec_id as u32; // safe: checked above
            spectrum_meta_vec.push(K::spectrum_meta(data_vals));
            spectrum_offsets.push(spectrum_mz.len() as u32);

            for (&mz, &data) in mz_vals.iter().zip(data_vals.iter()) {
                peak_mz_flat.push(mz);
                peak_spec_id_flat.push(spec_id);
                peak_data_flat.push(data);
                peak_nl_flat.push(*prec_mz - mz);
                spectrum_mz.push(mz);
                spectrum_data.push(data);
            }
        }
        spectrum_offsets.push(spectrum_mz.len() as u32);

        // Build a permutation array and sort it by m/z.
        let mut product_perm: Vec<u32> = (0..total_peaks as u32).collect();
        product_perm.sort_unstable_by(|&a, &b| {
            peak_mz_flat[a as usize].total_cmp(&peak_mz_flat[b as usize])
        });

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
        nl_perm.sort_unstable_by(|&a, &b| {
            peak_nl_flat[a as usize].total_cmp(&peak_nl_flat[b as usize])
        });

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
            n_spectra,
            tolerance,
        })
    }

    /// Create a [`SearchState`] sized for this index, suitable for reuse
    /// across multiple queries.
    pub(crate) fn new_search_state(&self) -> SearchState {
        SearchState::new(self.n_spectra as usize, self.product_mz.len())
    }

    pub(crate) fn spectrum_slices(&self, spec_id: u32) -> (&[f64], &[f64]) {
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

    pub(crate) fn for_each_product_spectrum_in_window(&self, mz: f64, mut emit: impl FnMut(u32)) {
        let lo = mz - self.tolerance;
        let hi = mz + self.tolerance;
        let start = self.product_mz.partition_point(|&v| v < lo);

        for idx in start..self.product_mz.len() {
            if self.product_mz[idx] > hi {
                break;
            }
            if self.product_mz[idx] < mz - self.tolerance
                || self.product_mz[idx] > mz + self.tolerance
            {
                continue;
            }
            emit(self.product_spec_id[idx]);
        }
    }

    pub(crate) fn mark_candidates_from_query_prefix_indices(
        &self,
        query_mz: &[f64],
        query_prefix_indices: &[usize],
        state: &mut SearchState,
    ) {
        state.ensure_candidate_capacity(self.n_spectra as usize);
        for &query_index in query_prefix_indices {
            self.for_each_product_spectrum_in_window(query_mz[query_index], |spec_id| {
                state.mark_candidate(spec_id);
            });
        }
    }

    pub(crate) fn mark_candidates_from_query_prefix_mz(
        &self,
        query_prefix_mz: &[f64],
        state: &mut SearchState,
    ) {
        state.ensure_candidate_capacity(self.n_spectra as usize);
        for &mz in query_prefix_mz {
            self.for_each_product_spectrum_in_window(mz, |spec_id| {
                state.mark_candidate(spec_id);
            });
        }
    }

    pub(crate) fn intersect_candidates_with_library_prefixes(
        &self,
        query_mz: &[f64],
        library_prefixes: &ThresholdPrefixPostings,
        state: &mut SearchState,
    ) {
        state.ensure_secondary_candidate_capacity(self.n_spectra as usize);
        for &mz in query_mz {
            library_prefixes.for_each_spectrum_in_window(mz, self.tolerance, |spec_id| {
                if state.is_candidate(spec_id) {
                    state.mark_secondary_candidate(spec_id);
                }
            });
        }
    }

    /// Direct (unshifted) search: for each query peak, binary-search the
    /// product index and accumulate scores.
    pub(crate) fn search_direct(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &K::SpectrumMeta,
    ) -> Vec<FlashSearchResult> {
        let mut state = self.new_search_state();
        self.search_direct_with_state(query_mz, query_data, query_meta, &mut state)
    }

    /// Direct search using a caller-provided [`SearchState`] to avoid
    /// per-query allocation.
    pub(crate) fn search_direct_with_state(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &K::SpectrumMeta,
        state: &mut SearchState,
    ) -> Vec<FlashSearchResult> {
        let mut results = Vec::new();
        self.for_each_direct_with_state(query_mz, query_data, query_meta, state, |result| {
            results.push(result);
        });
        results
    }

    pub(crate) fn for_each_direct_with_state<Emit>(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &K::SpectrumMeta,
        state: &mut SearchState,
        mut emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        if self.n_spectra == 0 || query_mz.is_empty() {
            return;
        }

        state.acc.ensure_capacity(self.n_spectra as usize);
        let acc = &mut state.acc;

        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            let lo = qmz - self.tolerance;
            let hi = qmz + self.tolerance;

            // Binary search for the start of the tolerance window.
            let start = self.product_mz.partition_point(|&v| v < lo);

            for idx in start..self.product_mz.len() {
                if self.product_mz[idx] > hi {
                    break;
                }
                // Guard against FP inconsistency between window arithmetic
                // (qmz ± tol) and the canonical |diff| ≤ tol check used by
                // the linear oracle.
                if self.product_mz[idx] < qmz - self.tolerance
                    || self.product_mz[idx] > qmz + self.tolerance
                {
                    continue;
                }
                let score = K::pair_score(query_data[q_idx], self.product_data[idx]);
                acc.accumulate(self.product_spec_id[idx], score);
            }
        }

        acc.drain(|spec_id, raw, count| {
            let score = K::finalize(
                raw,
                count as usize,
                query_meta,
                &self.spectrum_meta[spec_id as usize],
            );
            if score > 0.0 {
                emit(FlashSearchResult {
                    spectrum_id: spec_id,
                    score,
                    n_matches: count as usize,
                });
            }
        });
    }

    pub(crate) fn direct_score_for_spectrum(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
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
            while library_index < library_mz.len()
                && library_mz[library_index] < qmz - self.tolerance
            {
                library_index += 1;
            }
            if library_index < library_mz.len()
                && library_mz[library_index] >= qmz - self.tolerance
                && library_mz[library_index] <= qmz + self.tolerance
            {
                let score = K::pair_score(query_data[query_index], library_data[library_index]);
                if score != 0.0 {
                    raw += score;
                    n_matches += 1;
                }
                library_index += 1;
            }
        }

        (raw, n_matches)
    }

    pub(crate) fn emit_exact_primary_candidates<Emit, TargetRaw, LibraryBound>(
        &self,
        search: DirectThresholdSearch<'_, K>,
        state: &mut SearchState,
        emit: &mut Emit,
        target_raw_score: &mut TargetRaw,
        library_bound: &mut LibraryBound,
    ) where
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        for &spec_id in state.candidate_touched() {
            self.emit_exact_threshold_candidate(
                &search,
                spec_id,
                emit,
                target_raw_score,
                library_bound,
            );
        }

        state.reset_candidates();
    }

    pub(crate) fn emit_exact_secondary_candidates<Emit, TargetRaw, LibraryBound>(
        &self,
        search: DirectThresholdSearch<'_, K>,
        state: &mut SearchState,
        emit: &mut Emit,
        target_raw_score: &mut TargetRaw,
        library_bound: &mut LibraryBound,
    ) where
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        for &spec_id in state.secondary_candidate_touched() {
            self.emit_exact_threshold_candidate(
                &search,
                spec_id,
                emit,
                target_raw_score,
                library_bound,
            );
        }

        state.reset_secondary_candidates();
        state.reset_candidates();
    }

    fn emit_exact_threshold_candidate<Emit, TargetRaw, LibraryBound>(
        &self,
        search: &DirectThresholdSearch<'_, K>,
        spec_id: u32,
        emit: &mut Emit,
        target_raw_score: &mut TargetRaw,
        library_bound: &mut LibraryBound,
    ) where
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        let lib_meta = &self.spectrum_meta[spec_id as usize];
        if library_bound(lib_meta) == 0.0 {
            return;
        }

        let (raw, count) =
            self.direct_score_for_spectrum(search.query_mz, search.query_data, spec_id);
        if raw < target_raw_score(lib_meta) {
            return;
        }

        let score = K::finalize(raw, count, search.query_meta, lib_meta);
        if score > 0.0 && score >= search.score_threshold {
            emit(FlashSearchResult {
                spectrum_id: spec_id,
                score,
                n_matches: count,
            });
        }
    }

    pub(crate) fn for_each_direct_threshold_with_state<Emit, TargetRaw, LibraryBound>(
        &self,
        search: DirectThresholdSearch<'_, K>,
        state: &mut SearchState,
        mut emit: Emit,
        mut target_raw_score: TargetRaw,
        mut library_bound: LibraryBound,
    ) where
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
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
        let target_query_norm = state.query_suffix_norm[0] * search.score_threshold;
        let prefix_len = state
            .query_suffix_norm
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

        let query_prefix_indices: Vec<usize> = state.query_order[..prefix_len].to_vec();
        self.mark_candidates_from_query_prefix_indices(
            search.query_mz,
            &query_prefix_indices,
            state,
        );
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
    pub(crate) fn search_modified(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
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
    pub(crate) fn search_modified_with_state(
        &self,
        query_mz: &[f64],
        query_data: &[f64],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: f64,
        state: &mut SearchState,
    ) -> Vec<FlashSearchResult> {
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
            let lo = qmz - self.tolerance;
            let hi = qmz + self.tolerance;
            let start = self.product_mz.partition_point(|&v| v < lo);

            let mut idx = start;
            while idx < self.product_mz.len() {
                if self.product_mz[idx] > hi {
                    break;
                }
                if self.product_mz[idx] < qmz - self.tolerance
                    || self.product_mz[idx] > qmz + self.tolerance
                {
                    idx += 1;
                    continue;
                }
                let score = K::pair_score(query_data[q_idx], self.product_data[idx]);
                acc.accumulate(self.product_spec_id[idx], score);
                matched_products.set(idx, true);
                direct_scores[idx] = score;
                set_indices.push(idx);
                idx += 1;
            }
        }

        // Phase 2: shifted (neutral loss) matches.
        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            let query_nl = query_precursor_mz - qmz;
            let lo = query_nl - self.tolerance;
            let hi = query_nl + self.tolerance;
            let start = self.nl_value.partition_point(|&v| v < lo);

            for idx in start..self.nl_value.len() {
                if self.nl_value[idx] > hi {
                    break;
                }
                if self.nl_value[idx] < query_nl - self.tolerance
                    || self.nl_value[idx] > query_nl + self.tolerance
                {
                    continue;
                }
                let product_idx = self.nl_to_product[idx] as usize;
                let nl_score = K::pair_score(query_data[q_idx], self.nl_data[idx]);
                if matched_products[product_idx] {
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
                    spectrum_id: spec_id,
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

        fn spectrum_meta(_peak_data: &[f64]) -> Self::SpectrumMeta {}

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

    fn build_test_index(
        spectra: Vec<(f64, Vec<f64>, Vec<f64>)>,
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
                (210.0, vec![110.0, 100.0], vec![4.0, 1.0]),
                (205.0, vec![90.0], vec![2.0]),
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
    fn threshold_prefix_postings_support_candidate_intersection() {
        let spectra = vec![
            (200.0, vec![100.0, 150.0], vec![4.0, 1.0]),
            (300.0, vec![100.05, 250.0], vec![1.0, 5.0]),
        ];
        let prefixes =
            ThresholdPrefixPostings::build(&spectra, |data| l2_threshold_prefix_indices(data, 0.9))
                .expect("prefix postings should build");
        assert_eq!(prefixes.n_prefix_peaks(), 2);
        assert_eq!(prefixes.spectrum_prefix_mz(0), &[100.0]);
        assert_eq!(prefixes.spectrum_prefix_mz(1), &[250.0]);

        let index = build_test_index(spectra, 0.1);
        let mut state = index.new_search_state();
        index.mark_candidates_from_query_prefix_mz(&[100.02], &mut state);
        assert!(state.is_candidate(0));
        assert!(state.is_candidate(1));

        index.intersect_candidates_with_library_prefixes(&[100.02], &prefixes, &mut state);
        assert_eq!(state.secondary_candidate_touched(), &[0]);

        state.reset_secondary_candidates();
        state.reset_candidates();
        assert!(!state.is_candidate(0));
        assert!(!state.is_candidate(1));
    }

    #[test]
    fn search_direct_filters_zero_scores_and_handles_empty_inputs() {
        let empty_index = build_test_index(vec![], 0.1);
        assert!(empty_index.search_direct(&[100.0], &[1.0], &()).is_empty());

        let zero_score_index = build_test_index(vec![(200.0, vec![100.0], vec![0.0])], 0.1);
        let mut state = zero_score_index.new_search_state();
        let zero_results =
            zero_score_index.search_direct_with_state(&[100.0], &[5.0], &(), &mut state);
        assert!(zero_results.is_empty());

        let repeated_empty = zero_score_index.search_direct_with_state(&[], &[], &(), &mut state);
        assert!(repeated_empty.is_empty());
    }

    #[test]
    fn modified_search_upgrades_direct_matches_and_reuses_state_cleanly() {
        let index = build_test_index(vec![(200.0, vec![100.0], vec![1.0])], 0.1);
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
