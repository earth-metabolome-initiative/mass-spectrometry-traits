//! Shared internals for the Flash inverted m/z index.
//!
//! All items are `pub(crate)` — individual variant modules (`flash_cosine_index`,
//! `flash_entropy_index`) expose only their public wrappers.

use alloc::vec::Vec;

use bitvec::prelude::*;

use super::similarity_errors::SimilarityComputationError;

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlashSearchResult {
    /// Index of the library spectrum (0-based, insertion order).
    pub spectrum_id: u32,
    /// Similarity score in `[0, 1]`.
    pub score: f64,
    /// Number of matched peak pairs.
    pub n_matches: usize,
}

// ---------------------------------------------------------------------------
// DenseAccumulator
// ---------------------------------------------------------------------------

/// Per-query score accumulator. Uses dense arrays (one slot per library
/// spectrum) with a `touched` list for efficient reset.
struct DenseAccumulator {
    scores: Vec<f64>,
    counts: Vec<u32>,
    touched: Vec<u32>,
}

impl DenseAccumulator {
    fn new(n_spectra: usize) -> Self {
        Self {
            scores: alloc::vec![0.0; n_spectra],
            counts: alloc::vec![0; n_spectra],
            touched: Vec::with_capacity(n_spectra / 4),
        }
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
/// Allocating a `SearchState` once and passing it to repeated search calls
/// avoids per-query allocation of the dense accumulator and bitvec. For
/// single queries the simple `search` / `search_modified` methods (which
/// allocate internally) are more convenient.
pub struct SearchState {
    acc: DenseAccumulator,
    matched_products: BitVec,
    direct_scores: Vec<f64>,
}

impl SearchState {
    /// Create a new `SearchState` sized for the given index.
    fn new(n_spectra: usize, n_products: usize) -> Self {
        Self {
            acc: DenseAccumulator::new(n_spectra),
            matched_products: bitvec![0; n_products],
            direct_scores: alloc::vec![0.0; n_products],
        }
    }
}

// ---------------------------------------------------------------------------
// FlashIndex<K> — the inverted m/z index
// ---------------------------------------------------------------------------

/// Inverted m/z index shared by all Flash search variants.
pub(crate) struct FlashIndex<K: FlashKernel> {
    // Product ion index (sorted by m/z).
    pub(crate) product_mz: Vec<f64>,
    product_spec_id: Vec<u32>,
    pub(crate) product_data: Vec<f64>,

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

        for (spec_id, (prec_mz, mz_vals, data_vals)) in spectra.iter().enumerate() {
            let spec_id = spec_id as u32; // safe: checked above
            spectrum_meta_vec.push(K::spectrum_meta(data_vals));

            for (&mz, &data) in mz_vals.iter().zip(data_vals.iter()) {
                peak_mz_flat.push(mz);
                peak_spec_id_flat.push(spec_id);
                peak_data_flat.push(data);
                peak_nl_flat.push(*prec_mz - mz);
            }
        }

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
        if self.n_spectra == 0 || query_mz.is_empty() {
            return Vec::new();
        }

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
                if (self.product_mz[idx] - qmz).abs() > self.tolerance {
                    continue;
                }
                let score = K::pair_score(query_data[q_idx], self.product_data[idx]);
                acc.accumulate(self.product_spec_id[idx], score);
            }
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

        // Destructure state to avoid borrow conflicts between fields.
        let SearchState {
            acc,
            matched_products,
            direct_scores,
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
                if (self.product_mz[idx] - qmz).abs() > self.tolerance {
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
                if (self.nl_value[idx] - query_nl).abs() > self.tolerance {
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
