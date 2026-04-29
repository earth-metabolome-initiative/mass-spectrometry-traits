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
};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::Spectrum;

// ---------------------------------------------------------------------------
// CosineKernel
// ---------------------------------------------------------------------------

pub(crate) struct CosineKernel;

/// Norm stored per library spectrum.
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
            let mz_vals: Vec<f64> = spectrum.mz().collect();
            let data_vals = normalized_peak_products(spectrum, mz_power, intensity_power)
                .map_err(FlashCosineIndexError::Computation)?;

            validate_well_separated(&mz_vals, tolerance, "library spectrum")
                .map_err(FlashCosineIndexError::Computation)?;

            let precursor_f64 = ensure_finite(spectrum.precursor_mz(), "precursor_mz")
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
        mut emit: Emit,
    ) -> Result<(), SimilarityComputationError>
    where
        S: Spectrum,
        Emit: FnMut(FlashSearchResult),
    {
        ensure_finite(score_threshold, "score_threshold")?;
        if score_threshold <= 0.0 {
            for result in self.search_with_state(query, state)? {
                emit(result);
            }
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
        let precursor_f64 = ensure_finite(query.precursor_mz(), "query_precursor_mz")?;
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
        let precursor_f64 = ensure_finite(query.precursor_mz(), "query_precursor_mz")?;
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
        let mz_vals: Vec<f64> = query.mz().collect();
        let data_vals = normalized_peak_products(query, self.mz_power, self.intensity_power)?;

        validate_well_separated(&mz_vals, self.inner.tolerance, "query spectrum")?;

        Ok((mz_vals, data_vals))
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
