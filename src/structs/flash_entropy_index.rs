//! Flash inverted m/z index for spectral entropy similarity.
//!
//! Provides O(Q_peaks * log(P_total)) library-scale search instead of
//! O(N * pairwise_cost). Exact equivalence to [`super::LinearEntropy`] on
//! well-separated spectra (consecutive peaks > 2 * tolerance apart).

use alloc::vec::Vec;

use super::cosine_common::{
    ensure_finite, validate_non_negative_tolerance, validate_numeric_parameter,
    validate_well_separated,
};
use super::entropy_common::{entropy_pair, prepare_entropy_peaks};
use super::flash_common::{FlashIndex, FlashKernel, FlashSearchResult, SearchState};
use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::Spectrum;

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
/// let library = [
///     GenericSpectrum::cocaine().unwrap(),
///     GenericSpectrum::glucose().unwrap(),
/// ];
/// let index = FlashEntropyIndex::new(0.0, 1.0, 0.1, true, library.iter())
///     .expect("index build should succeed");
///
/// let query = GenericSpectrum::cocaine().unwrap();
/// let results = index.search(&query).expect("search should succeed");
/// assert!(results.iter().any(|r| r.spectrum_id == 0 && r.score > 0.99));
/// ```
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
        validate_numeric_parameter(mz_power, "mz_power").map_err(FlashEntropyIndexError::Config)?;
        validate_numeric_parameter(intensity_power, "intensity_power")
            .map_err(FlashEntropyIndexError::Config)?;
        validate_non_negative_tolerance(mz_tolerance).map_err(FlashEntropyIndexError::Config)?;

        let tolerance = mz_tolerance;
        let mz_power_f64 = mz_power;
        let intensity_power_f64 = intensity_power;

        let mut prepared: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();

        for spectrum in spectra {
            let peaks =
                prepare_entropy_peaks(spectrum, weighted, mz_power_f64, intensity_power_f64)
                    .map_err(FlashEntropyIndexError::Computation)?;

            if peaks.int.is_empty() {
                let precursor_f64 = ensure_finite(spectrum.precursor_mz(), "precursor_mz")
                    .map_err(FlashEntropyIndexError::Computation)?;
                prepared.push((precursor_f64, Vec::new(), Vec::new()));
                continue;
            }

            validate_well_separated(&peaks.mz, tolerance, "library spectrum")
                .map_err(FlashEntropyIndexError::Computation)?;

            let precursor_f64 = ensure_finite(spectrum.precursor_mz(), "precursor_mz")
                .map_err(FlashEntropyIndexError::Computation)?;

            prepared.push((precursor_f64, peaks.mz, peaks.int));
        }

        let inner = FlashIndex::<EntropyKernel>::build(tolerance, prepared)
            .map_err(FlashEntropyIndexError::Computation)?;

        Ok(Self {
            inner,
            weighted,
            mz_power_f64,
            intensity_power_f64,
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

    /// Modified (direct + shifted) search against the library.
    pub fn search_modified<S>(
        &self,
        query: &S,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let precursor_f64 = ensure_finite(query.precursor_mz(), "query_precursor_mz")?;
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
        let precursor_f64 = ensure_finite(query.precursor_mz(), "query_precursor_mz")?;
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
