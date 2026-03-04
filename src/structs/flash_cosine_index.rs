//! Flash inverted m/z index for cosine spectral similarity.
//!
//! Provides O(Q_peaks * log(P_total)) library-scale search instead of
//! O(N * pairwise_cost). Exact equivalence to [`super::LinearCosine`] on
//! well-separated spectra (consecutive peaks > 2 * tolerance apart).

use alloc::vec::Vec;

use num_traits::{Float, Pow, ToPrimitive};

use geometric_traits::prelude::{Finite, Number, TotalOrd};

use super::cosine_common::{
    to_f64_checked_for_computation, validate_non_negative_tolerance, validate_well_separated,
};
use super::flash_common::{FlashIndex, FlashKernel, FlashSearchResult, SearchState};
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
/// let library: [GenericSpectrum<f64, f64>; 2] = [
///     GenericSpectrum::cocaine().unwrap(),
///     GenericSpectrum::glucose().unwrap(),
/// ];
/// let index = FlashCosineIndex::new(1.0_f64, 1.0_f64, 0.1_f64, library.iter())
///     .expect("index build should succeed");
///
/// let query: GenericSpectrum<f64, f64> = GenericSpectrum::cocaine().unwrap();
/// let results = index.search(&query).expect("search should succeed");
/// assert!(results.iter().any(|r| r.spectrum_id == 0 && r.score > 0.99));
/// ```
pub struct FlashCosineIndex<EXP> {
    inner: FlashIndex<CosineKernel>,
    mz_power: EXP,
    intensity_power: EXP,
}

impl<EXP: Number> FlashCosineIndex<EXP> {
    /// Returns the m/z power used for scoring.
    #[inline]
    pub fn mz_power(&self) -> EXP {
        self.mz_power
    }

    /// Returns the intensity power used for scoring.
    #[inline]
    pub fn intensity_power(&self) -> EXP {
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
}

impl<EXP> FlashCosineIndex<EXP>
where
    EXP: Number + ToPrimitive,
{
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
    pub fn new<'a, MZ, S>(
        mz_power: EXP,
        intensity_power: EXP,
        mz_tolerance: MZ,
        spectra: impl IntoIterator<Item = &'a S>,
    ) -> Result<Self, FlashCosineIndexError>
    where
        MZ: Number + ToPrimitive + PartialOrd,
        S: Spectrum + 'a,
        S::Mz: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
        S::Intensity: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    {
        // Validate config.
        validate_non_negative_tolerance(mz_tolerance).map_err(FlashCosineIndexError::Config)?;
        let tolerance = to_f64_checked_for_computation(mz_tolerance, "mz_tolerance")
            .map_err(FlashCosineIndexError::Computation)?;

        // Validate that power parameters are representable as f64.
        to_f64_checked_for_computation(mz_power, "mz_power")
            .map_err(FlashCosineIndexError::Computation)?;
        to_f64_checked_for_computation(intensity_power, "intensity_power")
            .map_err(FlashCosineIndexError::Computation)?;

        // Prepare spectrum data.
        let mut prepared: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();

        for spectrum in spectra {
            let mut mz_vals = Vec::with_capacity(spectrum.len());
            let mut data_vals = Vec::with_capacity(spectrum.len());

            for (mz, intensity) in spectrum.peaks() {
                let product = mz.pow(mz_power) * intensity.pow(intensity_power);
                let product_f64 = to_f64_checked_for_computation(product, "peak_product")
                    .map_err(FlashCosineIndexError::Computation)?;
                let mz_f64 = to_f64_checked_for_computation(mz, "mz")
                    .map_err(FlashCosineIndexError::Computation)?;
                mz_vals.push(mz_f64);
                data_vals.push(product_f64);
            }

            validate_well_separated(&mz_vals, tolerance, "library spectrum")
                .map_err(FlashCosineIndexError::Computation)?;

            let precursor_f64 =
                to_f64_checked_for_computation(spectrum.precursor_mz(), "precursor_mz")
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
        S::Mz: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
        S::Intensity: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
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
        S::Mz: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
        S::Intensity: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        Ok(self
            .inner
            .search_direct_with_state(&query_mz, &query_data, &query_meta, state))
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
        S::Mz: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
        S::Intensity: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        let precursor_f64 =
            to_f64_checked_for_computation(query.precursor_mz(), "query_precursor_mz")?;
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
        S::Mz: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
        S::Intensity: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    {
        let (query_mz, query_data) = self.prepare_query(query)?;
        let query_meta = CosineKernel::spectrum_meta(&query_data);
        let precursor_f64 =
            to_f64_checked_for_computation(query.precursor_mz(), "query_precursor_mz")?;
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
        S::Mz: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
        S::Intensity: Pow<EXP, Output = S::Mz> + Float + Number + Finite + TotalOrd + ToPrimitive,
    {
        let mut mz_vals = Vec::with_capacity(query.len());
        let mut data_vals = Vec::with_capacity(query.len());

        for (mz, intensity) in query.peaks() {
            let product = mz.pow(self.mz_power) * intensity.pow(self.intensity_power);
            let product_f64 = to_f64_checked_for_computation(product, "peak_product")?;
            let mz_f64 = to_f64_checked_for_computation(mz, "mz")?;
            mz_vals.push(mz_f64);
            data_vals.push(product_f64);
        }

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
