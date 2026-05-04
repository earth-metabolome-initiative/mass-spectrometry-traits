//! Common interface for spectrum library indices.

use alloc::vec::Vec;

use crate::structs::{
    FlashIndexBuildProgress, FlashSearchResult, NoopFlashIndexBuildProgress, PepmassFilter,
    SearchState, SimilarityComputationError, SpectraIndexSetupError, TopKSearchState,
};
use crate::traits::Spectrum;

/// Common interface for spectrum library indices.
///
/// Implementations may represent all results, a fixed score threshold, or a
/// different scoring kernel, but they all expose the same index metadata,
/// reusable per-query scratch state, external-query search methods, top-k
/// search methods, and optional precursor-mass filtering.
pub trait SpectraIndex {
    /// Returns the number of spectra stored in the index.
    fn n_spectra(&self) -> u32;

    /// Returns the m/z tolerance used for peak matching.
    fn tolerance(&self) -> f64;

    /// Create reusable per-query scratch state for this index.
    fn new_search_state(&self) -> SearchState;

    /// Returns the optional precursor-mass filter used by this index.
    fn pepmass_filter(&self) -> PepmassFilter;

    /// Enable precursor-mass filtering while reporting the lazy PEPMASS index
    /// build, if one is needed.
    ///
    /// # Errors
    ///
    /// Returns [`SpectraIndexSetupError::Computation`] if the lazy PEPMASS
    /// index cannot be built.
    fn with_pepmass_filter_and_progress<G>(
        self,
        filter: PepmassFilter,
        progress: &G,
    ) -> Result<Self, SpectraIndexSetupError>
    where
        Self: Sized,
        G: FlashIndexBuildProgress + ?Sized;

    /// Enable precursor-mass filtering with the provided filter.
    ///
    /// # Errors
    ///
    /// Returns [`SpectraIndexSetupError::Computation`] if the lazy PEPMASS
    /// index cannot be built.
    fn with_pepmass_filter(self, filter: PepmassFilter) -> Result<Self, SpectraIndexSetupError>
    where
        Self: Sized,
    {
        let progress = NoopFlashIndexBuildProgress;
        self.with_pepmass_filter_and_progress(filter, &progress)
    }

    /// Enable precursor-mass filtering with an absolute Da tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`SpectraIndexSetupError::Config`] if `tolerance` is invalid.
    /// Returns [`SpectraIndexSetupError::Computation`] if the lazy PEPMASS
    /// index cannot be built.
    fn with_pepmass_tolerance(self, tolerance: f64) -> Result<Self, SpectraIndexSetupError>
    where
        Self: Sized,
    {
        self.with_pepmass_filter(PepmassFilter::within_tolerance(tolerance)?)
    }

    /// Enable precursor-mass filtering with an absolute Da tolerance while
    /// reporting the lazy PEPMASS index build.
    ///
    /// # Errors
    ///
    /// Returns [`SpectraIndexSetupError::Config`] if `tolerance` is invalid.
    /// Returns [`SpectraIndexSetupError::Computation`] if the lazy PEPMASS
    /// index cannot be built.
    fn with_pepmass_tolerance_and_progress<G>(
        self,
        tolerance: f64,
        progress: &G,
    ) -> Result<Self, SpectraIndexSetupError>
    where
        Self: Sized,
        G: FlashIndexBuildProgress + ?Sized,
    {
        self.with_pepmass_filter_and_progress(PepmassFilter::within_tolerance(tolerance)?, progress)
    }

    /// Disable precursor-mass filtering.
    fn without_pepmass_filter(self) -> Self
    where
        Self: Sized;

    /// Search an external query spectrum.
    ///
    /// Threshold-specialized indices return only results above their fixed
    /// threshold. Non-threshold indices return all positive-score direct
    /// matches.
    fn search<S>(&self, query: &S) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum;

    /// Search an external query spectrum using caller-provided scratch state.
    fn search_with_state<S>(
        &self,
        query: &S,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum;

    /// Return the best `k` results for an external query.
    fn search_top_k<S>(
        &self,
        query: &S,
        k: usize,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum;

    /// Return the best `k` results using caller-provided scratch state.
    fn search_top_k_with_state<S>(
        &self,
        query: &S,
        k: usize,
        state: &mut SearchState,
    ) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>
    where
        S: Spectrum;

    /// Stream the best `k` results using caller-provided scratch state.
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
        Emit: FnMut(FlashSearchResult);
}
