//! Common builder interface for spectrum library indices.

use crate::structs::{
    FlashIndexBuildOptions, FlashIndexBuildProgress, PepmassFilter, SimilarityConfigError,
};
use crate::traits::Spectrum;

/// Common interface for index builders.
///
/// The concrete builders hold metric-specific parameters, while this trait
/// standardizes cross-cutting construction options such as parallelism,
/// progress reporting, and precursor-mass filtering.
pub trait SpectraIndexBuilder<'a>: Sized {
    /// Index produced by this builder.
    type Index;

    /// Error returned when construction fails.
    type Error;

    /// Returns the shared build options.
    fn options(&self) -> &FlashIndexBuildOptions<'a>;

    /// Returns mutable shared build options.
    fn options_mut(&mut self) -> &mut FlashIndexBuildOptions<'a>;

    /// Build the index from a stable collection of borrowed spectra.
    fn build<S>(self, spectra: &[S]) -> Result<Self::Index, Self::Error>
    where
        S: Spectrum + Sync;

    /// Use sequential construction.
    #[inline]
    fn sequential(mut self) -> Self {
        self.options_mut().set_parallel(false);
        self
    }

    /// Use Rayon-backed construction where available.
    #[cfg(feature = "rayon")]
    #[inline]
    fn parallel(mut self) -> Self {
        self.options_mut().set_parallel(true);
        self
    }

    /// Report construction progress to the provided sink.
    #[inline]
    fn progress(mut self, progress: &'a (dyn FlashIndexBuildProgress + Sync + 'a)) -> Self {
        self.options_mut().set_progress(progress);
        self
    }

    /// Configure precursor-mass filtering.
    #[inline]
    fn pepmass_filter(mut self, filter: PepmassFilter) -> Self {
        self.options_mut().set_pepmass_filter(filter);
        self
    }

    /// Configure precursor-mass filtering with an absolute Da tolerance.
    #[inline]
    fn pepmass_tolerance(mut self, tolerance: f64) -> Result<Self, SimilarityConfigError> {
        self.options_mut()
            .set_pepmass_filter(PepmassFilter::within_tolerance(tolerance)?);
        Ok(self)
    }
}
