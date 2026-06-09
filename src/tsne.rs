//! 2D t-SNE embedding of spectra from the crate's spectral similarities.
//!
//! Gated behind the `bhtsne` feature (and `std`). [`SpectralTsne::embed`] cleans
//! the spectra, builds the matching FLASH index, and feeds each spectrum's top-k
//! neighbors to bhtsne's `barnes_hut_with_neighbors`, so it scales without
//! all-pairs scoring. Distances come from the scorer ([`SpectralDistanceMetric`]):
//! cosine `arccos(sim)`, entropy `sqrt(1 - sim)`. Supported scorers:
//! [`LinearCosine`], [`ModifiedLinearCosine`], [`LinearEntropy`],
//! [`ModifiedLinearEntropy`].
//!
//! # Example
//!
//! ```
//! use mass_spectrometry::prelude::*;
//!
//! let library: Vec<GenericSpectrum> = vec![
//!     GenericSpectrum::cocaine().unwrap(),
//!     GenericSpectrum::glucose().unwrap(),
//!     GenericSpectrum::aspirin().unwrap(),
//!     GenericSpectrum::phenylalanine().unwrap(),
//!     GenericSpectrum::salicin().unwrap(),
//! ];
//! let scorer = LinearCosine::new(1.0, 1.0, 0.1).unwrap();
//!
//! let embedding = SpectralTsne::new()
//!     .perplexity(1.0)
//!     .epochs(250)
//!     .mz_tolerance(0.1)
//!     .embed(&library, &scorer)
//!     .unwrap();
//!
//! assert_eq!(embedding.len(), library.len());
//! assert!(embedding.iter().all(|[x, y]| x.is_finite() && y.is_finite()));
//! ```

use alloc::vec::Vec;

use crate::structs::{
    FlashCosineIndex, FlashCosineIndexError, FlashEntropyIndex, FlashEntropyIndexError,
    FlashSearchResult, GenericSpectrum, GenericSpectrumMutationError, LinearCosine, LinearEntropy,
    ModifiedLinearCosine, ModifiedLinearEntropy, SimilarityComputationError, SimilarityConfigError,
    SiriusMergeClosePeaks,
};
use crate::traits::{
    SpectraIndexBuilder, SpectralDistanceMetric, SpectralProcessor, Spectrum, SpectrumFloat,
    SpectrumMut,
};

/// Error returned by [`SpectralTsne::embed`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpectralTsneError {
    /// Fewer spectra than needed to fit an embedding.
    #[error("t-SNE needs at least {needed} spectra, got {found}")]
    TooFewSpectra {
        /// Spectra supplied.
        found: usize,
        /// Minimum required.
        needed: usize,
    },
    /// Perplexity was not strictly positive.
    #[error("perplexity must be > 0, got {0}")]
    InvalidPerplexity(f64),
    /// Theta was not strictly positive.
    #[error("theta must be > 0, got {0}")]
    InvalidTheta(f64),
    /// An invalid configuration (cleaning tolerance or index parameter).
    #[error(transparent)]
    Config(SimilarityConfigError),
    /// An index build or neighbor search failed.
    #[error(transparent)]
    Computation(SimilarityComputationError),
    /// A spectrum could not be materialized.
    #[error(transparent)]
    Conversion(GenericSpectrumMutationError),
}

/// Each spectrum's top-k neighbors via the scorer's matching FLASH index.
pub trait SpectralNeighbors<P: SpectrumFloat> {
    /// Up to `k` neighbors per spectrum as `(index, similarity)`, self excluded,
    /// in descending similarity.
    ///
    /// # Errors
    ///
    /// [`SpectralTsneError`] if the index build or a search fails.
    fn top_k_neighbors(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError>;
}

/// Per-spectrum top-k neighbors from a search closure, self-hit dropped.
fn collect_neighbors(
    n: usize,
    k: usize,
    mut query: impl FnMut(usize) -> Result<Vec<FlashSearchResult>, SimilarityComputationError>,
) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let hits = query(i).map_err(SpectralTsneError::Computation)?;
        rows.push(
            hits.into_iter()
                .filter(|hit| hit.spectrum_id as usize != i)
                .take(k)
                .map(|hit| (hit.spectrum_id, hit.score))
                .collect(),
        );
    }
    Ok(rows)
}

fn map_cosine_build_error(error: FlashCosineIndexError) -> SpectralTsneError {
    match error {
        FlashCosineIndexError::Config(config) => SpectralTsneError::Config(config),
        FlashCosineIndexError::Computation(computation) => {
            SpectralTsneError::Computation(computation)
        }
    }
}

fn map_entropy_build_error(error: FlashEntropyIndexError) -> SpectralTsneError {
    match error {
        FlashEntropyIndexError::Config(config) => SpectralTsneError::Config(config),
        FlashEntropyIndexError::Computation(computation) => {
            SpectralTsneError::Computation(computation)
        }
    }
}

impl<P: SpectrumFloat + Send + Sync> SpectralNeighbors<P> for LinearCosine {
    fn top_k_neighbors(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        let index = FlashCosineIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .build(spectra)
            .map_err(map_cosine_build_error)?;
        collect_neighbors(spectra.len(), k, |i| index.search_top_k(&spectra[i], k + 1))
    }
}

impl<P: SpectrumFloat + Send + Sync> SpectralNeighbors<P> for ModifiedLinearCosine {
    fn top_k_neighbors(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        let index = FlashCosineIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .build(spectra)
            .map_err(map_cosine_build_error)?;
        collect_neighbors(spectra.len(), k, |i| {
            index.search_modified_top_k(&spectra[i], k + 1)
        })
    }
}

impl<P: SpectrumFloat + Send + Sync> SpectralNeighbors<P> for LinearEntropy {
    fn top_k_neighbors(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        let index = FlashEntropyIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .weighted(self.is_weighted())
            .build(spectra)
            .map_err(map_entropy_build_error)?;
        collect_neighbors(spectra.len(), k, |i| index.search_top_k(&spectra[i], k + 1))
    }
}

impl<P: SpectrumFloat + Send + Sync> SpectralNeighbors<P> for ModifiedLinearEntropy {
    fn top_k_neighbors(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        let index = FlashEntropyIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .weighted(self.is_weighted())
            .build(spectra)
            .map_err(map_entropy_build_error)?;
        collect_neighbors(spectra.len(), k, |i| {
            index.search_modified_top_k(&spectra[i], k + 1)
        })
    }
}

/// Builder and runner for a 2D t-SNE embedding of spectra.
#[derive(Debug, Clone, Copy)]
pub struct SpectralTsne {
    perplexity: f64,
    epochs: usize,
    theta: f64,
    learning_rate: f64,
    mz_tolerance: f64,
    auto_clean: bool,
}

impl Default for SpectralTsne {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            epochs: 1000,
            theta: 0.5,
            learning_rate: 200.0,
            mz_tolerance: 0.1,
            auto_clean: true,
        }
    }
}

impl SpectralTsne {
    /// Creates a runner with default hyperparameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the perplexity. Clamped to at most `(n - 1) / 3` at embed time.
    #[must_use]
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Sets the number of fitting iterations.
    #[must_use]
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the Barnes-Hut theta (smaller is more accurate and slower).
    #[must_use]
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Sets the gradient-descent learning rate.
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the auto-clean m/z tolerance (match the scorer's `mz_tolerance`).
    #[must_use]
    pub fn mz_tolerance(mut self, mz_tolerance: f64) -> Self {
        self.mz_tolerance = mz_tolerance;
        self
    }

    /// Sets whether inputs are auto-cleaned with [`SiriusMergeClosePeaks`]
    /// before indexing. Enabled by default.
    #[must_use]
    pub fn auto_clean(mut self, auto_clean: bool) -> Self {
        self.auto_clean = auto_clean;
        self
    }

    /// Embeds `spectra` into 2D, one `[x, y]` per spectrum in input order, in `f64`.
    ///
    /// `scorer` must be a FLASH-backed scorer ([`SpectralNeighbors`] +
    /// [`SpectralDistanceMetric`]).
    ///
    /// # Errors
    ///
    /// [`SpectralTsneError`] for too few spectra, a non-positive perplexity or
    /// theta, an invalid tolerance, an index or search failure, or a spectrum
    /// that cannot be materialized.
    pub fn embed<P, S, Sim>(
        &self,
        spectra: &[S],
        scorer: &Sim,
    ) -> Result<Vec<[f64; 2]>, SpectralTsneError>
    where
        P: SpectrumFloat + Send + Sync,
        S: Spectrum<Precision = P>,
        Sim: SpectralNeighbors<P> + SpectralDistanceMetric,
    {
        if self.perplexity <= 0.0 {
            return Err(SpectralTsneError::InvalidPerplexity(self.perplexity));
        }
        if self.theta <= 0.0 {
            return Err(SpectralTsneError::InvalidTheta(self.theta));
        }

        // bhtsne requires `n - 1 >= 3 * perplexity`; require n >= 4 and clamp.
        let n = spectra.len();
        if n < 4 {
            return Err(SpectralTsneError::TooFewSpectra {
                found: n,
                needed: 4,
            });
        }
        let perplexity = self.perplexity.min((n as f64 - 1.0) / 3.0);
        // Neighbors per point (bhtsne uses 3 * perplexity).
        let k = ((3.0 * perplexity) as usize).clamp(1, n - 1);

        // Clean and own every spectrum for the index.
        let merger = SiriusMergeClosePeaks::<P>::new_with_precision(self.mz_tolerance)
            .map_err(SpectralTsneError::Config)?;
        let cleaned: Vec<GenericSpectrum<P>> = spectra
            .iter()
            .map(|spectrum| {
                let generic = to_generic(spectrum).map_err(SpectralTsneError::Conversion)?;
                Ok(if self.auto_clean {
                    merger.process(&generic)
                } else {
                    generic
                })
            })
            .collect::<Result<_, SpectralTsneError>>()?;

        // Top-k neighbors as (index, similarity); descending similarity is
        // ascending distance.
        let neighbor_sims = scorer.top_k_neighbors(&cleaned, k)?;
        let max_distance = scorer.distance(0.0);
        let neighbors: Vec<Vec<bhtsne::Neighbor<f64>>> = neighbor_sims
            .iter()
            .enumerate()
            .map(|(i, row_sims)| neighbor_row(i, n, k, row_sims, scorer, max_distance))
            .collect();

        // One placeholder sample per point; only the count is used here.
        let index_samples: Vec<[f64; 1]> = (0..n).map(|i| [i as f64]).collect();
        let samples: Vec<&[f64]> = index_samples.iter().map(|s| s.as_slice()).collect();

        let mut tsne = bhtsne::tSNE::new(&samples);
        tsne.embedding_dim(2)
            .perplexity(perplexity)
            .epochs(self.epochs)
            .learning_rate(self.learning_rate);
        tsne.barnes_hut_with_neighbors(self.theta, &neighbors);

        Ok(tsne
            .embedding()
            .chunks_exact(2)
            .map(|point| [point[0], point[1]])
            .collect())
    }
}

/// Spectrum `i`'s row of exactly `k` neighbors at their metric distance, padded
/// with distinct other indices at `max_distance` (bhtsne needs fixed-length rows).
fn neighbor_row<Sim: SpectralDistanceMetric>(
    i: usize,
    n: usize,
    k: usize,
    row_sims: &[(u32, f64)],
    scorer: &Sim,
    max_distance: f64,
) -> Vec<bhtsne::Neighbor<f64>> {
    let mut row: Vec<bhtsne::Neighbor<f64>> = row_sims
        .iter()
        .map(|&(id, similarity)| bhtsne::Neighbor {
            index: id as usize,
            distance: scorer.distance(similarity),
        })
        .collect();

    if row.len() < k {
        let mut used = alloc::vec![false; n];
        used[i] = true;
        for neighbor in &row {
            used[neighbor.index] = true;
        }
        // k <= n - 1, so there are always enough distinct other indices to pad.
        let mut candidate = 0;
        while row.len() < k {
            while used[candidate] {
                candidate += 1;
            }
            used[candidate] = true;
            row.push(bhtsne::Neighbor {
                index: candidate,
                distance: max_distance,
            });
        }
    }

    row
}

/// Owns any spectrum as a [`GenericSpectrum`] of the same precision.
fn to_generic<P, S>(spectrum: &S) -> Result<GenericSpectrum<P>, GenericSpectrumMutationError>
where
    P: SpectrumFloat,
    S: Spectrum<Precision = P>,
{
    let mut generic =
        GenericSpectrum::<P>::try_with_capacity(spectrum.precursor_mz().to_f64(), spectrum.len())?;
    for (mz, intensity) in spectrum.peaks() {
        generic.add_peak(mz, intensity)?;
    }
    Ok(generic)
}
