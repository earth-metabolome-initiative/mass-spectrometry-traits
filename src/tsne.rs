//! 2D t-SNE embedding of spectra from the crate's spectral similarities.
//!
//! Gated behind the `bhtsne` feature (and `std`). [`SpectralTsne::embed`] cleans
//! the spectra, builds the matching FLASH index, and feeds each spectrum's top-k
//! neighbors to bhtsne's `barnes_hut_with_neighbors`, so it scales without
//! all-pairs scoring. Distances come from the scorer ([`SpectralDistanceMetric`]):
//! cosine `arccos(sim)`, entropy `sqrt(1 - sim)`. Inputs are auto-cleaned with
//! [`SiriusMergeClosePeaks`]. The embedding is seeded, so it is reproducible by
//! default. Supported scorers: [`LinearCosine`], [`ModifiedLinearCosine`],
//! [`LinearEntropy`], [`ModifiedLinearEntropy`].
//!
//! [`SpectralTsne::embed_with_progress`] reports coarse [`SpectralTsnePhase`]s,
//! [`SpectralTsne::embed_with_frames`] also streams the layout once per epoch so
//! a caller can animate the fit, and [`SpectralTsne::embed_from_neighbors`]
//! reuses neighbors a caller already has (for example from the same index a
//! similarity graph built).
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

/// Default RNG seed for the initial embedding, so [`SpectralTsne::embed`] is
/// reproducible out of the box.
const DEFAULT_SEED: u64 = 0x6D61_7373_7370_6563;

/// Padded and floored neighbor slots get this multiple of the maximum metric
/// distance, large enough that bhtsne's Gaussian affinity underflows to zero,
/// but finite so the perplexity search never sees `INF`/`NaN`.
const NEUTRAL_DISTANCE_FACTOR: f64 = 1.0e3;

/// A coarse phase reported by [`SpectralTsne::embed_with_progress`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralTsnePhase {
    /// Merging close peaks, one tick per spectrum.
    Cleaning,
    /// Building the FLASH index (coarse: it runs in parallel, so it reports
    /// `0/1` then `1/1` rather than per spectrum).
    Indexing,
    /// Searching each spectrum's neighbors, one tick per spectrum.
    Searching,
    /// Fitting the embedding, one tick per epoch.
    Fitting,
}

/// A progress sink: `(phase, done, total)`.
pub type ProgressFn<'a> = dyn FnMut(SpectralTsnePhase, usize, usize) + 'a;

/// A frame sink: `(epoch, embedding)`, where `embedding` is the current flat
/// layout (`2 * n` values, `x0, y0, x1, y1, ...`). Borrowed, so streaming the
/// fit costs no allocation per epoch.
pub type FrameFn<'a> = dyn FnMut(usize, &[f64]) + 'a;

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
    /// in descending similarity, reporting [`SpectralTsnePhase::Indexing`] and
    /// [`SpectralTsnePhase::Searching`] progress.
    ///
    /// # Errors
    ///
    /// [`SpectralTsneError`] if the index build or a search fails.
    fn top_k_neighbors_with_progress(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
        on_progress: &mut ProgressFn<'_>,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError>;

    /// Up to `k` neighbors per spectrum, without progress reporting.
    ///
    /// # Errors
    ///
    /// [`SpectralTsneError`] if the index build or a search fails.
    fn top_k_neighbors(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        self.top_k_neighbors_with_progress(spectra, k, &mut |_, _, _| {})
    }
}

/// Per-spectrum top-k neighbors from a read-only search closure, self-hit
/// dropped, reporting [`SpectralTsnePhase::Searching`].
///
/// The queries are independent read-only lookups against the already-built
/// index, so they run in parallel within chunks; progress is reported once per
/// chunk from the calling thread, keeping the sink single-threaded (no `Send`,
/// no lock). `into_par_iter().collect()` preserves order, so `rows[i]` is still
/// spectrum `i`'s neighbors and the result is identical to a serial run.
fn collect_neighbors(
    n: usize,
    k: usize,
    on_progress: &mut ProgressFn<'_>,
    query: impl Fn(usize) -> Result<Vec<FlashSearchResult>, SimilarityComputationError> + Sync,
) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
    use rayon::prelude::*;

    let chunk = (n / 100).max(1);
    let mut rows: Vec<Vec<(u32, f64)>> = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let end = (start + chunk).min(n);
        let part: Result<Vec<Vec<(u32, f64)>>, SimilarityComputationError> = (start..end)
            .into_par_iter()
            .map(|i| {
                let hits = query(i)?;
                Ok(hits
                    .into_iter()
                    .filter(|hit| hit.spectrum_id as usize != i)
                    .take(k)
                    .map(|hit| (hit.spectrum_id, hit.score))
                    .collect())
            })
            .collect();
        rows.extend(part.map_err(SpectralTsneError::Computation)?);
        on_progress(SpectralTsnePhase::Searching, end, n);
        start = end;
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
    fn top_k_neighbors_with_progress(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
        on_progress: &mut ProgressFn<'_>,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        on_progress(SpectralTsnePhase::Indexing, 0, 1);
        let index = FlashCosineIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .parallel()
            .build(spectra)
            .map_err(map_cosine_build_error)?;
        on_progress(SpectralTsnePhase::Indexing, 1, 1);
        collect_neighbors(spectra.len(), k, on_progress, |i| {
            index.search_top_k(&spectra[i], k + 1)
        })
    }
}

impl<P: SpectrumFloat + Send + Sync> SpectralNeighbors<P> for ModifiedLinearCosine {
    fn top_k_neighbors_with_progress(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
        on_progress: &mut ProgressFn<'_>,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        on_progress(SpectralTsnePhase::Indexing, 0, 1);
        let index = FlashCosineIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .parallel()
            .build(spectra)
            .map_err(map_cosine_build_error)?;
        on_progress(SpectralTsnePhase::Indexing, 1, 1);
        collect_neighbors(spectra.len(), k, on_progress, |i| {
            index.search_modified_top_k(&spectra[i], k + 1)
        })
    }
}

impl<P: SpectrumFloat + Send + Sync> SpectralNeighbors<P> for LinearEntropy {
    fn top_k_neighbors_with_progress(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
        on_progress: &mut ProgressFn<'_>,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        on_progress(SpectralTsnePhase::Indexing, 0, 1);
        let index = FlashEntropyIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .weighted(self.is_weighted())
            .parallel()
            .build(spectra)
            .map_err(map_entropy_build_error)?;
        on_progress(SpectralTsnePhase::Indexing, 1, 1);
        collect_neighbors(spectra.len(), k, on_progress, |i| {
            index.search_top_k(&spectra[i], k + 1)
        })
    }
}

impl<P: SpectrumFloat + Send + Sync> SpectralNeighbors<P> for ModifiedLinearEntropy {
    fn top_k_neighbors_with_progress(
        &self,
        spectra: &[GenericSpectrum<P>],
        k: usize,
        on_progress: &mut ProgressFn<'_>,
    ) -> Result<Vec<Vec<(u32, f64)>>, SpectralTsneError> {
        on_progress(SpectralTsnePhase::Indexing, 0, 1);
        let index = FlashEntropyIndex::<P>::builder()
            .mz_power(self.mz_power())
            .intensity_power(self.intensity_power())
            .mz_tolerance(self.mz_tolerance())
            .weighted(self.is_weighted())
            .parallel()
            .build(spectra)
            .map_err(map_entropy_build_error)?;
        on_progress(SpectralTsnePhase::Indexing, 1, 1);
        collect_neighbors(spectra.len(), k, on_progress, |i| {
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
    seed: u64,
    min_neighbor_similarity: f64,
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
            seed: DEFAULT_SEED,
            min_neighbor_similarity: 0.0,
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

    /// Sets the RNG seed for the initial embedding. Defaults to a fixed value,
    /// so the embedding is reproducible unless this is changed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets a similarity floor below which a neighbor contributes no affinity.
    ///
    /// Real neighbors with similarity below `floor` are given an
    /// effectively-infinite distance, so a spectrum whose matches are all weak
    /// collapses toward its single nearest neighbor instead of forming an
    /// equidistant "crown" at the plot edge. The nearest neighbor of each
    /// spectrum is always kept, so no row becomes all-zero affinity. Defaults to
    /// `0.0` (off; only the always-on padding neutralization applies).
    #[must_use]
    pub fn min_neighbor_similarity(mut self, floor: f64) -> Self {
        self.min_neighbor_similarity = floor;
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
        self.embed_with_progress(spectra, scorer, &mut |_, _, _| {})
    }

    /// Like [`Self::embed`], reporting [`SpectralTsnePhase`] progress.
    ///
    /// Cleaning, Indexing, and Searching tick once per spectrum, and Fitting
    /// ticks once per epoch.
    ///
    /// # Errors
    ///
    /// As [`Self::embed`].
    pub fn embed_with_progress<P, S, Sim>(
        &self,
        spectra: &[S],
        scorer: &Sim,
        on_progress: &mut ProgressFn<'_>,
    ) -> Result<Vec<[f64; 2]>, SpectralTsneError>
    where
        P: SpectrumFloat + Send + Sync,
        S: Spectrum<Precision = P>,
        Sim: SpectralNeighbors<P> + SpectralDistanceMetric,
    {
        self.embed_with_frames(spectra, scorer, on_progress, &mut |_, _| {})
    }

    /// Like [`Self::embed_with_progress`], also calling `on_frame` once per epoch
    /// with the current flat embedding (`2 * n` values), so a caller can animate
    /// the fit converging.
    ///
    /// # Errors
    ///
    /// As [`Self::embed`].
    pub fn embed_with_frames<P, S, Sim>(
        &self,
        spectra: &[S],
        scorer: &Sim,
        on_progress: &mut ProgressFn<'_>,
        on_frame: &mut FrameFn<'_>,
    ) -> Result<Vec<[f64; 2]>, SpectralTsneError>
    where
        P: SpectrumFloat + Send + Sync,
        S: Spectrum<Precision = P>,
        Sim: SpectralNeighbors<P> + SpectralDistanceMetric,
    {
        let (n, perplexity, k) = self.validate(spectra.len())?;

        // Clean and own every spectrum for the index.
        let merger = SiriusMergeClosePeaks::<P>::new_with_precision(self.mz_tolerance)
            .map_err(SpectralTsneError::Config)?;
        let mut cleaned: Vec<GenericSpectrum<P>> = Vec::with_capacity(n);
        for (i, spectrum) in spectra.iter().enumerate() {
            let generic = to_generic(spectrum).map_err(SpectralTsneError::Conversion)?;
            cleaned.push(if self.auto_clean {
                merger.process(&generic)
            } else {
                generic
            });
            on_progress(SpectralTsnePhase::Cleaning, i + 1, n);
        }

        let neighbor_sims = scorer.top_k_neighbors_with_progress(&cleaned, k, on_progress)?;
        let neighbors =
            build_neighbor_rows(n, k, &neighbor_sims, scorer, self.min_neighbor_similarity);

        // Opens the band immediately; run_fit then ticks once per epoch.
        on_progress(SpectralTsnePhase::Fitting, 0, self.epochs);
        Ok(self.run_fit(n, perplexity, &neighbors, on_progress, on_frame))
    }

    /// Embeds from precomputed top-k neighbors, skipping cleaning, the index
    /// build, and the search.
    ///
    /// `neighbors[i]` are spectrum `i`'s neighbors as `(index, similarity)` in
    /// descending similarity, self excluded; `distance` maps similarity to the
    /// metric distance. Lets a caller share neighbors it already computed (for
    /// example for a similarity graph).
    ///
    /// # Errors
    ///
    /// [`SpectralTsneError`] for too few spectra or a non-positive perplexity or
    /// theta.
    pub fn embed_from_neighbors<D: SpectralDistanceMetric>(
        &self,
        neighbors: &[Vec<(u32, f64)>],
        distance: &D,
    ) -> Result<Vec<[f64; 2]>, SpectralTsneError> {
        let (n, perplexity, k) = self.validate(neighbors.len())?;
        let rows = build_neighbor_rows(n, k, neighbors, distance, self.min_neighbor_similarity);
        Ok(self.run_fit(n, perplexity, &rows, &mut |_, _, _| {}, &mut |_, _| {}))
    }

    /// Validates the inputs, returning `(n, clamped perplexity, neighbors k)`.
    fn validate(&self, n: usize) -> Result<(usize, f64, usize), SpectralTsneError> {
        if self.perplexity <= 0.0 {
            return Err(SpectralTsneError::InvalidPerplexity(self.perplexity));
        }
        if self.theta <= 0.0 {
            return Err(SpectralTsneError::InvalidTheta(self.theta));
        }
        // bhtsne requires `n - 1 >= 3 * perplexity`; require n >= 4 and clamp.
        if n < 4 {
            return Err(SpectralTsneError::TooFewSpectra {
                found: n,
                needed: 4,
            });
        }
        let perplexity = self.perplexity.min((n as f64 - 1.0) / 3.0);
        // Neighbors per point (bhtsne uses 3 * perplexity).
        let k = ((3.0 * perplexity) as usize).clamp(1, n - 1);
        Ok((n, perplexity, k))
    }

    /// Runs the bhtsne fit from fixed-length neighbor rows and a seeded initial
    /// embedding (so the result is deterministic), ticking
    /// [`SpectralTsnePhase::Fitting`] and emitting the layout once per epoch.
    fn run_fit(
        &self,
        n: usize,
        perplexity: f64,
        neighbors: &[Vec<bhtsne::Neighbor<f64>>],
        on_progress: &mut ProgressFn<'_>,
        on_frame: &mut FrameFn<'_>,
    ) -> Vec<[f64; 2]> {
        let initial = self.seeded_initial_embedding(n);
        // One placeholder sample per point; only the count is used here.
        let index_samples: Vec<[f64; 1]> = (0..n).map(|i| [i as f64]).collect();
        let samples: Vec<&[f64]> = index_samples.iter().map(|s| s.as_slice()).collect();

        let epochs = self.epochs;
        let mut tsne = bhtsne::tSNE::new(&samples);
        tsne.embedding_dim(2)
            .perplexity(perplexity)
            .epochs(epochs)
            .learning_rate(self.learning_rate)
            .initial_embedding(initial)
            // bhtsne calls this sequentially once per epoch on this thread.
            .epoch_callback(move |epoch, embedding| {
                on_progress(SpectralTsnePhase::Fitting, epoch + 1, epochs);
                on_frame(epoch, embedding);
            });
        tsne.barnes_hut_with_neighbors(self.theta, neighbors);

        tsne.embedding()
            .chunks_exact(2)
            .map(|point| [point[0], point[1]])
            .collect()
    }

    /// Deterministic small initial coordinates (`n * 2` values in `[-1e-4, 1e-4]`)
    /// from the seed, via SplitMix64.
    fn seeded_initial_embedding(&self, n: usize) -> Vec<f64> {
        let mut state = self.seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Map the top 53 bits to [0, 1), then to [-1e-4, 1e-4].
            let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
            (unit * 2.0 - 1.0) * 1e-4
        };
        (0..n * 2).map(|_| next()).collect()
    }
}

/// Builds fixed-length-`k` neighbor rows from `(index, similarity)` rows: real
/// neighbors at their metric distance, padded with distinct other indices at the
/// maximum distance (bhtsne needs fixed-length rows of distinct indices).
fn build_neighbor_rows<D: SpectralDistanceMetric>(
    n: usize,
    k: usize,
    neighbor_sims: &[Vec<(u32, f64)>],
    distance: &D,
    min_similarity: f64,
) -> Vec<Vec<bhtsne::Neighbor<f64>>> {
    let max_distance = distance.distance(0.0);
    let neutral_distance = max_distance * NEUTRAL_DISTANCE_FACTOR;
    neighbor_sims
        .iter()
        .enumerate()
        .map(|(i, row_sims)| {
            neighbor_row(
                i,
                n,
                k,
                row_sims,
                distance,
                neutral_distance,
                min_similarity,
            )
        })
        .collect()
}

/// Spectrum `i`'s row of exactly `k` neighbors. Real neighbors get their metric
/// distance, except weak ones (similarity below `min_similarity`) and padded
/// slots get `neutral_distance` so they contribute no affinity. The nearest
/// neighbor is always kept finite, so the row is never all-zero affinity.
fn neighbor_row<D: SpectralDistanceMetric>(
    i: usize,
    n: usize,
    k: usize,
    row_sims: &[(u32, f64)],
    distance: &D,
    neutral_distance: f64,
    min_similarity: f64,
) -> Vec<bhtsne::Neighbor<f64>> {
    let mut row: Vec<bhtsne::Neighbor<f64>> = row_sims
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, &(id, similarity))| {
            // Keep the nearest neighbor (rank 0) finite so the row always has at
            // least one real attractor; neutralize weaker ones below the floor.
            let neighbor_distance = if rank > 0 && similarity < min_similarity {
                neutral_distance
            } else {
                distance.distance(similarity)
            };
            bhtsne::Neighbor {
                index: id as usize,
                distance: neighbor_distance,
            }
        })
        .collect();

    if row.len() < k {
        let mut used = alloc::vec![false; n];
        used[i] = true;
        for neighbor in &row {
            used[neighbor.index] = true;
        }
        // If the search returned nothing, the first padded slot must stay finite
        // so the row is not all-zero affinity.
        let mut needs_anchor = row.is_empty();
        // k <= n - 1, so there are always enough distinct other indices to pad.
        let mut candidate = 0;
        while row.len() < k {
            while used[candidate] {
                candidate += 1;
            }
            used[candidate] = true;
            let padded_distance = if needs_anchor {
                needs_anchor = false;
                distance.distance(0.0)
            } else {
                neutral_distance
            };
            row.push(bhtsne::Neighbor {
                index: candidate,
                distance: padded_distance,
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
