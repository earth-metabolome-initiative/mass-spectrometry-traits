# Adding the fitsne GPU backend

This describes how to drive [`fitsne`](https://github.com/LucaCappelletti94/fitsne) from this crate, alongside or instead of the current bhtsne path. fitsne is a GPU-accelerated FIt-SNE that takes the same caller-supplied neighbor graph that `barnes_hut_with_neighbors` takes, so the existing spectral scorers feed it unchanged. It runs the optimization on the GPU (CUDA or wgpu) or the CPU, with an O(N) interpolation-plus-FFT repulsion for scale, and it is metric-agnostic, so the non-metric spectral similarities are fine.

The integration is small because the scorers already produce the neighbor graph and the similarity-to-distance mapping.

## Dependency

Add fitsne and a cubecl runtime. fitsne pins cubecl and cubek-fft by git rev, so pin fitsne by git rev too.

```toml
[dependencies]
fitsne = { git = "https://github.com/LucaCappelletti94/fitsne", rev = "..." }

[features]
# Pick the backends you ship. cuda for the 4090, wgpu for portability, cpu for development.
tsne-gpu = ["fitsne/cuda"]
```

fitsne is f32 today, while the spectral distances are f64, so the impl below casts. When fitsne grows an f64 path the cast goes away.

## Implement `NeighborIndex` for the four scorers

The scorers already implement `SpectralNeighbors<P>` (which returns `(index, similarity)`) and `SpectralDistanceMetric` (which maps a similarity to a distance). fitsne's `NeighborIndex` trait just wants the converted `Neighbor` graph, so the impl is a thin wrapper. A blanket `impl for S` is blocked by the orphan rule, so use a macro over the four concrete scorers.

```rust
use crate::{
    GenericSpectrum, LinearCosine, LinearEntropy, ModifiedLinearCosine, ModifiedLinearEntropy,
    SpectralDistanceMetric, SpectralNeighbors, SpectrumFloat, SpectralTsneError,
};

macro_rules! impl_fitsne_index {
    ($scorer:ty) => {
        impl<P: SpectrumFloat + Send + Sync> fitsne::NeighborIndex<GenericSpectrum<P>> for $scorer {
            type Error = SpectralTsneError;

            fn neighbor_graph(
                &self,
                spectra: &[GenericSpectrum<P>],
                k: usize,
            ) -> Result<Vec<Vec<fitsne::Neighbor<f32>>>, SpectralTsneError> {
                let sims = self.top_k_neighbors(spectra, k)?;
                Ok(sims
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&(idx, sim)| {
                                // Descending similarity becomes ascending distance.
                                fitsne::Neighbor::new(idx as usize, self.distance(sim) as f32)
                            })
                            .collect()
                    })
                    .collect())
            }
        }
    };
}

impl_fitsne_index!(LinearCosine);
impl_fitsne_index!(ModifiedLinearCosine);
impl_fitsne_index!(LinearEntropy);
impl_fitsne_index!(ModifiedLinearEntropy);
```

If you want to keep the `min_neighbor_similarity` floor that `SpectralTsne` applies (weak neighbors pushed to a neutral large distance so a spectrum with only weak matches does not get glued to them), fold the same neutral-distance logic into `neighbor_graph`, since fitsne's trait does not know about that floor. The reference is `build_neighbor_rows` and `neighbor_row` in `src/tsne.rs`.

## Drive the embedding

The call mirrors the current bhtsne path. `fit_with_index` builds the graph from the scorer and runs the embedding in one step.

```rust
use cubecl::cuda::CudaRuntime;
use fitsne::FitSne;

let embedding: Vec<f32> = FitSne::<CudaRuntime>::new(spectra.len())
    .embedding_dim(2)
    .perplexity(30.0)
    .epochs(1_000)
    .fft_repulsion(64) // O(N) GPU repulsion. Omit for the dense O(n^2) path.
    .fit_with_index(&scorer, &spectra, k, &Default::default())?;

// The loss, comparable out of band against the bhtsne run on the same graph.
// (kl_divergence is available on the builder after the fit.)
```

To keep both backends, make the `SpectralTsne` embed generic over a cubecl `Runtime`, or add a parallel `embed_gpu` that uses fitsne while the existing `embed` keeps bhtsne. The scorer, the spectra, the perplexity, and `k` are shared, so only the optimizer call changes.

## Notes and caveats

- Repulsion choice. The default dense O(n^2) repulsion is exact and best at small N. Enable `fft_repulsion(boxes)` for the O(N) path that scales to millions. The FFT is fast on CUDA and wgpu but slow on the CPU runtime at large grids, so use the dense path on CPU and the FFT path on the GPU.
- Grid sizing. `fft_repulsion(boxes)` uses a fixed box count, and the grid stretches to the embedding bounds each epoch. Too few boxes underestimates the normalization and lets the repulsion run away, so prefer a generous value (tens). This is a known rough edge, a fixed node spacing will replace the fixed count later.
- Determinism. fitsne seeds its initial embedding deterministically, so a fixed seed and the same neighbor graph reproduce the run.

## Validation

Run both backends on the same neighbor graph and seed and compare. fitsne and bhtsne both expose a KL divergence, and the downstream classifier-ensemble score from the benchmark applies to either embedding. The expectation is that the fitsne dense path matches bhtsne closely, and the fitsne FFT path settles at a slightly higher KL because the interpolation repulsion is coarser, while still separating the same structure.
