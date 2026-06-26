# LSH neighbor index for spectral t-SNE: status and plan

## Goal

Produce a 2D t-SNE embedding of the harmonized MS2 dataset (about 439,000 spectra) on CPU, using modified-cosine spectral similarity as the neighbor metric, fast enough to be practical and faithful enough that chemically related spectra land together. The neighbor stage must scale roughly sublinearly with the number of spectra, because exact modified-cosine neighbor search does not.

The quantity we ultimately care about is the **embedding**: whether the layout recovers chemical structure (for example NPClassifier classes forming coherent regions). The quantity we optimize against during development is **recall at k of the LSH neighbor sets versus the exact modified top-k**, because the embedding is built from those neighbor sets.

Canonical configuration (from the harmonized t-SNE pipeline):

- `mz_power = 0.0`, `intensity_power = 0.25`, `mz_tolerance = 0.02`
- perplexity 30, so `k = 3 * perplexity = 90` neighbors per spectrum
- sketch backend `MinHash<u32, 128>`, default 16 bands

## Background: the pipeline and the pieces

1. **Modified cosine.** Matches a query peak to a library peak either at the same m/z (direct match) or shifted by the precursor mass difference (neutral-loss or analog match). The shifted matching is the entire reason modified cosine exists: it connects a compound to its chemical analogs, which plain cosine misses.
2. **Exact index.** `FlashCosineIndex` is an inverted index over m/z buckets that computes the exact modified top-k. It is exact and parallel but its per-query cost grows with the library size.
3. **LSH sketch index.** `FlashCosineSketchIndex<P, K>` (in `src/structs/flash_cosine_index.rs`, module `sketch_index`) summarizes each spectrum with a MinHash sketch over a bucket set, bands the sketch into LSH tables, retrieves the spectra that collide in any band, and exactly rescores only those candidates. The bucket keys (in `src/traits/sketcher.rs`) are, per peak, an absolute m/z bucket and a precursor-relative neutral-loss bucket `precursor - mz`, each at width `2 * mz_tolerance`. The two key spaces are banded separately, so a collision in either space generates a candidate.
4. **t-SNE.** `src/tsne.rs` cleans each spectrum with `SiriusMergeClosePeaks`, builds the matching index, collects each spectrum's top-k neighbors in parallel, and feeds them to bhtsne. `NeighborSearch` selects `Exact`, `Approximate`, or `Lsh { bands }`.
5. **bhtsne.** Now pinned to git master (`frjnn/bhtsne`, version 0.7.2) instead of the crates.io 0.5.10, for the Morton linear quadtree and parallel Barnes-Hut. This made the fit about 15x faster per epoch.

## Current situation (measured)

All numbers below are from the real harmonized dataset on a 64-thread machine, with the exact modified top-k as ground truth. Reproduce with the examples named in each section.

### bhtsne is no longer the bottleneck

Full 439k embedding, 500 epochs, using LSH neighbors (`examples/spectral_tsne_lsh_timing.rs`, `spectral_tsne_lsh_npc_plot.rs`):

| phase | time |
|---|---|
| clean | 0.8 s |
| LSH index build | 8 to 9 s |
| LSH neighbor search | 2.8 s |
| bhtsne fit (500 epochs) | 51 s (0.10 s per epoch) |
| total | about 64 s |

Per-epoch fit dropped from 1.56 s on bhtsne 0.5.10 to 0.10 s on 0.7.2, about 15x. The fit is now a minor cost.

### Exact neighbor search does not scale

`examples/spectral_tsne_stratified_control.rs` and `lsh_vs_exact_metrics.rs`:

| library | exact neighbor search |
|---|---|
| 10,000 | 0.78 s |
| 50,000 | 15 to 23 s |
| 439,403 | about 60 minutes (361 s for the first 10% of queries) |

The search is fully parallel (`collect_neighbors` uses `into_par_iter` across queries, 64 threads, independent per-query state). At 439k it is about 0.53 s of single-core CPU per query, so single-threaded it would be roughly 64 hours. The cost is genuine compute: each query rescores every spectrum sharing any m/z or neutral-loss bucket, which is a large and growing fraction of the library. This is O(N^2)-ish and parallelism only buys a flat 64x. Exact is fine at 50k (about 20 s end to end with the fast fit) and unusable at full scale.

### The LSH index is fast but produces a degenerate embedding

Controlled A/B on an identical stratified 50k subset (`spectral_tsne_stratified_control.rs`, `MODE=both`):

| | neighbor search | fit | embedding |
|---|---|---|---|
| exact | 15.1 s | 4.4 s | clean: coherent NPC-class regions, a separated lipid island, no artifacts |
| lsh (16 bands) | 0.17 s | 6.2 s | degenerate: straight-line filaments, fragmentation, large voids |

Same spectra, same pipeline, same bhtsne, only the neighbor source differs. So the metric, the pipeline, and bhtsne are sound. The LSH neighbors are the problem.

### Ground-truth metrics expose uniform under-retrieval

`examples/lsh_vs_exact_metrics.rs` (2000 sampled queries against the full 439k library, exact truth computed once):

- Every query has a full 90-neighbor exact set. `exact neighbors per query` is 90 at every percentile. So there are **no sparse spectra and no singletons**: modified cosine, through neutral-loss matching, connects every spectrum to at least 90 others.
- The default LSH (16 bands, 8 rows per band) retrieves a **median of 2 candidates** per query, and recall at 90 is **0.048**. 97% of queries retrieve fewer than k candidates. 0% retrieve zero.

This corrected an earlier wrong hypothesis (that the failure was sparse spectra and empty-row padding). It is not. It is uniform, severe under-retrieval.

### Loosening rows per band recovers recall but explodes candidates

Sweep over band count (more bands means fewer rows per band, a looser collision threshold), same 2000 queries and full library:

| bands | rows/band | recall mean/median/p90 | recall >= 0.5 | candidates/q median (% of library) |
|---|---|---|---|---|
| 16 | 8 | 0.05 / 0.01 / 0.12 | 1.4% | 2 (0.0005%) |
| 32 | 4 | 0.23 / 0.12 / 0.63 | 15.7% | 78 (0.02%) |
| 64 | 2 | 0.76 / 0.87 / 1.00 | 84.0% | 20,142 (4.6%) |
| 128 | 1 | 0.97 / 1.00 / 1.00 | 99.5% | 197,979 (45%) |

Recall does climb from 0.05 to 0.97, so the collision threshold was the binding constraint. But the threshold is non-selective: to catch the low-Jaccard true neighbors you must drop to 1 or 2 rows per band, and then the candidate set explodes to 4.6% (2 rows) or 45% (1 row) of the library per query. At 1 row this is just faster brute force, and the candidate count grows with the library, so it returns to O(N) per query and O(N^2) total. Loosening alone does not scale.

## Root cause

Bucket-set Jaccard is a weak proxy for modified cosine, so the true neighbors sit at low bucket-Jaccard (about 0.1 to 0.3), and a single global threshold cannot separate them from unrelated spectra.

Three reasons the proxy is weak:

1. **Modified cosine is not set Jaccard.** For an analog (precursor and fragments shifted by the modification mass), the shifted fragments share only the neutral-loss keys, not the m/z keys, and any unshifted fragments share only the m/z keys, not the neutral-loss keys. So per matched peak only one of its roughly three keys in the union is shared, which caps the Jaccard near 1/3 however high the modified cosine. Measured in `examples/minhash_analog_behavior.rs`: a pure analog has cosine about 0.005, modified cosine about 1.0, and key Jaccard 0.334.
2. **Real spectra are diverse.** Even direct neighbors lose bucket overlap to boundary straddling and intensity differences, pulling Jaccard down further.
3. **Intensity is ignored.** The sketch is an unweighted set of buckets, while modified cosine weights peaks by `intensity^0.25`. Even when enough candidates are retrieved (the 2.7% of queries with at least k candidates at 16 bands), recall is only 0.37, so the retrieved set is mis-ranked relative to the metric, not just too small.

The earlier synthetic validation (recall and purity looked great) was misleading because the synthetic data was dense near-duplicate clusters where every spectrum had about 63 near-identical neighbors at Jaccard about 0.88, the index's best case. Real data has no such structure.

## What has been tried

1. **Two-space banding** (band m/z and neutral-loss key spaces separately, candidate on collision in either). Fixed analog non-retrieval on synthetic planted analogs (recall went from about 0 to 1.0 at the default banding). Necessary but not sufficient on real data, because the Jaccard is still low.
2. **Word width and permutation choices.** `u32` is Pareto-optimal versus `u64` (identical recall at half the signature bytes). Recall is driven by permutation count and band count, not word width.
3. **Rows-per-band loosening.** Recovers recall but explodes the candidate set, as shown above. Does not scale.
4. **Exact neighbors.** Correct, produces a clean embedding, but about 60 minutes at 439k. Usable only up to roughly 50k.

## Hypotheses: what could work

### Primary: multi-resolution layered index

Band at several bucket tolerances and union the candidates, while always rescoring at the true 0.02 tolerance.

Mechanism: a coarser bucket (larger tolerance) raises the bucket-Jaccard of genuinely related spectra more than it raises it for unrelated spectra. Related peaks systematically fall into the same wide bucket even when they straddle a fine bucket boundary or match only loosely, while unrelated peaks are scattered across different m/z regions and gain fewer shared coarse buckets. So a coarse layer can use a tighter threshold (more rows per band) and still retrieve true neighbors, which is the selectivity the single-resolution threshold lacks. False positives from the coarse layer are harmless because the final exact rescore at 0.02 gives them low scores.

Proposed API addition to `FlashCosineSketchIndex` (in `src/structs/flash_cosine_index.rs`):

```rust
pub fn build_with_layers<S>(
    spectra: &[S],
    mz_power: f64,
    intensity_power: f64,
    score_tolerance: f64,        // inner FLASH rescore tolerance, the true metric (0.02)
    layers: &[(f64, usize)],     // (sketch_tolerance, bands) per layer
) -> Result<Self, FlashCosineIndexError>;
```

The inner `FlashCosineIndex` (for rescore) stays at `score_tolerance`. Each layer builds the two-space sketches at its own `sketch_tolerance` and bands them. A query bands its sketch at each layer's tolerance, unions the candidate set across all layers, then rescores at `score_tolerance`. `build_with_bands` becomes the single-layer special case where `sketch_tolerance == score_tolerance`. The decoupling of sketch tolerance from score tolerance is the essential part.

Concrete hypothesis to falsify: a small stack of layers (for example `[(0.02, 16), (0.1, 16), (0.5, 16)]`) reaches recall around 0.9 at a few thousand candidates per query, rather than the 200,000 candidates that 1-row banding needs for the same recall.

### Secondary: intensity-weighted sketch

The set Jaccard ignores intensity, but modified cosine weights peaks by `intensity^0.25`. Weight the buckets by intensity (repeat a bucket key proportionally to its quantized weight, or use a weighted MinHash) so the sketch similarity tracks the modified-cosine numerator rather than raw set overlap. This targets the retrieval-quality ceiling (the 0.37 recall when candidates are adequate), which layering alone may not fix.

### Tertiary and cleanups

- **Per-query candidate floor with brute-force top-up.** If a query retrieves fewer than some threshold of candidates, widen its search or top up with a bounded brute-force pass, so no query is starved. Bounds the worst case.
- **Precursor-bucket prefilter** to cut coarse-layer false positives cheaply.
- **More permutations** for finer threshold control (does not address the Jaccard mismatch by itself).
- **Remove the empty-row anchor in `neighbor_row`** (`src/tsne.rs`). When a row is padded, a fully empty row is anchored to arbitrary low indices, which fabricates filaments. Under-retrieval is the dominant cause of the bad embedding, but this anchoring should be removed regardless because it is an artifact generator.

## How to evaluate

### Harness

`examples/lsh_vs_exact_metrics.rs` is the comparator. It cleans the spectra (matching the embedding pipeline), builds the exact index, computes the exact top-k ground truth once, then for each candidate-generation config (currently a band sweep, to be extended to layer stacks) runs the LSH search over the same library and reports recall and candidate-volume distributions. Run:

```text
LIBRARY=0 QUERIES=2000 K=90 BANDS=16,32,64,128 \
  cargo run --release --example lsh_vs_exact_metrics --features rayon,minhash
```

### Metrics and what counts as success

1. **Recall at k versus exact modified top-k**, reported as a distribution: mean, median, p90, and the fraction of queries at or above 0.5 (and 0.8). Report the distribution, never just the mean. A mean hides the zero-recall mass that wrecked us before.
2. **Candidates rescored per query**, as a distribution and as a fraction of the library. This is the cost and the scalability signal. It must stay roughly flat as the library grows.
3. **Scaling check.** Run the metrics at increasing library sizes (50k, 100k, 439k) and confirm that candidates per query stays roughly constant (true sublinearity) while recall holds. A config whose candidate count grows with N is faster brute force, not a real index.
4. **End-to-end embedding.** The decisive test. Run the winning config through `examples/spectral_tsne_stratified_control.rs` on a stratified 50k subset and plot colored by NPClassifier class. It must visually match the exact 50k control (coherent class regions, a separated lipid island, no straight-line filaments). The exact control is the quality target.

### Acceptance criteria

A config is acceptable when, on the full 439k library, recall at 90 is at least about 0.8 with candidates per query bounded (low thousands, not growing with N), and the end-to-end stratified embedding is visually indistinguishable in structure from the exact control. Speed should remain in the seconds range for the full neighbor stage.

### Methodology guardrails (lessons learned)

- Validate on real data, not on synthetic clusters engineered to be the index's best case.
- Report distributions, not means.
- Treat candidates per query below k, or candidate counts that grow with the library, as failure signals, not acceptable tradeoffs.
- Look at the actual embedding, not only proxy metrics. The first real-data embedding revealed the failure that every synthetic proxy metric had hidden.

## Reference: example programs

- `examples/lsh_vs_exact_metrics.rs`: exact-vs-LSH recall and candidate metrics on real data (the primary comparator).
- `examples/minhash_analog_behavior.rs`: why the bucket Jaccard mismatches modified cosine for analogs.
- `examples/spectral_tsne_stratified_control.rs`: stratified subset, exact vs LSH embeddings for the visual control.
- `examples/spectral_tsne_lsh_timing.rs`: per-phase timing of the full pipeline, exact vs LSH.
- `examples/spectral_tsne_lsh_npc_plot.rs`: full-dataset LSH embedding to CSV for plotting.
- `examples/tsne_recall_effect.rs`: how neighbor recall maps to the t-SNE affinity matrix (synthetic).
