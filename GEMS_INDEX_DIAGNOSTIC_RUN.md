# GeMS Index Diagnostic Run

Use this note for the next exploratory GeMS run when the one-shot cosine index
does not speed up as expected. The goal is to measure where index time is spent,
not to produce the graph or train embeddings.

Run the same graph parameters as the slow job: cosine one-shot self-similarity,
`top_k = 4`, score threshold `0.95`, PEPMASS tolerance `10 Da`, and the same
top-40 spectra cache. Keep the index precision at `f32`.

In the downstream graph builder, temporarily replace the normal row iterator:

```rust
index.rows().ids(&row_ids).into_par_iter()
```

with the diagnostic iterator:

```rust
index.rows().ids(&row_ids).with_diagnostics().into_par_iter()
```

Each item is then a `FlashCosineSelfSimilarityRow` with `query_id`, `hits`, and
`diagnostics`. Aggregate at least these counters over a representative sample of
rows, ideally a strided sample of 10k to 100k rows across the whole GeMS index:

```rust
product_postings_visited
spectrum_block_bound_entries_visited
candidates_marked
candidates_rescored
results_emitted
spectrum_blocks_evaluated
spectrum_blocks_allowed
spectrum_blocks_pruned
```

Also log total sampled rows, total emitted hits, total graph-build wall time for
the sampled rows, and rows per second. Do not include training time in this run.

Interpretation:

- High `spectrum_block_bound_entries_visited` with low postings/candidates means
  block-bound preparation dominates.
- High `product_postings_visited` means PEPMASS/block pruning is not selective
  enough for the real data.
- High `candidates_rescored` means exact candidate scoring dominates.
- Low index counters but high wall time means the bottleneck is outside these
  index phases, such as row scheduling, graph aggregation, or downstream code.

For comparison against the synthetic benchmark, this command exercises the same
shape locally:

```bash
env INDEX_SEARCH_TOP_K_LIBRARY_SIZE=1000000 \
    INDEX_SEARCH_TOP_K_QUERY_COUNT=10000 \
    INDEX_SEARCH_TOP_K=4 \
    INDEX_SEARCH_TOP_K_THRESHOLDS=0.95 \
    INDEX_SEARCH_TOP_K_PEPMASS_TOLERANCE=10 \
    INDEX_SEARCH_TOP_K_METRICS=cosine \
    INDEX_SEARCH_TOP_K_PRECISIONS=f32 \
    INDEX_SEARCH_TOP_K_SAMPLE_SIZE=10 \
    cargo bench --bench index_top_k_large --features rayon
```

The useful output lines start with `index_top_k_large diagnostics`. Compare the
per-query synthetic counters with the GeMS sampled counters before deciding on
the next optimization.
