# Flash Similarity Index Performance Improvement Plan

This document is the working engineering plan for improving exact indexed
thresholded top-k search in `mass_spectrometry` for both cosine and spectral
entropy similarity. The immediate measured target is not a 32-query smoke
benchmark. The current development target is a large indexed cosine
self-similarity workload:

```text
Library size: 1,000,000 spectra
Query count: 10,000 indexed self-queries
Top k: 16
Threshold: 0.9
Benchmark: library_search_cosine_top_k_large/threshold_index_indexed_top_k/0.900
Criterion baseline: 1m-10k-current
Baseline time: [24.221 s 24.378 s 24.559 s]
```

The eventual production pressure is closer to `1M x 1M` and, later, tens of
millions of nodes. The downstream graph still needs thresholded top-k
neighbors, not every threshold edge. At that scale, rayon only multiplies
whatever single-query algorithm we already have. The main objective is therefore
to reduce postings visited, candidates generated, and candidates exactly
rescored per query.

## Search Contract

The active target is exact thresholded top-k search:

```text
search_top_k_threshold(query, k, threshold)
```

Every returned result must have an exact score at least `threshold`, and no
spectrum whose exact score is at least `threshold` and belongs in the top `k`
may be missed. The live pruning lower bound is therefore
`max(threshold, current_kth_score)`.

Parallelism is not an index experiment in this roadmap. Production callers can
and should parallelize across independent queries with rayon and per-worker
scratch state. The index work here must improve the single-query exact
algorithm first.

The following are outside the active roadmap unless explicitly reintroduced:

- Threshold-only graph traversal that emits all edges above threshold.
- Approximate candidate generation that cannot prove exact top-k recovery.
- Batch sparse-matrix scoring that changes the query-level execution model.
- Rayon inside the index as a substitute for pruning.

## Metric Parity Contract

Cosine and entropy must move together. A new index architecture can be
implemented for one metric first, but it is not considered complete until the
other metric has either the same capability or a documented reason why the same
bound is mathematically invalid.

To keep duplication low, every experiment should split into two layers:

- Shared traversal and scratch infrastructure in `flash_common` or a small
  private helper module.
- Metric-specific bound functions in the cosine and entropy modules.

The public surface should remain parallel where the algorithms support it:
threshold search, top-k threshold search, low-allocation emitter APIs, and
indexed self-query APIs for thresholded top-k graph construction. If one metric
lacks an indexed self-query API for this exact thresholded top-k workload, that
is a parity gap and should be fixed before tuning that metric's large graph
path.

## Benchmark Contract

The cosine benchmark that currently matters for this roadmap is:

```bash
INDEX_SEARCH_TOP_K_LIBRARY_SIZE=1000000 \
INDEX_SEARCH_TOP_K_QUERY_COUNT=10000 \
INDEX_SEARCH_TOP_K=16 \
INDEX_SEARCH_TOP_K_THRESHOLDS=0.9 \
INDEX_SEARCH_TOP_K_SAMPLE_SIZE=10 \
cargo bench --bench index_top_k_large -- \
  --baseline 1m-10k-current \
  threshold_index_indexed_top_k/0.900
```

Create or refresh the baseline with the same command and
`--save-baseline 1m-10k-current`.

The entropy counterpart exists as metric modes in `index_top_k_large`. Use
`INDEX_SEARCH_TOP_K_METRICS=entropy-weighted`, `entropy-unweighted`, `entropy`,
or `all`. Entropy uses the same synthetic data generator and diagnostic output.
Weighted and unweighted entropy are both covered, with weighted entropy treated
as the main user-facing mode.

Required entropy target shape:

```text
Library size: 1,000,000 spectra
Query count: 10,000 indexed or reusable self-queries
Top k: 16
Thresholds: 0.75 and 0.9
Metrics: weighted entropy and unweighted entropy
Diagnostics: same counters as cosine, plus any entropy-specific bound counters
```

Keep the 32-query run only as a fast smoke test. Do not use it to make decisions
about index architecture. A change is only considered a performance improvement
if it improves the relevant 1M x 10k filtered benchmark and the diagnostics
explain why.

For spectrum-block size experiments, build the benchmark with the private
`experimental_block_size_env` feature and set
`MASS_SPECTROMETRY_FLASH_SPECTRUM_BLOCK_SIZE`. This is an internal benchmark
hook, not a public API.

Every benchmark comparison must record:

- Criterion timing and percent change.
- `product_postings_visited / query`.
- `prefix_postings_visited / query`.
- `candidates_marked / query`.
- `secondary_candidates_marked / query`.
- `candidates_rescored / query`.
- `results_emitted / query`.
- Metric-specific bound counters, when an experiment adds such a bound.
- Index construction time and resident memory, when the index layout changes.

## Current Index State

`FlashCosineThresholdIndex` is currently an exact threshold-specialized cosine
index. For high-threshold indexed top-k queries it does the following:

1. Uses precomputed threshold prefix m/z peaks and prepared normalized query
   weights for the indexed query.
2. Uses the shared spectrum-block upper-bound index to select only blocks that
   can still reach `max(threshold, current_kth_score)`.
3. Scans block-local product postings in query suffix-bound order.
4. Exact-scores each touched candidate on first encounter and updates the top-k
   lower bound.
5. Stops when the remaining query suffix bound proves that no unseen candidate
   in the allowed blocks can enter the thresholded top-k result.

`FlashEntropyIndex` is exact for spectral entropy and exposes threshold and
top-k threshold APIs, including indexed self-query variants for thresholded
top-k graph workloads. It uses an additive per-query entropy upper bound through
`entropy_peak_upper_bound` and `entropy_raw_threshold`, and now shares the same
spectrum-block upper-bound auxiliary index used by cosine for high-threshold
paths. Its current high-threshold path is fastest with a block-restricted exact
accumulator rather than per-candidate dynamic rescoring.

The current code already has several useful pieces:

- A reusable `SearchState` and `TopKSearchState`.
- Shared Flash inverted product and neutral-loss indexes.
- Threshold-prefix postings for the cosine threshold-specialized index.
- A shared spectrum-block upper-bound index with metric-specific contribution
  functions for cosine and entropy.
- Shared block-local product postings, used after the block upper-bound filter
  has reduced the eligible spectrum-id blocks.
- A block-restricted exact accumulator for high-threshold block-pruned paths.
  Once a small set of spectrum blocks is allowed, exact direct scores are
  accumulated directly from block-local postings instead of marking candidates
  and rescoring each spectrum with a second two-pointer pass.
- A cosine-specific dynamic top-k scorer over allowed spectrum blocks. It visits
  candidate spectra in query suffix-bound order, exact-scores each candidate on
  first encounter, and stops when no unseen candidate can beat the current top-k
  floor.
- Diagnostics counters for postings, candidates, rescoring, and emitted results.
- A dynamic exact rescoring bound based on the current top-k floor.
- A 1M x 10k Criterion benchmark target in `benches/index_top_k_large.rs`.

The current architecture still materializes or accumulates candidates inside the
allowed spectrum blocks. That means it is closer to a block-pruned sparse join
than to a true document-at-a-time top-k query processor. The likely remaining
large wins require changing the internal index layout so the query can skip
whole lists, subblocks, or candidates before materialization. Future changes
must state whether the layout applies to cosine, entropy, or both.

## Experiments Already Tried

These experiments should not be retried without a materially different
hypothesis.

### Epoch-Cleared Candidate Bitset

The idea was to avoid clearing every touched candidate bit by using an epoch per
bitset word.

Result on 1M x 10k:

```text
time: [23.883 s 24.004 s 24.114 s]
change: [-2.4014% -1.5301% -0.7327%]
Criterion verdict: Change within noise threshold
```

Decision: rejected. The measured gain was too small and not robust enough to
justify the extra scratch memory.

### Fused Prefix-Intersection And Immediate Top-K Scoring

The idea was to score a candidate immediately when prefix intersection first
proved it viable, instead of building a secondary candidate list and rescoring it
afterward.

Result on 1M x 10k:

```text
time: [25.514 s 25.596 s 25.672 s]
change: [+4.1714% +4.9993% +5.7511%]
Criterion verdict: Performance has regressed
```

Decision: rejected. The fused path worsened locality and/or top-k floor dynamics.

## Experiment 0: Benchmark Diagnostics Upgrade

Status: implemented in `benches/index_top_k_large.rs` for cosine, weighted
entropy, and unweighted entropy.

This is the first implementation task before another index experiment. The large
benchmark prints per-query diagnostic rates for cosine and entropy paths.

Implementation:

- Add a small benchmark-local diagnostic accumulator in `index_top_k_large`.
- For each benchmarked path, compute totals and divide by `query_ids.len()`.
- Print one compact line per benchmark path after setup or expose the values via
  Criterion throughput metadata if feasible.
- Keep these diagnostics outside the timed loop if printing; keep accumulation
  inside the timed loop so the optimizer cannot erase diagnostic updates.

Acceptance:

- No public API changes.
- The cosine target reports per-query postings and candidates.
- The entropy target reports the same per-query diagnostics for weighted and
  unweighted entropy.
- The report is stable enough to compare before and after each experiment.

The diagnostic line is printed after Criterion finishes an executed benchmark
path. Filtered-out paths are suppressed, so targeted runs remain readable.

Expected speedup: none. This is instrumentation needed to make later work
scientific.

## Experiment 1: Exact Block-Max WAND Internal Index

This is the highest-priority exact architecture change. It maps the problem to
state-of-the-art top-k retrieval: avoid scoring documents whose maximum possible
score cannot beat the current lower bound.

Status: implemented as a shared spectrum-id block upper-bound prefilter for
cosine and entropy high-threshold paths. This is not full Block-Max WAND because
it still traverses product and prefix postings after selecting allowed spectrum
blocks. The cosine version is kept because the 1M x 10k benchmark confirms the
smaller-scale speedup. The entropy version now uses the same shared block
traversal and scratch layout with entropy-specific additive raw-score bounds.

Initial same-environment comparison on 100k x 100:

```text
without block prefilter:
  time: [38.056 ms 41.448 ms 44.548 ms]
  candidates_marked/q=8723.890
  candidates_rescored/q=2172.450

with spectrum-id block prefilter:
  time: [11.642 ms 11.850 ms 12.068 ms]
  candidates_marked/q=256.000
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 391.000
```

The criterion percentage printed during the first block run was ignored because
it compared against cached samples from a different benchmark environment.

Cosine 1M x 10k result for the implemented prefilter before block-aware
candidate marking:

```text
time: [3.5944 s 3.6131 s 3.6337 s]
baseline: [24.221 s 24.378 s 24.559 s]
speedup: about 6.7x
candidates_rescored/q=256.000
spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Cosine 1M x 10k result after block-aware candidate marking:

```text
time: [2.4258 s 2.4430 s 2.4618 s]
change from previous block-prefilter run: about 1.48x faster
speedup from original baseline: about 10.0x
product_postings/q=752.410
prefix_postings/q=85545.319
candidates_rescored/q=256.000
spectrum_blocks_allowed/q=1.000 out of 3907.000
```

The cosine path is no longer dominated by product-posting traversal. The next
cosine-specific bottleneck is library-prefix intersection, which still scans
about 85k prefix postings/query.

Cosine 1M x 10k result after candidate-local prefix verification and
block-local product postings:

```text
time: [1.2902 s 1.2985 s 1.3081 s]
change from previous block-aware run: about 1.88x faster
speedup from original baseline: about 18.8x
product_postings/q=752.410
prefix_postings/q=15121.018
candidates_rescored/q=256.000
spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: keep candidate-local prefix verification. It is exact, removes the
global prefix posting scan, reduces prefix work from about 85k/query to about
15k/query, and improves the decision benchmark by roughly 47% relative to the
previous implementation.

Cosine 1M x 10k result after replacing mark-and-rescore with block-restricted
exact accumulation:

```text
time: [1.0593 s 1.0651 s 1.0721 s]
change from candidate-local prefix run: about 1.22x faster
speedup from original baseline: about 22.9x
product_postings/q=18427.802
prefix_postings/q=0.000
candidates_marked/q=256.000
candidates_rescored/q=0.000
spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: keep block-restricted exact accumulation. It is still exact, but the
high-threshold path no longer needs the library-prefix intersection or a second
per-candidate exact rescore. Product postings increase because every matched
peak in the allowed block contributes to the exact score, but this is cheaper
than the previous candidate-local prefix verification plus 256 two-pointer
rescoring passes.

Entropy implementation: done for the current spectrum-block prefilter. The
shared block index stores the maximum prepared library intensity per m/z
bin/block. The entropy query contribution is
`entropy_pair(query_intensity, block_max_intensity)`, and pruning compares the
summed raw bound against `entropy_raw_threshold(score_threshold)`. This is safe
because `entropy_pair(q, d)` is monotone increasing in `d` for fixed positive
`q`; every emitted result is still produced from exact matched peak
contributions. Regression tests cover both external and indexed entropy top-k
threshold queries pruning a low-bound block without losing hits.

Entropy 1M x 10k result for the implemented prefilter before block-aware
candidate marking:

```text
weighted entropy, threshold 0.75:
  time: [22.259 s 23.023 s 23.953 s]
  product_postings/q=1088613.895
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

weighted entropy, threshold 0.9:
  time: [22.190 s 22.258 s 22.319 s]
  product_postings/q=1088613.895
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.75:
  time: [22.171 s 22.411 s 22.711 s]
  product_postings/q=1088613.895
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.9:
  time: [22.075 s 22.268 s 22.483 s]
  product_postings/q=1088613.895
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: keep the shared block prefilter because it is exact and sharply
reduces rescoring. It is not sufficient for entropy at 1M x 10k because the
current candidate marking still scans every product posting in the active m/z
windows and only then checks whether the spectrum block is allowed. The next
entropy-relevant step is therefore block-aware postings traversal: candidate
generation must skip postings from disallowed spectrum blocks instead of
counting them in `product_postings/q`.

Entropy 1M x 10k result after block-aware candidate marking:

```text
weighted entropy, threshold 0.75:
  time: [5.4074 s 5.4182 s 5.4305 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

weighted entropy, threshold 0.9:
  time: [5.4091 s 5.4265 s 5.4490 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.75:
  time: [5.4056 s 5.4201 s 5.4339 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.9:
  time: [5.4072 s 5.4190 s 5.4321 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: keep block-aware candidate marking. It is exact, shared by cosine and
entropy, and reduces entropy time by about 4.1x from the previous block-prefilter
implementation. The remaining entropy cost is mostly block-local spectrum-window
checks plus exact rescoring of 256 candidates/query.

Entropy 1M x 10k result after replacing per-spectrum block checks with
block-local product postings:

```text
weighted entropy, threshold 0.75:
  time: [4.6239 s 4.6366 s 4.6503 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

weighted entropy, threshold 0.9:
  time: [4.6304 s 4.6413 s 4.6533 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.75:
  time: [4.7075 s 4.7349 s 4.7642 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.9:
  time: [4.6891 s 4.7075 s 4.7281 s]
  product_postings/q=9591.782
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: keep the shared block-local product posting layout. The diagnostic
count is unchanged on this synthetic benchmark because each selected block-local
window still contributes the same number of matching peak postings, but the
implementation no longer performs a binary search in every spectrum in the
allowed block. That removes enough hot-path overhead for a measured 13-14%
entropy improvement.

Entropy 1M x 10k result after replacing mark-and-rescore with block-restricted
exact accumulation:

```text
weighted entropy, threshold 0.75:
  time: [3.9228 s 3.9378 s 3.9530 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

weighted entropy, threshold 0.9:
  time: [3.9025 s 3.9222 s 3.9406 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.75:
  time: [3.8702 s 3.8880 s 3.9101 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

unweighted entropy, threshold 0.9:
  time: [3.8611 s 3.8653 s 3.8698 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: keep the same exact accumulator path for entropy. It gives a further
15-18% improvement over the block-local product-posting run and removes the
separate `candidates_rescored/q=256` work for both weighted and unweighted
entropy.

After these targeted passes, both metrics still consider 256 spectra/query in
the allowed block. Entropy remains fastest with one exact block-local accumulator
pass. Cosine is now fastest with a dynamic top-k candidate scorer that only scans
enough block-local postings to prove no unseen candidate can enter the result
set, then exact-scores the touched spectra. The next material architecture step
is therefore a stronger bound that can skip spectra inside the allowed block,
not just shorten the query-prefix scan. Full Block-Max WAND, candidate-level
upper bounds, or impact-ordered intra-block postings are the next reasonable
candidates.

Cosine 1M x 10k result after adding dynamic allowed-block top-k candidate
scoring:

```text
time: [972.05 ms 972.86 ms 974.06 ms]
change from block-restricted exact accumulation: about 1.09x faster
speedup from original baseline: about 25.1x
product_postings/q=256.000
prefix_postings/q=0.000
candidates_marked/q=256.000
candidates_rescored/q=256.000
spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: keep this path for cosine top-k on the threshold-specialized index.
The query suffix bound fills the top-k set quickly on the clustered benchmark
and stops after the first ordered query peak, reducing product posting work from
about 18.4k/query to 256/query. It is still exact because every emitted result
is produced by the existing per-spectrum direct scorer.

### Spectrum-Block Size Sweep

The first intra-block experiment was a sweep of spectrum-id block size. Smaller
blocks did not help the clustered 1M x 10k cosine target because each synthetic
cluster contains 256 near-duplicates. Splitting a cluster across smaller blocks
kept the same 256 candidates but multiplied block-bound traversal work.

Cosine indexed threshold top-k, `k=16`, threshold `0.9`:

```text
block size 32:
  time: [3.3813 s 3.3954 s 3.4113 s]
  candidates_marked/q=256.000
  spectrum_blocks_allowed/q=8.000 out of 31250.000

block size 64:
  time: [1.9365 s 1.9416 s 1.9480 s]
  candidates_marked/q=256.000
  spectrum_blocks_allowed/q=4.000 out of 15625.000

block size 128:
  time: [1.3334 s 1.3380 s 1.3435 s]
  candidates_marked/q=256.000
  spectrum_blocks_allowed/q=2.000 out of 7813.000

block size 256:
  previous default, about 972.86 ms after dynamic candidate scoring
  candidates_marked/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

block size 512:
  time: [842.18 ms 846.91 ms 851.95 ms]
  candidates_marked/q=262.979
  spectrum_blocks_allowed/q=1.000 out of 1954.000

block size 1024:
  time: [764.53 ms 766.72 ms 769.58 ms]
  candidates_marked/q=277.772
  spectrum_blocks_allowed/q=1.000 out of 977.000

block size 2048:
  time: [770.55 ms 772.50 ms 774.77 ms]
  candidates_marked/q=311.291
  spectrum_blocks_allowed/q=1.032 out of 489.000

block size 4096:
  time: [13.516 s 13.563 s 13.615 s]
  candidates_marked/q=5364.863
  spectrum_blocks_allowed/q=30.962 out of 245.000
```

Decision: change the cosine threshold-index default block size to 1024. This
improves the main cosine top-k target from `972.86 ms` to `766.72 ms`, roughly
1.27x faster, while the indexed threshold-then-sort path also improved from
`[1.1078 s 1.1111 s 1.1149 s]` at block size 256 to
`[1.0579 s 1.0632 s 1.0678 s]` at block size 1024. The default build without
the experimental feature confirmed the new cosine top-k default at
`[761.98 ms 765.39 ms 769.40 ms]`.

Weighted entropy, indexed threshold top-k, threshold `0.9`:

```text
block size 128:
  time: [5.4930 s 5.5075 s 5.5261 s]
  candidates_marked/q=256.000
  spectrum_blocks_allowed/q=2.000 out of 7813.000

block size 256:
  time: [3.8499 s 3.8565 s 3.8639 s]
  candidates_marked/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000

block size 512:
  time: [3.8132 s 3.8222 s 3.8318 s]
  candidates_marked/q=473.491
  spectrum_blocks_allowed/q=1.000 out of 1954.000

block size 1024:
  time: [3.8926 s 3.9041 s 3.9178 s]
  candidates_marked/q=911.031
  spectrum_blocks_allowed/q=1.000 out of 977.000
```

Decision: keep the entropy default block size at 256. The 512 result is only a
noise-level improvement and doubles the number of accumulated candidates; 1024
regresses.

### Candidate-Level Upper Bound Attempt

The next intra-block attempt was a cosine-only candidate bound inside the
dynamic allowed-block top-k path. On first candidate touch, the path computed
the contribution of the first matched query peak plus a Cauchy bound over the
remaining query suffix:

```text
upper_raw = first_matched_raw + remaining_query_norm * library_norm
target_raw = current_top_k_floor * query_norm * library_norm
```

Candidates with `upper_raw < target_raw` were skipped before calling the exact
two-pointer scorer. A targeted regression test confirmed the bound was safe and
could reduce exact scoring on constructed weak candidates.

Cosine 1M x 10k result, `k=16`, threshold `0.9`:

```text
time: [780.05 ms 787.74 ms 795.94 ms]
change from the current default: about 2.9% slower
product_postings/q=277.772
candidates_marked/q=277.772
candidates_rescored/q=272.556
spectrum_blocks_allowed/q=1.000 out of 977.000
```

Decision: reject. The bound only avoided about 5.2 exact rescoring calls/query,
and the extra branch/arithmetic overhead made the large benchmark slower. A
candidate-level bound needs to be much stronger, or it needs to be integrated
into a real document-at-a-time WAND traversal where the bound avoids candidate
materialization rather than adding work after candidate materialization.

### Impact-Ordered Intra-Block Attempt

The next exact attempt added a private m/z-bin/spectrum-block posting layout
ordered by normalized library peak contribution. The cosine top-k path used the
existing spectrum-block filter, then scanned the selected block's bin postings
from highest to lowest impact. For each posting list it stopped when:

```text
query_peak_weight * current_library_weight + query_suffix_norm
    < current_top_k_floor
```

Every emitted candidate was still scored by the existing exact two-pointer
cosine scorer. A targeted regression test confirmed that this can skip weak
candidates inside an allowed spectrum block.

Cosine 1M x 10k result, `k=16`, threshold `0.9`:

```text
time: [795.09 ms 799.60 ms 803.91 ms]
change from the current default: about 1.5% slower
product_postings/q=284.531
candidates_marked/q=269.238
candidates_rescored/q=269.238
spectrum_blocks_allowed/q=1.000 out of 977.000
```

Decision: reject. The layout only avoided about 8.5 candidates/query relative
to the current default (`277.772 -> 269.238`) while increasing posting work and
adding another full auxiliary index. A future impact-ordered experiment needs a
materially stronger traversal, such as a true multi-list impact/WAND loop, not a
per-bin intra-block scan layered after the current block filter.

### Block-Local WAND Attempt

The next exact attempt added block-local document-at-a-time posting lists. Each
m/z bin stored one spectrum-id-sorted posting list per spectrum block, with a
conservative maximum normalized library peak weight for that list. The cosine
top-k path used the existing spectrum-block filter, then ran a WAND pivot loop
over the active bin/block lists with:

```text
lower_bound = max(user_threshold, current_kth_score)
list_bound = normalized_query_peak_weight * max_normalized_library_peak_weight
```

Every pivot candidate was still scored by the existing exact cosine scorer. A
targeted regression test confirmed that this can skip weak spectra inside an
allowed block on a constructed example.

Cosine 1M x 10k result, `k=16`, threshold `0.9`:

```text
time: [10.485 s 10.505 s 10.529 s]
change from the current default: about 13.1x slower
product_postings/q=20042.359
candidates_marked/q=256.000
candidates_rescored/q=256.000
spectrum_blocks_allowed/q=1.000 out of 977.000
```

Decision: reject. The WAND loop did not reduce rescoring below the already
cheap dynamic query-order path, and it replaced a short high-impact prefix scan
with many cursor advances and repeated active-list sorting. Do not retry this
block-local WAND shape unless the cursor scheduler is fundamentally different
and the benchmark target shows a much larger candidate-reduction opportunity.

Entropy rejected the same dynamic candidate scorer:

```text
weighted entropy, threshold 0.75:
  time: [4.6910 s 4.7034 s 4.7161 s]
  change from block-restricted exact accumulation: about 19.4% slower

weighted entropy, threshold 0.9:
  time: [4.6864 s 4.7008 s 4.7161 s]
  change from block-restricted exact accumulation: about 19.9% slower

unweighted entropy, threshold 0.75:
  time: [4.6680 s 4.6906 s 4.7150 s]
  change from block-restricted exact accumulation: about 20.6% slower

unweighted entropy, threshold 0.9:
  time: [4.6970 s 4.7035 s 4.7101 s]
  change from block-restricted exact accumulation: about 21.7% slower

diagnostics:
  product_postings/q=9591.782
  candidates_marked/q=256.000
  candidates_rescored/q=256.000
  spectrum_blocks_allowed/q=1.000 out of 3907.000
```

Decision: reject for entropy and keep the block-restricted exact accumulator.
Entropy pair scoring makes the 256 per-spectrum direct rescoring passes more
expensive than accumulating exact raw scores from block-local postings. After
reverting to the accumulator, the weighted entropy `0.9` target measured
`[3.8499 s 3.8565 s 3.8639 s]` with `candidates_rescored/q=0`.

The relevant retrieval literature is WAND and Block-Max WAND. WAND uses dynamic
upper bounds to avoid full evaluation of documents that cannot enter top-k.
Block-Max WAND strengthens that idea with per-block upper bounds stored in the
inverted index. The mass-spectrometry mapping is:

```text
IR document         -> spectrum
IR term             -> query peak m/z window or an m/z bucket covering it
term impact         -> metric-specific peak contribution
document score      -> cosine or entropy similarity
top-k threshold     -> max(user_threshold, current kth score)
full evaluation     -> exact score computation with current matching semantics
```

### Exactness Requirement

The WAND path must be a safe candidate-pruning layer only. Every emitted result
must still be produced by the existing exact score computation. False positives
are acceptable; false negatives are not.

Use normalized contribution bounds for pruning:

```text
q_i_norm = q_i / ||q||
d_j_norm = d_j / ||d||
cosine(q, d) = sum matched(q_i_norm * d_j_norm)
```

For an m/z bucket or block, store a conservative upper bound on `d_j_norm`. For a
query peak, the per-block bound is `q_i_norm * block_max_d_norm`. Summing those
bounds over active query peaks gives a safe upper bound on the final cosine.

If the bound is below `max(score_threshold, current_kth_score)`, the block or
candidate cannot enter the result set and can be skipped.

For entropy, use additive raw-score bounds:

```text
raw_entropy(q, d) = sum matched entropy_pair(q_i, d_j)
score_entropy(q, d) = raw_entropy(q, d) / 2
target_raw = 2 * max(score_threshold, current_kth_score)
block_bound = sum entropy_pair(q_i, block_max_intensity)
```

The shared block index should store only metric-neutral addresses and maxima.
The metric module should provide the per-peak contribution function and the
score-to-raw threshold conversion.

### Concrete Index Layout

Add private auxiliary structs; do not expose them publicly until they prove
useful.

Suggested shared internal shape:

```rust
struct BlockUpperBoundIndex<P, B> {
    bin_width: f64,
    bins: Vec<BlockUpperBoundBin<P, B>>,
}

struct BlockUpperBoundBin<P, B> {
    postings: Vec<BlockUpperBoundPosting<P>>, // sorted by spectrum_id
    blocks: Vec<BlockUpperBoundBlock<B>>,
}

struct BlockUpperBoundPosting<P> {
    spectrum_id: u32,
    mz: P,
    value: P,
}

struct BlockUpperBoundBlock<B> {
    start: u32,
    end: u32,
    max_bound_value: B,
    max_spectrum_id: u32,
}
```

For cosine, `value` is normalized peak weight. For entropy, `value` is prepared
library intensity. These values are for pruning only. Exact rescoring must still
use the existing per-spectrum m/z and data slices.

Start with these constants:

```text
bin_width = mz_tolerance
block_size = 128 postings
```

Then benchmark:

```text
bin_width in {mz_tolerance / 2, mz_tolerance, 2 * mz_tolerance}
block_size in {64, 128, 256, 512}
```

Each query peak maps to every bin whose m/z range overlaps
`[query_mz - tolerance, query_mz + tolerance]`. This can over-generate candidates
but must never miss exact matches.

### Query Algorithm, Version 1

Implement a simpler safe block-pruning path before full WAND:

1. Build the active bin-list set for each query peak.
2. For each active bin block, compute a block upper bound:
   `sum(q_i_norm * block_max_normalized_weight)` for the query peaks that can
   match that bin.
3. If the block upper bound is below the current lower bound, skip the block.
4. Otherwise mark candidate spectrum ids from that block.
5. Exact-score marked candidates using the current top-k lower bound.

This version is easier than a complete WAND pivot loop and can validate whether
block upper bounds are strong enough on spectra.

Acceptance:

- Exact parity with the current cosine indexed thresholded top-k API.
- Exact parity with the current entropy indexed thresholded top-k API.
- No false negatives on randomized and reference-spectrum tests.
- At least 2x faster than the metric's `1m-10k-current` baseline before keeping
  it.
- Diagnostics show a large drop in `candidates_rescored / query`.

Expected speedup:

- Low confidence lower bound: 2x.
- Plausible target at threshold `0.9`: 3x to 10x.
- If candidate counts remain high, abandon this version and move to full WAND.

### Query Algorithm, Version 2

If Version 1 is promising but still not enough, implement full document-at-a-time
Block-Max WAND:

1. Maintain one cursor per active bin list.
2. Sort active lists by current `spectrum_id`.
3. Compute a pivot where accumulated list upper bounds exceed the lower bound.
4. If the pivot spectrum id equals the candidate spectrum id, exact-score that
   spectrum.
5. Otherwise advance cursors to the pivot id or next block that can still beat
   the lower bound.
6. Update the lower bound whenever top-k improves.

Acceptance:

- Exact parity with the current indexed top-k path for each metric.
- At least 5x faster than the metric's `1m-10k-current` baseline to justify the
  more complex code.
- Memory overhead documented and below 2x the existing metric-specific threshold
  index unless the speedup is exceptional.

Status: the block-local cursor implementation described above was attempted for
cosine and rejected. It was exact on targeted tests, but it was about 13.1x
slower on the 1M x 10k target and still rescored 256 candidates/query. Do not
repeat that shape without a fundamentally different scheduler or a different
diagnostic profile showing that many more candidates can be skipped.

## Experiment 2: Exact Thresholded Top-K Join Bounds

This track keeps the L2AP-style idea only if it is adapted to exact thresholded
top-k. A threshold-only join is not sufficient, because the production graph
needs the best `k` neighbors above threshold per query.

The useful part of L2AP is the stronger prefix, suffix, and candidate-specific
upper bounds. Those bounds should feed the same live lower bound used by the
top-k search:

```text
lower_bound = max(user_threshold, current_kth_score)
```

Cosine implementation:

- Normalize spectra to unit L2 norm for pruning.
- Sort dimensions by a stable global order that favors rare or high-pruning
  features.
- Store prefix entries only up to the thresholded top-k stopping point.
- Accumulate partial dot products during candidate generation.
- Apply prefix residual L2 bounds, suffix residual L2 bounds, and
  candidate-specific verification bounds before exact rescoring.
- Exact-rescore every candidate before inserting it into the top-k heap.

Entropy implementation:

- Reuse the same traversal and scratch structure.
- Order peaks by `entropy_peak_upper_bound`.
- Use `entropy_raw_threshold(lower_bound)` as the pruning target.
- Store library-side maxima needed to bound additive entropy contributions.
- Apply candidate-specific additive residual bounds before exact rescoring.
- Exact-rescore every candidate with `EntropyKernel`.

Acceptance:

- Exact parity with the current cosine indexed thresholded top-k path.
- Exact parity with the current entropy indexed thresholded top-k path.
- No false negatives in deterministic randomized tests.
- Benchmark only the 1M x 10k thresholded top-k target, not threshold-only edge
  emission.
- Keep only if the path is at least 2x faster for one metric and has a clear
  parity route for the other metric.

Expected speedup:

- 2x to 10x is plausible if the residual bounds prevent candidate
  materialization.
- If it still enumerates most above-threshold candidates, abandon it in favor of
  a materially different traversal; the rejected block-local WAND and per-bin
  impact attempts should not be repeated.

### Residual Verification Bound Attempt

The first Experiment 2 implementation added a shared exact candidate verifier
for cosine and entropy. The verifier stored one residual-bound total per
library spectrum and walked the same m/z-ordered two-pointer path as the exact
scorer. During verification it rejected a candidate only when:

```text
raw_so_far + conservative_bound(query_tail, library_tail) < raw_target
```

For cosine the residual value was the squared prepared peak value and the bound
was Cauchy-Schwarz. For entropy the residual value was
`entropy_pair(peak, 1.0)` and the bound was the smaller remaining additive
query/library bound.

The implementation was exact on targeted unit tests and on the cosine/entropy
index integration suites, but it did not reduce candidate materialization. On
the cosine 1M x 10k indexed thresholded top-k target, `k=16`, threshold `0.9`,
it regressed:

```text
time: [816.23 ms 821.38 ms 826.75 ms]
change from current default: about 5.6% slower
product_postings/q=277.772
candidates_marked/q=277.772
candidates_rescored/q=277.772
spectrum_blocks_allowed/q=1.000 out of 977.000
```

Decision: reject and remove the code path. Tail verification bounds are too
late in the current traversal; they add branches to candidates that have already
been materialized. Any remaining Experiment 2 work must move the residual bound
into candidate generation itself, for example by accumulating partial prefix
scores before exact verification.

### Candidate Seed Ranking Attempt

The next cosine-only prototype kept the current allowed-block filter, collected
candidates with the score contribution of the query peak that first discovered
them, sorted those candidates by that seed score, and then exact-scored likely
strong candidates first. Before exact rescoring it used:

```text
seed_raw + query_suffix_norm_after_seed * library_norm
```

against the live `max(threshold, kth_score)` raw target. A second version reused
the candidate buffer through `SearchState` to remove per-query allocation.

Cosine 1M x 10k result, `k=16`, threshold `0.9`:

```text
allocation prototype:
  time: [745.54 ms 749.15 ms 752.84 ms]
  change from current default: about 8.8% faster

SearchState scratch prototype:
  time: [749.58 ms 754.74 ms 760.99 ms]
  no statistically significant improvement over the allocation prototype

diagnostics:
  product_postings/q=814.259
  candidates_marked/q=316.261
  candidates_rescored/q=268.888
  spectrum_blocks_allowed/q=1.000 out of 977.000
```

Decision: reject and remove the code path. The candidate ordering did reduce
exact rescoring from about `277.8` to `268.9` candidates/query, but it had to
scan about three times more postings/query because the top-k floor no longer
rose during candidate discovery. The net speedup stayed below the 10% local
hot-path keep threshold and did not approach the 2x requirement for a new
auxiliary traversal.

### Library-Side Prefix Candidate Generation Attempt

The next cosine-only attempt implemented the exact one-sided library-prefix
filter implied by L2AP suffix bounds. `FlashCosineThresholdIndex` built a
block-local posting layout containing only the library peaks selected by the
fixed threshold. During top-k search, each query peak probed those library
prefix postings in the allowed spectrum blocks, and every touched spectrum was
exact-scored by the existing two-pointer scorer.

The exactness proof is one-sided: if a library spectrum's stored prefix is
chosen so that its remaining suffix L2 norm is below
`threshold * ||library||`, then any query/library pair with cosine at least the
fixed threshold must match at least one library prefix peak. The query side
therefore does not need a shared global prefix order for safety, as long as all
query peaks are probed against the library prefixes.

Cosine 1M x 10k result, `k=16`, threshold `0.9`:

```text
time: [826.53 ms 829.25 ms 832.47 ms]
change from current default: about 8.5% slower
product_postings/q=143.539
prefix_postings/q=417.452
candidates_marked/q=295.705
candidates_rescored/q=295.705
spectrum_blocks_allowed/q=1.000 out of 977.000
```

Decision: reject and remove the code path. The prefix path was exact and did
reduce some full product-posting visits, but it replaced them with more total
posting work and touched more candidates than the current query-suffix dynamic
path. It also added a second block-local posting index without approaching the
2x speedup required for auxiliary index structures.

After removing the prototype, the same benchmark returned to the active dynamic
query-order path at `[751.94 ms 754.31 ms 756.97 ms]`, with
`product_postings/q=277.772`, `prefix_postings/q=0.000`, and
`candidates_rescored/q=277.772`.

Entropy was not implemented for this shape. The current entropy index accepts
the threshold at query time, so a library-prefix posting layout would either
need one auxiliary index per cutoff or a query-time rebuild. That does not match
the current entropy API or the exact top-k target. If entropy gets a fixed
threshold-specialized index in the future, the same one-sided additive suffix
idea can be revisited with `entropy_peak_upper_bound` and
`entropy_raw_threshold`.

## Experiment 3: Impact-Ordered Exact Top-K Traversal

Impact-ordered indexes store postings by contribution magnitude rather than only
by spectrum id. They can raise the top-k floor early, which makes
`max(threshold, current_kth_score)` stronger for the rest of the query.

Implementation:

- Build an auxiliary posting layout per m/z bin ordered by metric-specific peak
  contribution descending.
- For cosine, order by normalized peak weight.
- For entropy, order by the maximum entropy contribution that the library peak
  can provide, with query-specific refinement where possible.
- Query high-impact bins first.
- Maintain a conservative residual upper bound per query peak and per bin.
- Stop scanning an impact layer only when the residual bound proves that unseen
  postings cannot enter the exact thresholded top-k result.
- Exact-rescore all emitted candidates before returning them.

Acceptance:

- Exact parity with current cosine indexed thresholded top-k.
- Exact parity with current entropy indexed thresholded top-k.
- Lower `candidates_marked / query` or `candidates_rescored / query` on the 1M x
  10k benchmark.
- No more than 1.5x memory unless speedup is at least 3x.

Expected speedup:

- 1.5x to 5x if high-intensity peaks dominate the score.
- Weak if the benchmark spectra have many similarly weighted peaks.

Risk:

- Exact residual bounds are easy to get wrong. This experiment should start
  behind private code and be tested with property tests against the current
  implementation.
- Do not repeat the rejected per-bin intra-block impact scan. The next
  impact-ordered attempt must avoid candidate materialization more globally,
  for example with a multi-list impact/WAND traversal.
- Do not repeat the rejected block-local WAND shape either; its repeated
  active-list sorting and cursor advancement overwhelmed the small reduction in
  exact scoring.

## Experiment 4: Precision And Memory-Bandwidth Variants

The current index is generic over storage precision. Large index workloads may
be memory-bandwidth limited, so lower precision can matter. This track is not an
exact replacement unless score and ordering differences are accepted or the
lower-precision data is used only for safe pruning bounds.

Implementation:

- Benchmark `FlashCosineThresholdIndex::<f32>` and, where feasible, `::<f16>` on
  the large cosine benchmark.
- Benchmark `FlashEntropyIndex::<f32>` and, where feasible, `::<f16>` on the
  large entropy benchmark, both weighted and unweighted.
- Compare exact output against `f64` for a smaller deterministic subset.
- For exact index variants, keep exact rescoring in `f64` if original data is
  available.
- For pruning-only auxiliary indexes, ensure lower-precision bounds are rounded
  conservatively upward so they cannot cause false negatives.
- For entropy, explicitly test small intensities and underflow-prone spectra
  because entropy probabilities can be much smaller than cosine peak products.

Acceptance:

- If used for exact public results, document score/rank changes.
- If used only for pruning, prove no false negatives through conservative bounds.
- Keep only if it improves the relevant 1M x 10k benchmark by at least 20% or
  materially reduces memory.

Expected speedup:

- 1.1x to 2x if memory bandwidth dominates.
- Low or no gain if branchy candidate logic dominates.

### Precision Benchmark Results

The large benchmark now accepts `INDEX_SEARCH_TOP_K_PRECISIONS=f64,f32,f16`
and includes the precision in the Criterion group name. This keeps the workload
fixed while changing only the index storage precision.

Cosine indexed thresholded top-k, 1M x 10k, `k=16`, threshold `0.9`:

```text
f64:
  time: [760.67 ms 764.34 ms 768.53 ms]
  product_postings/q=277.772
  candidates_rescored/q=277.772

f32:
  time: [770.08 ms 772.58 ms 775.30 ms]
  product_postings/q=277.773
  candidates_rescored/q=277.773

f16:
  skipped at 1M because index construction failed with
  InvalidPeakSpacing("library spectrum")
```

Entropy indexed thresholded top-k, 1M x 10k, `k=16`, threshold `0.9`:

```text
weighted f64:
  time: [3.9359 s 3.9532 s 3.9731 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000

weighted f32:
  time: [3.8689 s 3.8875 s 3.9083 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000

unweighted f64:
  time: [3.9113 s 3.9268 s 3.9431 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000

unweighted f32:
  time: [3.9206 s 3.9289 s 3.9368 s]
  product_postings/q=18427.802
  candidates_marked/q=256.000
  candidates_rescored/q=0.000

f16:
  skipped at 1M because index construction failed with
  InvalidPeakSpacing("library spectrum")
```

Decision: do not switch the benchmark default or public recommendation to
lower precision. `f32` did not improve cosine and only gave a small weighted
entropy improvement, far below the 20% keep threshold. `f16` is not usable with
the current storage layout because m/z values are also stored as half precision;
quantization can collapse peaks enough to violate the well-separated invariant.
If lower precision is revisited, it should store m/z separately from score
values and keep m/z at least `f32`.

## Remaining High-Upside Options

The experiments above have likely exhausted the small local variants inside the
current per-query traversal. The remaining options that could plausibly produce
large speedups are structural. They should be treated as new tracks, not as
minor tweaks to the existing dynamic allowed-block scorer.

### Spectrum Reordering Before Block Construction

This is the most attractive next exact experiment. The block upper-bound filter
is only as good as the spectra grouped into the same spectrum-id block. The
current benchmark is clustered synthetically, which makes one block contain the
near-duplicate neighborhood. Real 24M-spectrum collections may not arrive in a
similarity-friendly order. If unrelated spectra share blocks, the block filter
will allow too many weak spectra or use larger blocks than necessary.

Implementation:

- Compute a cheap deterministic signature for every prepared spectrum before
  building block-level auxiliary indexes.
- Sort spectra by that signature, keeping a remapping from internal spectrum id
  back to insertion-order public spectrum id.
- Build the shared product, neutral-loss, block upper-bound, and block-local
  product indexes over the reordered spectra.
- Return public `spectrum_id` values in insertion order, not reordered internal
  ids.
- Try several signatures: precursor m/z bucket plus top product m/z peaks,
  binned top-intensity m/z sketch, sparse SimHash over m/z bins, and minhash over
  coarse m/z bins.
- Keep exact scoring unchanged. Reordering may improve pruning locality but must
  not alter scores.

Acceptance:

- Exact parity with current cosine and entropy public APIs, including indexed
  query ids.
- No public API change unless an opt-in constructor is needed for compatibility.
- Benchmark on the 1M x 10k synthetic target and, more importantly, on a
  shuffled version of the same target to measure the value when input order is
  bad.
- Keep only if it materially reduces allowed blocks, candidates, or wall time on
  non-clustered input.

Expected speedup:

- Low on already-clustered inputs.
- Potentially large on production data if input order is arbitrary.

Status: implemented behind the `experimental_reordered_index` feature as an
opt-in constructor path for the cosine threshold index and entropy index. The
shared `FlashIndex` now carries an internal/public spectrum-id map, so block
postings and candidate bitsets can use reordered internal ids while public
results and indexed query ids remain insertion-order ids. Regression tests cover
cosine and entropy indexed queries and verify that reordered indexes return the
same public ids and scores as the normal indexes.

The first signature based on top-intensity peaks was rejected during the
experiment because intensity jitter destroyed already-clustered input order. The
kept experimental signature is a coarse m/z sketch: the first eight m/z bins at
`10 * tolerance`, plus precursor bin, peak count, and public id as a
deterministic tie-breaker. This preserves the synthetic cluster order much
better while still recovering locality after a shuffle.

Cosine clustered 1M x 10k, `k=16`, threshold `0.9`:

```text
normal threshold index:
  time: [751.94 ms 754.31 ms 756.97 ms]
  product_postings/q=277.772
  candidates_rescored/q=277.772
  spectrum_blocks_allowed/q=1.000 out of 977.000

reordered threshold index:
  time: [906.39 ms 912.99 ms 919.69 ms]
  product_postings/q=287.422
  candidates_rescored/q=287.422
  spectrum_blocks_allowed/q=1.291 out of 977.000
```

Decision for already-clustered input: do not make this the default. It regresses
the current best-case ordering by about 21%, because the existing order already
places near-duplicates in the same block.

Cosine shuffled 1M x 100, `k=16`, threshold `0.9`:

```text
normal threshold index:
  time: [1.9647 s 1.9847 s 2.0056 s]
  product_postings/q=29005.190
  candidates_rescored/q=29005.190
  spectrum_blocks_allowed/q=977.000 out of 977.000

reordered threshold index:
  time: [10.376 ms 10.415 ms 10.491 ms]
  product_postings/q=281.820
  candidates_rescored/q=281.820
  spectrum_blocks_allowed/q=1.280 out of 977.000
```

The unreordered shuffled 1M x 10k run was interrupted after Criterion estimated
about 2008 seconds for ten samples. The reordered shuffled 1M x 10k path did
complete:

```text
reordered threshold index:
  time: [1.1306 s 1.1425 s 1.1553 s]
  product_postings/q=286.467
  candidates_rescored/q=286.467
  spectrum_blocks_allowed/q=1.294 out of 977.000
```

Weighted entropy shuffled 1M x 10, `k=16`, threshold `0.9`:

```text
normal entropy index:
  time: [3.7652 s 3.7837 s 3.8008 s]
  product_postings/q=2496063.300
  candidates_marked/q=911068.700
  spectrum_blocks_allowed/q=3907.000 out of 3907.000

reordered entropy index:
  time: [6.5548 ms 6.6036 ms 6.6735 ms]
  product_postings/q=22707.500
  candidates_marked/q=502.200
  spectrum_blocks_allowed/q=2.000 out of 3907.000
```

Weighted entropy reordered shuffled 1M x 10k:

```text
time: [5.9940 s 6.1419 s 6.3164 s]
product_postings/q=19041.122
candidates_marked/q=499.666
spectrum_blocks_allowed/q=2.049 out of 3907.000
```

Decision: keep the experiment as a feature-gated opt-in, not as the default
index order. The acceleration is large when the input order is arbitrary, and
the implementation preserves exact scores and public ids. The regression on
already-clustered input means production callers should either know that their
input is poorly ordered, or the crate needs an automatic ordering-quality probe
before enabling reordering by default.

### Exact-Certified Two-Stage Search

A lossy or approximate candidate generator is not acceptable by itself, but it
could be used as a first stage if the second stage can certify exact top-k
completeness. The fallback must be the current exact path whenever the cheap
stage cannot prove that no missed candidate can beat the live lower bound.

Implementation:

- Generate a small high-recall candidate set using a cheap sketch such as binned
  top peaks, SimHash buckets, or coarse minhash.
- Exact-score those candidates first to raise the top-k lower bound quickly.
- Run the existing exact dynamic block-pruned path with the raised lower bound.
- Return only exact scores from the existing scorer.
- Add diagnostics for prefilter candidates, exact fallback candidates, and how
  often the approximate stage alone certified nothing.

Acceptance:

- No false negatives relative to the current exact thresholded top-k path.
- Deterministic randomized tests with adversarial spectra where the sketch misses
  true neighbors, proving the fallback recovers them.
- At least 2x faster on the 1M x 10k target before keeping the extra index.

Risk:

- The certification step may still need nearly the same exact traversal, making
  this just another candidate-ordering variant. The experiment is only worth
  keeping if the early lower-bound raise prunes large parts of the exact
  fallback.

### Batch Query Execution

This changes execution shape rather than the scoring contract. Query-level rayon
is still the production parallelization model, but large graph construction will
issue millions of independent queries. A batched executor could improve cache
locality by processing a group of queries against the same block-local postings
before moving on.

Implementation:

- Add a private benchmark-only batch runner first, not a public API.
- Process `B` indexed queries at a time, with one `SearchState` and
  `TopKSearchState` per query.
- Reuse allowed-block computations and block-local posting scans where multiple
  queries touch the same block.
- Compare against simple rayon-over-query execution with the same number of
  worker threads.

Acceptance:

- Exact parity with independent per-query calls.
- Wall-clock improvement over query-level rayon, not just over sequential
  single-query execution.
- Memory bounded by `B * SearchState`, with a clear recommended batch size.

Risk:

- This can complicate the API and scheduling. It should remain an internal
  production helper until it beats straightforward rayon on realistic workloads.

### SIMD Or GPU Exact Rescoring

The current cosine top-k path still exact-scores hundreds of candidates per
query on the synthetic target. Faster exact rescoring may help when pruning is
already close to its limit.

Implementation:

- Start with portable SIMD or architecture-gated SIMD for the exact two-pointer
  direct scorer.
- Keep scalar fallback and exact score parity tests.
- Only consider GPU if queries are batched; single-query GPU dispatch overhead is
  unlikely to help.
- For GPU, keep candidate generation on CPU initially and move only exact
  candidate scoring for large batches.

Acceptance:

- Bitwise or tolerance-bounded parity with scalar scoring.
- At least 20% improvement in the rescoring-heavy benchmark slices.
- No regression for small spectra and low candidate counts.

Risk:

- Current cosine speed is no longer dominated by exact rescoring alone. SIMD may
  be useful, but it is unlikely to produce an order-of-magnitude improvement by
  itself.

### Exact Data Reduction Before Indexing

Real libraries often contain duplicate or replicate spectra. Reducing the number
of indexed spectra can be exact if the reduction preserves the expansion back to
all original spectrum ids.

Implementation:

- Detect exact duplicate prepared spectra after normalization and validation.
- Index one representative and store a list of duplicate public ids.
- Expand results after exact scoring, preserving top-k ordering and ties.
- Consider near-duplicate clustering only if the downstream graph contract
  accepts a non-exact approximation; otherwise keep this track to true
  duplicates.

Acceptance:

- Exact parity on libraries with duplicate spectra.
- Public `spectrum_id` expansion is deterministic.
- Benchmarks report duplicate compression ratio, index build time, memory, and
  query time.

Expected speedup:

- Proportional to duplicate compression when duplicates are common.
- No benefit on unique spectra.

## Dropped Or Deferred Tracks

These tracks are not part of the active index-improvement roadmap for the
current requirement.

Threshold-only graph traversal is dropped. It emits all edges above threshold,
while the required graph keeps top-k neighbors above threshold per query. It can
only return if the downstream contract changes to threshold-only edge emission.

BLINK-style sparse matrix or batched candidate generation is deferred to a
separate approximate-search discussion. It is not acceptable for this exact
roadmap unless a proof shows that exact top-k recovery is guaranteed before
returning results.

Batch execution inside the index is dropped as an algorithmic experiment. Rayon
belongs at the query level with independent per-worker state. The index
benchmark should remain single-query focused so pruning improvements are not
masked by parallelism.

A plain threshold L2AP join is dropped. Only the thresholded top-k adaptation in
Experiment 2 remains active.

## Recommended Work Order

1. Add diagnostic reporting to the 1M x 10k benchmark.
2. Add the entropy counterpart benchmark and indexed entropy self-query API
   needed for apples-to-apples thresholded top-k workloads. Done.
3. Extract the cosine block upper-bound prefilter into a shared private
   architecture with metric-specific bound providers. Done.
4. Add the entropy block upper-bound provider and prove exact parity. Done.
5. Benchmark the shared block prefilter on the entropy 1M x 10k targets. Done.
6. Add block-aware postings traversal so allowed spectrum blocks skip product
   postings instead of only filtering after each posting is visited. Done.
7. Reduce the next dominant traversal costs: cosine prefix-posting intersection
   and entropy block-local spectrum-window checks. Done.
8. Replace mark-and-rescore with block-restricted exact accumulation for the
   high-threshold block-pruned path. Done.
9. Implement full Block-Max WAND or an equivalent intra-block dynamic bound on
   the shared layout to reduce candidate materialization before exact scoring.
   Done and rejected for the block-local cursor design.
10. Adapt L2AP-style residual bounds into an exact thresholded top-k join for
   cosine and entropy. A tail-verification-only residual bound was attempted
   and rejected. Cosine candidate-seed ranking and one-sided library-prefix
   candidate generation prototypes were also rejected; the remaining viable
   shape must prune during candidate generation without increasing posting
   scans.
11. If residual-bound joins underperform, prototype impact-ordered exact
   postings with conservative residual bounds.
12. After the traversal architecture is materially better, benchmark precision
   and layout variants for memory-bandwidth pressure. Precision variants were
   benchmarked and rejected for the current layout.
13. Keep rayon outside the index, at query level, and benchmark it only as a
   production scaling layer after the single-query path improves.

## Test And Verification Requirements

Every experiment must include:

- Parity tests against the current exact indexed top-k path for every supported
  metric.
- Tests for `k=0`, empty library, empty query, threshold above 1, and query id
  validation.
- Deterministic randomized tests comparing all returned `(spectrum_id, score,
  n_matches)` triples.
- A smaller lower-precision smoke test if the experiment touches storage
  precision.
- Benchmark comparison against the metric's `1m-10k-current` baseline.
- Diagnostic comparison showing why the speed changed.
- A short note for any metric intentionally left unsupported by the experiment,
  including the mathematical reason.

Before keeping a patch:

```bash
cargo fmt --check
cargo test --release
cargo test --release --features rayon
cargo clippy --release --all-targets --all-features -- -D warnings
git diff --check
```

## Decision Rules

Use these thresholds to avoid keeping complexity that does not matter:

- Instrumentation patches can be kept without speedup if they are low-noise and
  useful.
- Local hot-path patches need at least 10% improvement on 1M x 10k.
- New auxiliary index structures need at least 2x improvement or a strong memory
  reduction.
- Full Block-Max WAND complexity needs at least 5x improvement.
- Approximate candidate generation is out of scope for this exact thresholded
  top-k roadmap.
- Threshold-only graph traversal is out of scope unless the downstream contract
  changes.
- Rayon speedups do not justify index complexity; they are measured separately
  as query-level production scaling.
- A cosine-only improvement can be merged only as an explicitly temporary step;
  the next index task must close or document the entropy parity gap.

Do not keep a patch because it improves the 32-query smoke benchmark. The
decision benchmarks are the metric-specific 1M x 10k targets.

## References

- Yuanyue Li et al., "Spectral entropy outperforms MS/MS dot product similarity
  for small-molecule compound identification", Nature Methods 2021, DOI
  [10.1038/s41592-021-01331-z](https://doi.org/10.1038/s41592-021-01331-z).
- Yuanyue Li and Oliver Fiehn, "Flash entropy search to query all mass spectral
  libraries in real time", Nature Methods 2023, DOI
  [10.1038/s41592-023-02012-9](https://doi.org/10.1038/s41592-023-02012-9).
- David C. Anastasiu and George Karypis, "L2AP: Fast cosine similarity search
  with prefix L-2 norm bounds", ICDE 2014, DOI
  [10.1109/ICDE.2014.6816700](https://doi.org/10.1109/ICDE.2014.6816700).
- Andrei Z. Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason
  Zien, "Efficient query evaluation using a two-level retrieval process", CIKM
  2003, DOI [10.1145/956863.956944](https://doi.org/10.1145/956863.956944).
- Shuai Ding and Torsten Suel, "Faster top-k document retrieval using block-max
  indexes", SIGIR 2011, DOI
  [10.1145/2009916.2010048](https://doi.org/10.1145/2009916.2010048).
- Thomas V. Harwood et al., "BLINK enables ultrafast tandem mass spectrometry
  cosine similarity scoring", Scientific Reports 2023, DOI
  [10.1038/s41598-023-40496-9](https://doi.org/10.1038/s41598-023-40496-9).
