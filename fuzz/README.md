# Fuzzing

`hungarian_cosine` is intentionally a thin wrapper. The harness and oracles
live in the main crate (`mass_spectrometry::fuzzing::run_hungarian_cosine_case`)
so regression tests can replay the same inputs.

## Run

```bash
cargo fuzz run hungarian_cosine
```

## Replay

`test_hungarian_cosine_fuzz_harness` contains one replay test for this harness.
On each run it:

1. copies all files from `fuzz/artifacts/hungarian_cosine/` into
   `tests/fixtures/fuzz/hungarian_cosine/crashes/`
2. replays every file in `tests/fixtures/fuzz/hungarian_cosine/crashes/`

Run:

```bash
cargo test --test test_hungarian_cosine_fuzz_harness
```
