# Fuzzing

All fuzz targets are intentionally thin wrappers. Harness logic and oracles
live in the main crate (`mass_spectrometry::fuzzing::*`) so regression tests
replay the exact same code paths.

## Run

```bash
cargo fuzz run hungarian_cosine
cargo fuzz run modified_hungarian_cosine
cargo fuzz run linear_entropy
cargo fuzz run modified_linear_entropy
cargo fuzz run flash_cosine
cargo fuzz run flash_entropy
```

## Replay

Each replay test mirrors crashes from `fuzz/artifacts/*` into
`tests/fixtures/fuzz/*/crashes/`, then replays every file in that fixture
directory:

1. `test_hungarian_cosine_fuzz_harness`
2. `test_modified_hungarian_cosine_fuzz_harness`
3. `test_linear_entropy_fuzz_harness`
4. `test_modified_linear_entropy_fuzz_harness`
5. `test_flash_cosine_fuzz_harness`
6. `test_flash_entropy_fuzz_harness`

Run:

```bash
cargo test --test test_hungarian_cosine_fuzz_harness
cargo test --test test_modified_hungarian_cosine_fuzz_harness
cargo test --test test_linear_entropy_fuzz_harness
cargo test --test test_modified_linear_entropy_fuzz_harness
cargo test --test test_flash_cosine_fuzz_harness
cargo test --test test_flash_entropy_fuzz_harness
```
