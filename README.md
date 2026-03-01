# Mass Spectrometry

A crate for traits and data structures for mass spectrometry.

## Quality Gates

This repository uses [`prek`](https://prek.j178.dev/) for local pre-commit
checks and for CI.

### Local setup

1. Install `prek` (for example with `cargo install --locked prek`).
2. Install hooks:

```bash
prek install --install-hooks
```

3. Run all hooks on demand:

```bash
prek run --all-files
```

### Enforced checks

The pre-commit hook and CI both run the same strict checks:

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo test --all-targets --all-features --locked`
4. `cargo doc --all-features --no-deps`

`cargo test --all-targets` includes bench targets as test runs.
