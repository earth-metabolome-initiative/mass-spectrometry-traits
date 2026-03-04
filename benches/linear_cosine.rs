//! Criterion benchmark to evaluate the performance of the `LinearCosine`
//! function.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::LinearCosine;

/// Benchmark for the `LinearCosine` function.
fn bench_linear_cosine(c: &mut Criterion) {
    let mz_power = 1.0;
    let intensity_power = 1.0;
    let mz_tolerance = 0.1;
    let spectra = common::benchmark_spectra_for_linear(mz_tolerance);
    let cosine =
        LinearCosine::new(mz_power, intensity_power, mz_tolerance).expect("valid scorer config");

    common::bench_standard_pairs(c, "linear_cosine", &spectra, &cosine);

    let mut epimeloscine_group = c.benchmark_group("epimeloscine_linear");
    epimeloscine_group.sample_size(10);
    common::bench_epimeloscine_pairs(&mut epimeloscine_group, "linear_cosine", &spectra, &cosine);
    epimeloscine_group.finish();
}

criterion_group!(benches, bench_linear_cosine);
criterion_main!(benches);
