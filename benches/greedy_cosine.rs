//! Criterion benchmark to evaluate the performance of the `GreedyCosine`
//! function.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::GreedyCosine;

/// Benchmark for the `GreedyCosine` function.
fn bench_greedy_cosine(c: &mut Criterion) {
    let spectra = common::benchmark_spectra();

    let mz_power = 1.0;
    let intensity_power = 1.0;
    let mz_tolerance = 0.1;
    let cosine =
        GreedyCosine::new(mz_power, intensity_power, mz_tolerance).expect("valid scorer config");

    common::bench_standard_pairs(c, "greedy_cosine", &spectra, &cosine);

    let mut epimeloscine_group = c.benchmark_group("epimeloscine");
    epimeloscine_group.sample_size(10);
    common::bench_epimeloscine_pairs(&mut epimeloscine_group, "greedy_cosine", &spectra, &cosine);
    epimeloscine_group.finish();
}

criterion_group!(benches, bench_greedy_cosine);
criterion_main!(benches);
