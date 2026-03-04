//! Criterion benchmark to evaluate the performance of `HungarianEntropy`
//! in both weighted and unweighted modes.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::HungarianEntropy;

fn bench_entropy_similarity(c: &mut Criterion) {
    let spectra = common::benchmark_spectra();

    let mz_tolerance = 0.1;
    let weighted = HungarianEntropy::weighted(mz_tolerance).expect("valid scorer config");
    let unweighted = HungarianEntropy::unweighted(mz_tolerance).expect("valid scorer config");

    common::bench_standard_pairs(c, "entropy_weighted", &spectra, &weighted);
    common::bench_standard_pairs(c, "entropy_unweighted", &spectra, &unweighted);

    let mut epimeloscine_group = c.benchmark_group("epimeloscine_entropy");
    epimeloscine_group.sample_size(10);
    common::bench_epimeloscine_pairs(
        &mut epimeloscine_group,
        "entropy_weighted",
        &spectra,
        &weighted,
    );
    common::bench_epimeloscine_pairs(
        &mut epimeloscine_group,
        "entropy_unweighted",
        &spectra,
        &unweighted,
    );
    epimeloscine_group.finish();
}

criterion_group!(benches, bench_entropy_similarity);
criterion_main!(benches);
