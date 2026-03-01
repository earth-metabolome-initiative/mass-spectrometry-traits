//! Criterion benchmark to evaluate the performance of the `HungarianCosine`
//! function.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::{
    EpimeloscineSpectrum, GenericSpectrum, HungarianCosine, HydroxyCholesterolSpectrum,
    SalicinSpectrum, ScalarSimilarity,
};

/// Benchmark for the `HungarianCosine` function.
fn bench_hungarian_cosine(c: &mut Criterion) {
    let salicin = GenericSpectrum::salicin();
    let hydroxy_cholesterol = GenericSpectrum::hydroxy_cholesterol();
    let epimeloscine: GenericSpectrum<f64, f64> = GenericSpectrum::epimeloscine();

    let mz_power = 1.0;
    let intensity_power = 1.0;
    let mz_tolerance = 0.1;
    let cosine =
        HungarianCosine::new(mz_power, intensity_power, mz_tolerance).expect("valid scorer config");

    c.bench_function("cosine_hydroxy_cholesterol_salicin", |b| {
        b.iter(|| {
            cosine
                .similarity(black_box(&hydroxy_cholesterol), black_box(&salicin))
                .expect("similarity computation should succeed")
        })
    });
    c.bench_function("cosine_hydroxy_cholesterol_hydroxy_cholesterol", |b| {
        b.iter(|| {
            cosine
                .similarity(
                    black_box(&hydroxy_cholesterol),
                    black_box(&hydroxy_cholesterol),
                )
                .expect("similarity computation should succeed")
        })
    });
    c.bench_function("cosine_salicin_salicin", |b| {
        b.iter(|| {
            cosine
                .similarity(black_box(&salicin), black_box(&salicin))
                .expect("similarity computation should succeed")
        })
    });

    let mut epimeloscine_group = c.benchmark_group("epimeloscine");
    epimeloscine_group.sample_size(10);

    epimeloscine_group.bench_function("cosine_salicin_epimeloscine", |b| {
        b.iter(|| {
            cosine
                .similarity(black_box(&salicin), black_box(&epimeloscine))
                .expect("similarity computation should succeed")
        })
    });

    epimeloscine_group.bench_function("cosine_hydroxy_cholesterol_epimeloscine", |b| {
        b.iter(|| {
            cosine
                .similarity(black_box(&hydroxy_cholesterol), black_box(&epimeloscine))
                .expect("similarity computation should succeed")
        })
    });

    epimeloscine_group.bench_function("cosine_epimeloscine_epimeloscine", |b| {
        b.iter(|| {
            cosine
                .similarity(black_box(&epimeloscine), black_box(&epimeloscine))
                .expect("similarity computation should succeed")
        })
    });

    epimeloscine_group.finish();
}

criterion_group!(benches, bench_hungarian_cosine);
criterion_main!(benches);
