//! Criterion benchmark to evaluate the performance of the `modified_cosine`
//! function.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::{
    EpimeloscineSpectrum, GenericSpectrum, HydroxyCholesterolSpectrum, ModifiedCosine,
    SalicinSpectrum, ScalarSimilarity,
};

/// Benchmark for the `modified_cosine` function.
fn bench_modified_cosine(c: &mut Criterion) {
    let salicin = GenericSpectrum::salicin();
    let hydroxy_cholesterol = GenericSpectrum::hydroxy_cholesterol();
    let epimeloscine: GenericSpectrum<f64, f64> = GenericSpectrum::epimeloscine();

    let mz_power = 1.0;
    let intensity_power = 1.0;
    let mz_tolerance = 0.1;
    let cosine = ModifiedCosine::new(mz_power, intensity_power, mz_tolerance);

    c.bench_function("modified_cosine_hydroxy_cholesterol_salicin", |b| {
        b.iter(|| cosine.similarity(black_box(&hydroxy_cholesterol), black_box(&salicin)))
    });
    c.bench_function(
        "modified_cosine_hydroxy_cholesterol_hydroxy_cholesterol",
        |b| {
            b.iter(|| {
                cosine.similarity(
                    black_box(&hydroxy_cholesterol),
                    black_box(&hydroxy_cholesterol),
                )
            })
        },
    );
    c.bench_function("modified_cosine_salicin_salicin", |b| {
        b.iter(|| cosine.similarity(black_box(&salicin), black_box(&salicin)))
    });

    let mut epimeloscine_group = c.benchmark_group("epimeloscine_modified");
    epimeloscine_group.sample_size(10);

    epimeloscine_group.bench_function("modified_cosine_salicin_epimeloscine", |b| {
        b.iter(|| cosine.similarity(black_box(&salicin), black_box(&epimeloscine)))
    });

    epimeloscine_group.bench_function(
        "modified_cosine_hydroxy_cholesterol_epimeloscine",
        |b| {
            b.iter(|| {
                cosine.similarity(black_box(&hydroxy_cholesterol), black_box(&epimeloscine))
            })
        },
    );

    epimeloscine_group.bench_function("modified_cosine_epimeloscine_epimeloscine", |b| {
        b.iter(|| cosine.similarity(black_box(&epimeloscine), black_box(&epimeloscine)))
    });

    epimeloscine_group.finish();
}

criterion_group!(benches, bench_modified_cosine);
criterion_main!(benches);
