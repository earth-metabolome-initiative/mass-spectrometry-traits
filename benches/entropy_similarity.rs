//! Criterion benchmark to evaluate the performance of `EntropySimilarity`
//! in both weighted and unweighted modes.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mass_spectrometry::prelude::{
    EntropySimilarity, EpimeloscineSpectrum, GenericSpectrum, HydroxyCholesterolSpectrum,
    SalicinSpectrum, ScalarSimilarity,
};

fn bench_entropy_similarity(c: &mut Criterion) {
    let salicin = GenericSpectrum::salicin();
    let hydroxy_cholesterol = GenericSpectrum::hydroxy_cholesterol();
    let epimeloscine: GenericSpectrum<f64, f64> = GenericSpectrum::epimeloscine();

    let mz_tolerance = 0.1;
    let weighted = EntropySimilarity::weighted(mz_tolerance);
    let unweighted = EntropySimilarity::unweighted(mz_tolerance);

    // ---------- weighted ----------

    c.bench_function("entropy_weighted_hydroxy_cholesterol_salicin", |b| {
        b.iter(|| weighted.similarity(black_box(&hydroxy_cholesterol), black_box(&salicin)))
    });
    c.bench_function(
        "entropy_weighted_hydroxy_cholesterol_hydroxy_cholesterol",
        |b| {
            b.iter(|| {
                weighted.similarity(
                    black_box(&hydroxy_cholesterol),
                    black_box(&hydroxy_cholesterol),
                )
            })
        },
    );
    c.bench_function("entropy_weighted_salicin_salicin", |b| {
        b.iter(|| weighted.similarity(black_box(&salicin), black_box(&salicin)))
    });

    // ---------- unweighted ----------

    c.bench_function("entropy_unweighted_hydroxy_cholesterol_salicin", |b| {
        b.iter(|| unweighted.similarity(black_box(&hydroxy_cholesterol), black_box(&salicin)))
    });
    c.bench_function(
        "entropy_unweighted_hydroxy_cholesterol_hydroxy_cholesterol",
        |b| {
            b.iter(|| {
                unweighted.similarity(
                    black_box(&hydroxy_cholesterol),
                    black_box(&hydroxy_cholesterol),
                )
            })
        },
    );
    c.bench_function("entropy_unweighted_salicin_salicin", |b| {
        b.iter(|| unweighted.similarity(black_box(&salicin), black_box(&salicin)))
    });

    // ---------- epimeloscine (large spectrum, reduced sample size) ----------

    let mut epimeloscine_group = c.benchmark_group("epimeloscine_entropy");
    epimeloscine_group.sample_size(10);

    epimeloscine_group.bench_function("entropy_weighted_salicin_epimeloscine", |b| {
        b.iter(|| weighted.similarity(black_box(&salicin), black_box(&epimeloscine)))
    });
    epimeloscine_group.bench_function("entropy_weighted_hydroxy_cholesterol_epimeloscine", |b| {
        b.iter(|| weighted.similarity(black_box(&hydroxy_cholesterol), black_box(&epimeloscine)))
    });
    epimeloscine_group.bench_function("entropy_weighted_epimeloscine_epimeloscine", |b| {
        b.iter(|| weighted.similarity(black_box(&epimeloscine), black_box(&epimeloscine)))
    });

    epimeloscine_group.bench_function("entropy_unweighted_salicin_epimeloscine", |b| {
        b.iter(|| unweighted.similarity(black_box(&salicin), black_box(&epimeloscine)))
    });
    epimeloscine_group.bench_function("entropy_unweighted_hydroxy_cholesterol_epimeloscine", |b| {
        b.iter(|| unweighted.similarity(black_box(&hydroxy_cholesterol), black_box(&epimeloscine)))
    });
    epimeloscine_group.bench_function("entropy_unweighted_epimeloscine_epimeloscine", |b| {
        b.iter(|| unweighted.similarity(black_box(&epimeloscine), black_box(&epimeloscine)))
    });

    epimeloscine_group.finish();
}

criterion_group!(benches, bench_entropy_similarity);
criterion_main!(benches);
