use std::hint::black_box;

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};
use mass_spectrometry::prelude::{
    EpimeloscineSpectrum, GenericSpectrum, HydroxyCholesterolSpectrum, SalicinSpectrum,
    ScalarSimilarity, SimilarityComputationError, SiriusMergeClosePeaks, SpectralProcessor,
};

pub type BenchSpectrum = GenericSpectrum;

type SimilarityResult = Result<(f64, usize), SimilarityComputationError>;

pub struct BenchmarkSpectra {
    pub salicin: BenchSpectrum,
    pub hydroxy_cholesterol: BenchSpectrum,
    pub epimeloscine: BenchSpectrum,
}

pub fn benchmark_spectra() -> BenchmarkSpectra {
    BenchmarkSpectra {
        salicin: GenericSpectrum::salicin().expect("reference spectrum should build"),
        hydroxy_cholesterol: GenericSpectrum::hydroxy_cholesterol()
            .expect("reference spectrum should build"),
        epimeloscine: GenericSpectrum::epimeloscine().expect("reference spectrum should build"),
    }
}

#[allow(dead_code)]
pub fn benchmark_spectra_for_linear(mz_tolerance: f64) -> BenchmarkSpectra {
    let spectra = benchmark_spectra();
    // Linear variants require strict peak spacing (> 2 * tolerance).
    let merger = SiriusMergeClosePeaks::new(mz_tolerance)
        .expect("linear benchmark tolerance should be valid");
    BenchmarkSpectra {
        salicin: merger.process(&spectra.salicin),
        hydroxy_cholesterol: merger.process(&spectra.hydroxy_cholesterol),
        epimeloscine: merger.process(&spectra.epimeloscine),
    }
}

pub fn bench_standard_pairs<S>(
    c: &mut Criterion,
    prefix: &str,
    spectra: &BenchmarkSpectra,
    scorer: &S,
) where
    S: ScalarSimilarity<BenchSpectrum, BenchSpectrum, Similarity = SimilarityResult>,
{
    bench_case(
        c,
        format!("{prefix}_hydroxy_cholesterol_salicin"),
        &spectra.hydroxy_cholesterol,
        &spectra.salicin,
        scorer,
    );
    bench_case(
        c,
        format!("{prefix}_hydroxy_cholesterol_hydroxy_cholesterol"),
        &spectra.hydroxy_cholesterol,
        &spectra.hydroxy_cholesterol,
        scorer,
    );
    bench_case(
        c,
        format!("{prefix}_salicin_salicin"),
        &spectra.salicin,
        &spectra.salicin,
        scorer,
    );
}

pub fn bench_epimeloscine_pairs<S>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    prefix: &str,
    spectra: &BenchmarkSpectra,
    scorer: &S,
) where
    S: ScalarSimilarity<BenchSpectrum, BenchSpectrum, Similarity = SimilarityResult>,
{
    bench_group_case(
        group,
        format!("{prefix}_salicin_epimeloscine"),
        &spectra.salicin,
        &spectra.epimeloscine,
        scorer,
    );
    bench_group_case(
        group,
        format!("{prefix}_hydroxy_cholesterol_epimeloscine"),
        &spectra.hydroxy_cholesterol,
        &spectra.epimeloscine,
        scorer,
    );
    bench_group_case(
        group,
        format!("{prefix}_epimeloscine_epimeloscine"),
        &spectra.epimeloscine,
        &spectra.epimeloscine,
        scorer,
    );
}

fn bench_case<S>(
    c: &mut Criterion,
    name: String,
    left: &BenchSpectrum,
    right: &BenchSpectrum,
    scorer: &S,
) where
    S: ScalarSimilarity<BenchSpectrum, BenchSpectrum, Similarity = SimilarityResult>,
{
    c.bench_function(&name, |b| {
        b.iter(|| {
            scorer
                .similarity(black_box(left), black_box(right))
                .expect("similarity computation should succeed")
        })
    });
}

fn bench_group_case<S>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: String,
    left: &BenchSpectrum,
    right: &BenchSpectrum,
    scorer: &S,
) where
    S: ScalarSimilarity<BenchSpectrum, BenchSpectrum, Similarity = SimilarityResult>,
{
    group.bench_function(&name, |b| {
        b.iter(|| {
            scorer
                .similarity(black_box(left), black_box(right))
                .expect("similarity computation should succeed")
        })
    });
}
