//! Criterion benchmark for SPLASH generation.

use std::{
    env,
    path::{Path, PathBuf},
    time::Duration,
};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use mass_spectrometry::prelude::{Spectrum, SpectrumSplash};
use std::hint::black_box;

#[derive(Clone)]
struct BenchSpectrum {
    peaks: Vec<(f64, f64)>,
}

impl Spectrum for BenchSpectrum {
    type Precision = f64;

    type SortedIntensitiesIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, (f64, f64)>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.peaks.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.peaks.iter().map(|peak| peak.1)
    }

    fn intensity_nth(&self, n: usize) -> f64 {
        self.peaks[n].1
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.peaks.iter().map(|peak| peak.0)
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.peaks[index..].iter().map(|peak| peak.0)
    }

    fn mz_nth(&self, n: usize) -> f64 {
        self.peaks[n].0
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.peaks.iter().copied()
    }

    fn peak_nth(&self, n: usize) -> (f64, f64) {
        self.peaks[n]
    }

    fn precursor_mz(&self) -> f64 {
        1.0
    }
}

struct ReferenceRow {
    expected: String,
    spectrum: String,
    peaks: BenchSpectrum,
}

fn reference_path() -> PathBuf {
    if let Some(path) = env::var_os("SPLASH_BENCH_REFERENCE") {
        return PathBuf::from(path);
    }

    let gnps_path = PathBuf::from("/tmp/gnps-splash-check/gnps_splash_java_10k.csv");
    if gnps_path.exists() {
        return gnps_path;
    }

    PathBuf::from("tests/fixtures/splash_reference.csv.zst")
}

fn parse_spectrum(spectrum: &str) -> BenchSpectrum {
    let peaks = spectrum
        .split_whitespace()
        .map(|peak| {
            let (mz, intensity) = peak.split_once(':').expect("peak separator");
            (
                mz.parse::<f64>().expect("mz value"),
                intensity.parse::<f64>().expect("intensity value"),
            )
        })
        .collect();
    BenchSpectrum { peaks }
}

fn load_reference() -> Vec<ReferenceRow> {
    let path = reference_path();
    let content = read_reference_to_string(&path);
    let mut rows = Vec::new();

    for line in content.lines() {
        if line.starts_with("expected,") {
            continue;
        }
        let (expected, rest) = line.split_once(',').expect("expected SPLASH column");
        let (_origin, spectrum) = rest.split_once(',').expect("origin and spectrum columns");
        rows.push(ReferenceRow {
            expected: expected.to_owned(),
            spectrum: spectrum.to_owned(),
            peaks: parse_spectrum(spectrum),
        });
    }

    rows
}

fn read_reference_to_string(path: &Path) -> String {
    if path.extension().and_then(|extension| extension.to_str()) == Some("zst") {
        let file = std::fs::File::open(path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", path.display()));
        let decoded = zstd::decode_all(file)
            .unwrap_or_else(|error| panic!("failed to decode {}: {error}", path.display()));
        return String::from_utf8(decoded)
            .unwrap_or_else(|error| panic!("{} is not valid UTF-8: {error}", path.display()));
    }

    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn bench_splash(c: &mut Criterion) {
    let rows = load_reference();
    assert!(!rows.is_empty(), "benchmark reference must not be empty");

    for row in &rows {
        assert_eq!(row.peaks.splash().unwrap(), row.expected);
    }

    let peak_count = rows.iter().map(|row| row.peaks.peaks.len()).sum::<usize>();
    let mut group = c.benchmark_group("splash");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(rows.len() as u64));
    group.bench_function(
        format!(
            "spectrum_splash_{}_spectra_{}_peaks",
            rows.len(),
            peak_count
        ),
        |b| {
            b.iter(|| {
                let mut bytes = 0usize;
                for row in &rows {
                    let splash = black_box(&row.peaks)
                        .splash()
                        .expect("reference row should produce SPLASH");
                    bytes += black_box(splash).len();
                }
                black_box(bytes);
            });
        },
    );
    group.bench_function(
        format!(
            "parse_and_splash_{}_spectra_{}_peaks",
            rows.len(),
            peak_count
        ),
        |b| {
            b.iter(|| {
                let mut bytes = 0usize;
                for row in &rows {
                    let spectrum = parse_spectrum(black_box(row.spectrum.as_str()));
                    let splash = spectrum
                        .splash()
                        .expect("reference row should produce SPLASH");
                    bytes += black_box(splash).len();
                }
                black_box(bytes);
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench_splash);
criterion_main!(benches);
