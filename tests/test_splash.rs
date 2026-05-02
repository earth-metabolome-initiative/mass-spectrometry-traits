use std::io::{BufRead, BufReader};

use mass_spectrometry::prelude::{
    GenericSpectrum, Spectrum, SpectrumMut, SpectrumSplash, SplashError,
};

struct RawSpectrum {
    peaks: Vec<(f64, f64)>,
}

impl Spectrum for RawSpectrum {
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

fn raw_splash(peaks: impl IntoIterator<Item = (f64, f64)>) -> Result<String, SplashError> {
    RawSpectrum {
        peaks: peaks.into_iter().collect(),
    }
    .splash()
}

fn raw_splash_slice(peaks: &[(f64, f64)]) -> Result<String, SplashError> {
    RawSpectrum {
        peaks: peaks.to_vec(),
    }
    .splash()
}

fn parse_spectrum(spectrum: &str) -> Vec<(f64, f64)> {
    spectrum
        .split_whitespace()
        .map(|peak| {
            let (mz, intensity) = peak
                .split_once(':')
                .expect("fixture peaks must contain a colon");
            (
                mz.parse().expect("fixture mz values must be numeric"),
                intensity
                    .parse()
                    .expect("fixture intensity values must be numeric"),
            )
        })
        .collect()
}

#[test]
fn splash_matches_committed_java_reference_fixture() {
    let fixture = zstd::decode_all(include_bytes!("fixtures/splash_reference.csv.zst") as &[_])
        .expect("compressed SPLASH fixture should decode");
    let mut reader = csv::Reader::from_reader(fixture.as_slice());
    let mut checked = 0usize;

    for record in reader.records() {
        let record = record.expect("fixture rows must parse");
        let expected = &record[0];
        let origin = &record[1];
        let spectrum = parse_spectrum(&record[2]);

        let observed = raw_splash_slice(&spectrum)
            .unwrap_or_else(|error| panic!("{origin} should produce a SPLASH code, got {error:?}"));
        assert_eq!(observed, expected, "{origin}");
        checked += 1;
    }

    assert_eq!(checked, 100);
}

#[test]
fn spectrum_extension_trait_generates_splash_for_strict_spectra() {
    let mut spectrum: GenericSpectrum = GenericSpectrum::try_with_capacity(250.0, 2).unwrap();
    spectrum.add_peak(100.0, 10.0).unwrap();
    spectrum.add_peak(200.0, 20.0).unwrap();

    assert_eq!(
        spectrum.splash().unwrap(),
        "splash10-0udi-0490000000-4425acda10ed7d4709bd"
    );
}

#[test]
fn spectrum_splash_allows_raw_zero_intensity_and_duplicate_mz_values() {
    let peaks = [(100.0, 0.0), (100.0, 10.0), (101.0, 10.0)];

    assert_eq!(
        raw_splash(peaks).unwrap(),
        "splash10-0udi-0900000000-eac7ecd274bda6320aeb"
    );
}

#[test]
fn splash_rejects_invalid_peak_sets() {
    assert_eq!(
        raw_splash(Vec::<(f64, f64)>::new()).unwrap_err(),
        SplashError::EmptySpectrum
    );
    assert_eq!(
        raw_splash([(100.0, 0.0)]).unwrap_err(),
        SplashError::AllZeroIntensities
    );
    assert_eq!(
        raw_splash([(f64::NAN, 1.0)]).unwrap_err(),
        SplashError::NonFiniteMz
    );
    assert_eq!(
        raw_splash([(100.0, f64::INFINITY)]).unwrap_err(),
        SplashError::NonFiniteIntensity
    );
    assert_eq!(
        raw_splash([(-1.0, 1.0)]).unwrap_err(),
        SplashError::NegativeMz
    );
    assert_eq!(
        raw_splash([(100.0, -1.0)]).unwrap_err(),
        SplashError::NegativeIntensity
    );
}

#[test]
#[ignore = "validates every usable GNPS spectrum against the local Java oracle under /tmp"]
fn full_gnps_splash_oracle_matches_java_reference() {
    let spectra = std::fs::File::open("/tmp/gnps-splash-check/gnps_splash_all.csv")
        .expect("GNPS spectrum oracle must be present under /tmp");
    let expected = std::fs::File::open("/tmp/gnps-splash-check/gnps_splash_java_all.csv")
        .expect("GNPS Java SPLASH oracle must be present under /tmp");

    let spectra = BufReader::new(spectra);
    let expected = BufReader::new(expected);
    let mut checked = 0usize;

    for (line_number, (spectrum_line, expected_line)) in
        spectra.lines().zip(expected.lines()).enumerate()
    {
        let spectrum_line = spectrum_line.expect("GNPS spectrum line should read");
        let expected_line = expected_line.expect("GNPS expected line should read");
        let (origin, spectrum) = spectrum_line
            .split_once(',')
            .expect("GNPS spectrum rows have origin,spectrum shape");
        let (expected_splash, expected_remainder) = expected_line
            .split_once(',')
            .expect("GNPS oracle rows have splash,origin,spectrum shape");
        let (expected_origin, expected_spectrum) = expected_remainder
            .split_once(',')
            .expect("GNPS oracle rows include origin and spectrum");

        assert_eq!(origin, expected_origin, "origin mismatch at {line_number}");
        assert_eq!(
            spectrum, expected_spectrum,
            "spectrum mismatch at {line_number}"
        );

        let peaks = parse_spectrum(spectrum);
        let observed = raw_splash_slice(&peaks).unwrap_or_else(|error| {
            panic!("{origin} should produce SPLASH at line {line_number}, got {error:?}")
        });
        assert_eq!(
            observed, expected_splash,
            "SPLASH mismatch for {origin} at line {line_number}"
        );
        checked += 1;
    }

    assert_eq!(checked, 2_090_422);
}
