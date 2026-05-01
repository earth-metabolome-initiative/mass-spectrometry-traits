use std::io::{BufRead, BufReader};

use mass_spectrometry::prelude::{
    GenericSpectrum, SpectrumMut, SpectrumSplash, SplashError, splash_from_peaks,
};

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

        let observed = splash_from_peaks(&spectrum)
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
fn raw_splash_api_allows_zero_intensity_and_duplicate_mz_values() {
    let peaks = [(100.0, 0.0), (100.0, 10.0), (101.0, 10.0)];

    assert_eq!(
        splash_from_peaks(peaks).unwrap(),
        "splash10-0udi-0900000000-eac7ecd274bda6320aeb"
    );
}

#[test]
fn splash_rejects_invalid_peak_sets() {
    assert_eq!(
        splash_from_peaks(Vec::<(f64, f64)>::new()).unwrap_err(),
        SplashError::EmptySpectrum
    );
    assert_eq!(
        splash_from_peaks([(100.0, 0.0)]).unwrap_err(),
        SplashError::AllZeroIntensities
    );
    assert_eq!(
        splash_from_peaks([(f64::NAN, 1.0)]).unwrap_err(),
        SplashError::NonFiniteMz
    );
    assert_eq!(
        splash_from_peaks([(100.0, f64::INFINITY)]).unwrap_err(),
        SplashError::NonFiniteIntensity
    );
    assert_eq!(
        splash_from_peaks([(-1.0, 1.0)]).unwrap_err(),
        SplashError::NegativeMz
    );
    assert_eq!(
        splash_from_peaks([(100.0, -1.0)]).unwrap_err(),
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
        let observed = splash_from_peaks(&peaks).unwrap_or_else(|error| {
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
