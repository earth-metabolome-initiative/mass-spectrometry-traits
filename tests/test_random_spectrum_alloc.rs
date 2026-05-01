use mass_spectrometry::prelude::{
    GenericSpectrum, GenericSpectrumMutationError, RandomSpectrumConfig,
    RandomSpectrumGenerationError, Spectrum, SpectrumAlloc,
};

const ZERO_SEED_FALLBACK: u64 = 0x9E37_79B9_7F4A_7C15;

fn base_config() -> RandomSpectrumConfig {
    RandomSpectrumConfig {
        precursor_mz: 900.0_f64,
        n_peaks: 32,
        mz_min: 50.0,
        mz_max: 250.0,
        min_peak_gap: 0.25,
        intensity_min: 1.0,
        intensity_max: 10.0,
    }
}

fn assert_identical(left: &GenericSpectrum, right: &GenericSpectrum) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.precursor_mz(), right.precursor_mz());
    for i in 0..left.len() {
        assert!((left.mz_nth(i) - right.mz_nth(i)).abs() < f64::EPSILON);
        assert!((left.intensity_nth(i) - right.intensity_nth(i)).abs() < f64::EPSILON);
    }
}

#[test]
fn random_spectrum_reproducible_for_same_seed() {
    let config = base_config();

    let left: GenericSpectrum =
        GenericSpectrum::random(config, 42).expect("random spectrum should build");
    let right: GenericSpectrum =
        GenericSpectrum::random(config, 42).expect("random spectrum should build");

    assert_identical(&left, &right);
}

#[test]
fn random_spectrum_respects_gap_and_ranges() {
    let config = RandomSpectrumConfig {
        precursor_mz: 800.0_f64,
        n_peaks: 64,
        mz_min: 100.0,
        mz_max: 400.0,
        min_peak_gap: 0.3,
        intensity_min: 5.0,
        intensity_max: 25.0,
    };

    let spectrum: GenericSpectrum =
        GenericSpectrum::random(config, 7).expect("random spectrum should build");
    assert_eq!(spectrum.len(), config.n_peaks);

    for i in 0..spectrum.len() {
        let mz = spectrum.mz_nth(i);
        let intensity = spectrum.intensity_nth(i);
        assert!(mz >= config.mz_min && mz <= config.mz_max);
        assert!(intensity >= config.intensity_min && intensity <= config.intensity_max);
    }

    for i in 1..spectrum.len() {
        let prev = spectrum.mz_nth(i - 1);
        let curr = spectrum.mz_nth(i);
        assert!(curr > prev, "m/z values must be strictly increasing");
        assert!(
            curr - prev >= config.min_peak_gap - 1e-12,
            "m/z gap must respect configured minimum"
        );
    }
}

#[test]
fn random_spectrum_rejects_invalid_config() {
    let config = RandomSpectrumConfig {
        precursor_mz: 500.0_f64,
        n_peaks: 4,
        mz_min: 200.0,
        mz_max: 100.0,
        min_peak_gap: 0.25,
        intensity_min: 1.0,
        intensity_max: 2.0,
    };

    let error = match GenericSpectrum::<f64>::random(config, 13) {
        Ok(_) => panic!("invalid config must return an error"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RandomSpectrumGenerationError::InvalidConfig(_)
    ));
}

#[test]
fn random_spectrum_allows_zero_peaks() {
    let mut config = base_config();
    config.n_peaks = 0;

    let spectrum: GenericSpectrum =
        GenericSpectrum::random(config, 42).expect("zero-peak spectrum should build");

    assert_eq!(spectrum.len(), 0);
    assert_eq!(spectrum.precursor_mz(), config.precursor_mz);
}

#[test]
fn random_spectrum_zero_seed_uses_deterministic_fallback() {
    let config = base_config();

    let zero_seed: GenericSpectrum =
        GenericSpectrum::random(config, 0).expect("zero seed should build");
    let normalized: GenericSpectrum =
        GenericSpectrum::random(config, ZERO_SEED_FALLBACK).expect("fallback seed should build");
    let different: GenericSpectrum =
        GenericSpectrum::random(config, 1).expect("different seed should build");

    assert_identical(&zero_seed, &normalized);
    assert!(
        (zero_seed.mz_nth(0) - different.mz_nth(0)).abs() > f64::EPSILON
            || (zero_seed.intensity_nth(0) - different.intensity_nth(0)).abs() > f64::EPSILON
    );
}

#[test]
fn random_spectrum_uses_constant_intensity_when_span_is_zero() {
    let mut config = base_config();
    config.n_peaks = 8;
    config.intensity_min = 3.5;
    config.intensity_max = 3.5;

    let spectrum: GenericSpectrum =
        GenericSpectrum::random(config, 99).expect("constant-intensity spectrum should build");

    assert_eq!(spectrum.len(), config.n_peaks);
    for intensity in spectrum.intensities() {
        assert_eq!(intensity, 3.5);
    }
}

#[test]
fn random_spectrum_rejects_non_finite_config_fields() {
    let cases = [
        (
            "mz_min",
            RandomSpectrumConfig {
                mz_min: f64::NAN,
                ..base_config()
            },
        ),
        (
            "mz_max",
            RandomSpectrumConfig {
                mz_max: f64::INFINITY,
                ..base_config()
            },
        ),
        (
            "min_peak_gap",
            RandomSpectrumConfig {
                min_peak_gap: f64::NAN,
                ..base_config()
            },
        ),
        (
            "intensity_min",
            RandomSpectrumConfig {
                intensity_min: f64::NEG_INFINITY,
                ..base_config()
            },
        ),
        (
            "intensity_max",
            RandomSpectrumConfig {
                intensity_max: f64::NAN,
                ..base_config()
            },
        ),
    ];

    for (label, config) in cases {
        let error =
            GenericSpectrum::<f64>::random(config, 7).expect_err("non-finite field must fail");
        assert!(
            matches!(error, RandomSpectrumGenerationError::NonFiniteValue(name) if name == label),
            "expected NonFiniteValue({label}), got {error:?}"
        );
    }
}

#[test]
fn random_spectrum_rejects_non_positive_min_peak_gap() {
    let config = RandomSpectrumConfig {
        min_peak_gap: 0.0,
        ..base_config()
    };

    let error = GenericSpectrum::<f64>::random(config, 7).expect_err("zero gap must fail");
    assert!(matches!(
        error,
        RandomSpectrumGenerationError::InvalidConfig("min_peak_gap must be > 0")
    ));
}

#[test]
fn random_spectrum_rejects_span_exhaustion() {
    let config = RandomSpectrumConfig {
        n_peaks: 4,
        mz_min: 10.0,
        mz_max: 10.2,
        min_peak_gap: 0.1,
        ..base_config()
    };

    let error = GenericSpectrum::<f64>::random(config, 7).expect_err("span exhaustion must fail");
    assert!(matches!(
        error,
        RandomSpectrumGenerationError::InvalidConfig(
            "n_peaks and min_peak_gap exceed [mz_min, mz_max] span"
        )
    ));
}

#[test]
fn random_spectrum_propagates_constructor_failures_as_mutation_errors() {
    let config = RandomSpectrumConfig {
        precursor_mz: f64::NAN,
        ..base_config()
    };

    let error = GenericSpectrum::<f64>::random(config, 7)
        .expect_err("invalid precursor must propagate as error");
    assert!(matches!(
        error,
        RandomSpectrumGenerationError::Mutation(GenericSpectrumMutationError::NonFinitePrecursorMz)
    ));
}
