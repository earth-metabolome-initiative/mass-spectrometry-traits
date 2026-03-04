use mass_spectrometry::prelude::{
    GenericSpectrum, RandomSpectrumConfig, RandomSpectrumGenerationError, Spectrum, SpectrumAlloc,
};

#[test]
fn random_spectrum_reproducible_for_same_seed() {
    let config = RandomSpectrumConfig {
        precursor_mz: 900.0_f64,
        n_peaks: 32,
        mz_min: 50.0,
        mz_max: 250.0,
        min_peak_gap: 0.25,
        intensity_min: 1.0,
        intensity_max: 10.0,
    };

    let left =
        GenericSpectrum::<f64, f64>::random(config, 42).expect("random spectrum should build");
    let right =
        GenericSpectrum::<f64, f64>::random(config, 42).expect("random spectrum should build");

    assert_eq!(left.len(), right.len());
    assert_eq!(left.precursor_mz(), right.precursor_mz());
    for i in 0..left.len() {
        assert!((left.mz_nth(i) - right.mz_nth(i)).abs() < f64::EPSILON);
        assert!((left.intensity_nth(i) - right.intensity_nth(i)).abs() < f64::EPSILON);
    }
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

    let spectrum =
        GenericSpectrum::<f64, f64>::random(config, 7).expect("random spectrum should build");
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

    let error = match GenericSpectrum::<f64, f64>::random(config, 13) {
        Ok(_) => panic!("invalid config must return an error"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RandomSpectrumGenerationError::InvalidConfig(_)
    ));
}
