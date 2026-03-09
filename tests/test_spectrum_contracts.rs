use arbitrary::{Arbitrary, Unstructured};
use mass_spectrometry::prelude::{
    ELECTRON_MASS, GenericSpectrum, MAX_MZ, Spectrum, SpectrumAlloc, SpectrumMut,
};

fn make_spectrum(precursor_mz: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz, peaks.len())
        .expect("test spectrum allocation should succeed");
    for &(mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .expect("test peaks must be valid and sorted");
    }
    spectrum
}

#[test]
fn try_with_capacity_accepts_valid_precursor_and_starts_empty() {
    let spectrum =
        GenericSpectrum::try_with_capacity(512.25, 4).expect("valid precursor should build");

    assert_eq!(spectrum.len(), 0);
    assert!(spectrum.is_empty());
    assert_eq!(spectrum.precursor_mz(), 512.25);
    assert!(spectrum.mz().next().is_none());
    assert!(spectrum.intensities().next().is_none());
    assert!(spectrum.peaks().next().is_none());
    assert!(spectrum.mz_from(0).next().is_none());
}

#[test]
fn with_capacity_accepts_precursor_boundaries() {
    let at_min = GenericSpectrum::with_capacity(ELECTRON_MASS, 0)
        .expect("ELECTRON_MASS precursor boundary should be accepted");
    let at_max =
        GenericSpectrum::with_capacity(MAX_MZ, 0).expect("MAX_MZ precursor boundary should work");

    assert_eq!(at_min.precursor_mz(), ELECTRON_MASS);
    assert_eq!(at_max.precursor_mz(), MAX_MZ);
}

#[test]
fn sorted_peak_additions_keep_accessors_consistent() {
    let spectrum = make_spectrum(
        400.0,
        &[(50.0, 1.5), (75.0, 3.0), (120.0, 7.25), (150.5, 9.0)],
    );

    assert_eq!(spectrum.len(), 4);
    assert!(!spectrum.is_empty());
    assert_eq!(
        spectrum.mz().collect::<Vec<_>>(),
        vec![50.0, 75.0, 120.0, 150.5]
    );
    assert_eq!(
        spectrum.intensities().collect::<Vec<_>>(),
        vec![1.5, 3.0, 7.25, 9.0]
    );
    assert_eq!(
        spectrum.peaks().collect::<Vec<_>>(),
        vec![(50.0, 1.5), (75.0, 3.0), (120.0, 7.25), (150.5, 9.0)]
    );
    assert_eq!(spectrum.mz_from(2).collect::<Vec<_>>(), vec![120.0, 150.5]);
    assert!(spectrum.mz_from(spectrum.len()).next().is_none());
    assert_eq!(spectrum.mz_nth(1), 75.0);
    assert_eq!(spectrum.intensity_nth(1), 3.0);
    assert_eq!(spectrum.peak_nth(2), (120.0, 7.25));
}

#[test]
fn arbitrary_generated_spectra_are_sanitized_and_sorted() {
    let buffers = [
        vec![0_u8; 2048],
        vec![0xFF_u8; 2048],
        (0..2048).map(|value| value as u8).collect::<Vec<_>>(),
        (0..2048)
            .map(|value| if value % 2 == 0 { 0x55 } else { 0xAA })
            .collect::<Vec<_>>(),
    ];

    for bytes in &buffers {
        let mut input = Unstructured::new(bytes);
        let spectrum =
            GenericSpectrum::arbitrary(&mut input).expect("fixed buffer should be large enough");

        assert!(spectrum.precursor_mz().is_finite());
        assert!((ELECTRON_MASS..=MAX_MZ).contains(&spectrum.precursor_mz()));
        assert!(spectrum.len() <= 64);

        let mz = spectrum.mz().collect::<Vec<_>>();
        let intensities = spectrum.intensities().collect::<Vec<_>>();
        assert_eq!(mz.len(), intensities.len());

        for pair in mz.windows(2) {
            assert!(
                pair[0] < pair[1],
                "sanitized m/z values must be strictly increasing"
            );
        }

        for value in mz {
            assert!(value.is_finite());
            assert!((ELECTRON_MASS..=MAX_MZ).contains(&value));
        }

        for intensity in intensities {
            assert!(intensity.is_finite());
            assert!(intensity > 0.0);
        }
    }
}
