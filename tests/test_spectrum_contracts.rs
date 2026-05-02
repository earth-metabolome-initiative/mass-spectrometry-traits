use arbitrary::{Arbitrary, Unstructured};
use half::f16;
use mass_spectrometry::prelude::{
    ELECTRON_MASS, GenericSpectrum, LinearCosine, MAX_MZ, MsEntropyCleanSpectrum, ScalarSimilarity,
    SiriusMergeClosePeaks, SpectralProcessor, Spectrum, SpectrumAlloc, SpectrumMut,
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
    let spectrum: GenericSpectrum =
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
    let at_min: GenericSpectrum = GenericSpectrum::with_capacity(ELECTRON_MASS, 0)
        .expect("ELECTRON_MASS precursor boundary should be accepted");
    let at_max: GenericSpectrum =
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
fn top_k_peaks_keeps_most_intense_peaks_in_mz_order() {
    let spectrum = make_spectrum(
        400.0,
        &[
            (50.0, 1.0),
            (75.0, 9.0),
            (100.0, 4.0),
            (120.0, 9.0),
            (150.0, 2.0),
        ],
    );

    let top_three: GenericSpectrum = spectrum
        .top_k_peaks(3)
        .expect("top-k spectrum should build");
    assert_eq!(top_three.precursor_mz(), spectrum.precursor_mz());
    assert_eq!(
        top_three.peaks().collect::<Vec<_>>(),
        vec![(75.0, 9.0), (100.0, 4.0), (120.0, 9.0)]
    );

    let all: GenericSpectrum = spectrum
        .top_k_peaks(usize::MAX)
        .expect("oversized k should keep every peak");
    assert_eq!(
        all.peaks().collect::<Vec<_>>(),
        spectrum.peaks().collect::<Vec<_>>()
    );

    let empty: GenericSpectrum = spectrum
        .top_k_peaks(0)
        .expect("zero k should return an empty spectrum");
    assert_eq!(empty.precursor_mz(), spectrum.precursor_mz());
    assert!(empty.is_empty());
}

#[test]
fn top_k_peaks_breaks_intensity_ties_by_mz_and_preserves_precision() {
    let mut spectrum: GenericSpectrum<f32> =
        GenericSpectrum::try_with_capacity(250.0, 4).expect("f32 precursor should fit");
    spectrum
        .add_peaks([(50.0_f32, 5.0_f32), (75.0, 5.0), (100.0, 4.0), (125.0, 5.0)])
        .expect("f32 peaks should fit");

    let top_two: GenericSpectrum<f32> = spectrum
        .top_k_peaks(2)
        .expect("top-k f32 spectrum should build");
    assert_eq!(
        top_two.peaks().collect::<Vec<_>>(),
        vec![(50.0_f32, 5.0_f32), (75.0, 5.0)]
    );
}

#[test]
fn generic_spectrum_can_store_f32_precision() {
    let mut spectrum: GenericSpectrum<f32> =
        GenericSpectrum::try_with_capacity(250.125, 2).expect("f32 precursor should fit");
    spectrum
        .add_peaks([(50.5_f32, 1.25_f32), (75.25, 2.5)])
        .expect("f32 peaks should fit");

    assert_eq!(spectrum.precursor_mz(), 250.125_f32);
    assert_eq!(spectrum.mz().collect::<Vec<_>>(), vec![50.5_f32, 75.25]);
    assert_eq!(
        spectrum.intensities().collect::<Vec<_>>(),
        vec![1.25_f32, 2.5]
    );
    assert_eq!(spectrum.peak_nth(0), (50.5_f32, 1.25_f32));
}

#[test]
fn generic_spectrum_can_store_f16_precision_when_values_fit() {
    let mut spectrum: GenericSpectrum<f16> =
        GenericSpectrum::try_with_capacity(250.0, 2).expect("f16 precursor should fit");
    spectrum
        .add_peaks([
            (f16::from_f64(50.0), f16::from_f64(1.5)),
            (f16::from_f64(75.0), f16::from_f64(2.0)),
        ])
        .expect("f16 peaks should fit");

    assert_eq!(spectrum.precursor_mz(), f16::from_f64(250.0));
    assert_eq!(
        spectrum.peaks().collect::<Vec<_>>(),
        vec![
            (f16::from_f64(50.0), f16::from_f64(1.5)),
            (f16::from_f64(75.0), f16::from_f64(2.0)),
        ]
    );
}

#[test]
fn add_peak_and_add_peaks_are_fluent_and_precision_native() {
    let mut spectrum: GenericSpectrum<f32> =
        GenericSpectrum::try_with_capacity(250.0, 3).expect("f32 precursor should fit");

    spectrum
        .add_peak(50.0_f32, 1.25_f32)
        .expect("first f32 peak should fit")
        .add_peaks([(75.0_f32, 2.5_f32), (100.0_f32, 4.0_f32)])
        .expect("remaining f32 peaks should fit");

    assert_eq!(
        spectrum.peaks().collect::<Vec<_>>(),
        vec![(50.0_f32, 1.25_f32), (75.0, 2.5), (100.0, 4.0)]
    );
}

#[test]
fn binned_intensities_use_spectrum_precision() {
    let mut spectrum: GenericSpectrum<f32> =
        GenericSpectrum::try_with_capacity(250.0, 3).expect("f32 precursor should fit");
    spectrum.add_peak(50.0, 1.25).expect("f32 peak should fit");
    spectrum.add_peak(75.0, 2.5).expect("f32 peak should fit");
    spectrum.add_peak(100.0, 4.0).expect("f32 peak should fit");

    let linear_bins: Vec<f32> = spectrum
        .linear_binned_intensities(50.0, 100.0, 2)
        .expect("linear binning should succeed");
    assert_eq!(linear_bins, vec![1.25_f32, 6.5]);

    let logarithmic_bins: Vec<f32> = spectrum
        .logarithmic_binned_intensities(50.0, 100.0, 2)
        .expect("logarithmic binning should succeed");
    assert_eq!(logarithmic_bins, vec![1.25_f32, 6.5]);
}

#[test]
fn built_in_processors_preserve_spectrum_precision() {
    let mut spectrum: GenericSpectrum<f32> =
        GenericSpectrum::try_with_capacity(250.0, 3).expect("f32 precursor should fit");
    spectrum.add_peak(50.0, 1.0).expect("f32 peak should fit");
    spectrum.add_peak(75.0, 2.0).expect("f32 peak should fit");
    spectrum.add_peak(100.0, 4.0).expect("f32 peak should fit");

    let cleaner = MsEntropyCleanSpectrum::<f32>::builder_with_precision()
        .normalize_intensity(false)
        .expect("normalization flag should be valid")
        .build()
        .expect("cleaner config should be valid");
    let cleaned: GenericSpectrum<f32> = cleaner.process(&spectrum);
    assert_eq!(
        cleaned.peaks().collect::<Vec<_>>(),
        vec![(50.0_f32, 1.0_f32), (75.0, 2.0), (100.0, 4.0)]
    );

    let merger = SiriusMergeClosePeaks::<f32>::new_with_precision(0.1)
        .expect("merge tolerance should be valid");
    let merged: GenericSpectrum<f32> = merger.process(&cleaned);
    assert_eq!(
        merged.peaks().collect::<Vec<_>>(),
        vec![(50.0_f32, 1.0_f32), (75.0, 2.0), (100.0, 4.0)]
    );
}

#[test]
fn generic_spectrum_rejects_values_that_do_not_fit_selected_precision() {
    let error = GenericSpectrum::<f16>::try_with_capacity(70_000.0, 0)
        .expect_err("f16 cannot represent this precursor finitely");
    assert_eq!(
        error,
        mass_spectrometry::prelude::GenericSpectrumMutationError::NonFinitePrecursorMz
    );

    let mut spectrum: GenericSpectrum<f16> =
        GenericSpectrum::try_with_capacity(100.0, 1).expect("f16 precursor should fit");
    let error = spectrum
        .add_peak(f16::from_f64(70_000.0), f16::from_f64(1.0))
        .expect_err("f16 cannot represent this mz finitely");
    assert_eq!(
        error,
        mass_spectrometry::prelude::GenericSpectrumMutationError::NonFiniteMz
    );
}

#[test]
fn generic_spectrum_validates_after_precision_conversion() {
    let mut spectrum: GenericSpectrum<f16> =
        GenericSpectrum::try_with_capacity(2_000.0, 2).expect("f16 precursor should fit");
    spectrum
        .add_peak(f16::from_f64(2_000.0), f16::from_f64(1.0))
        .expect("first f16 peak should fit");
    let error = spectrum
        .add_peak(f16::from_f64(2_000.49), f16::from_f64(1.0))
        .expect_err("distinct inputs can collapse to the same f16 mz");
    assert_eq!(
        error,
        mass_spectrometry::prelude::GenericSpectrumMutationError::DuplicateMz
    );

    let mut spectrum: GenericSpectrum<f16> =
        GenericSpectrum::try_with_capacity(100.0, 1).expect("f16 precursor should fit");
    let error = spectrum
        .add_peak(f16::from_f64(50.0), f16::from_f64(1.0e-20))
        .expect_err("tiny positive intensity underflows to zero in f16");
    assert_eq!(
        error,
        mass_spectrometry::prelude::GenericSpectrumMutationError::NonPositiveIntensity
    );
}

#[test]
fn f64_similarity_kernels_accept_lower_precision_spectra() {
    let mut left: GenericSpectrum<f32> =
        GenericSpectrum::try_with_capacity(300.0, 2).expect("valid precursor");
    left.add_peak(100.0, 10.0).expect("valid peak");
    left.add_peak(200.0, 20.0).expect("valid peak");

    let mut right: GenericSpectrum<f32> =
        GenericSpectrum::try_with_capacity(300.0, 2).expect("valid precursor");
    right.add_peak(100.0, 10.0).expect("valid peak");
    right.add_peak(200.0, 20.0).expect("valid peak");

    let cosine = LinearCosine::new(0.0, 1.0, 0.1).expect("valid cosine config");
    let (score, matches) = cosine
        .similarity(&left, &right)
        .expect("generic f32 spectra should score");

    assert_eq!(matches, 2);
    assert!((score - 1.0).abs() <= f64::EPSILON);
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
        let spectrum: GenericSpectrum =
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
