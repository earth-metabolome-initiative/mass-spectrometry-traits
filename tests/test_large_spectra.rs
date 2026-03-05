//! Large-spectrum regression tests for index/count overflow.

use geometric_traits::prelude::*;
use mass_spectrometry::prelude::{
    GenericSpectrum, HungarianCosine, LinearEntropy, ModifiedHungarianCosine, ScalarSimilarity,
    Spectrum, SpectrumAlloc, SpectrumMut,
};
use multi_ranged::{BiRange, SimpleRange};

fn linear_spectrum(size: usize, precursor_mz: f32) -> GenericSpectrum<f32, f32> {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor_mz, size).expect("valid spectrum allocation");
    for i in 0..size {
        spectrum
            .add_peak(i as f32, 1.0)
            .expect("test peaks must be sorted by m/z");
    }
    spectrum
}

fn shifted_linear_spectrum(
    size: usize,
    precursor_mz: f32,
    mz_offset: f32,
) -> GenericSpectrum<f32, f32> {
    let mut spectrum =
        GenericSpectrum::with_capacity(precursor_mz, size).expect("valid spectrum allocation");
    for i in 0..size {
        spectrum
            .add_peak(mz_offset + i as f32, 1.0)
            .expect("test peaks must be sorted by m/z");
    }
    spectrum
}

#[test]
fn matching_peaks_supports_indices_above_u16() {
    let n = (u16::MAX as usize) + 10;
    let left = linear_spectrum(n, 1_000_000.0);
    let right = linear_spectrum(n, 1_000_000.0);

    let graph: RangedCSR2D<u32, u32, SimpleRange<u32>> = left
        .matching_peaks(&right, 0.0)
        .expect("matching graph construction should succeed");

    assert_eq!(graph.number_of_rows() as usize, n);
    assert_eq!(graph.number_of_columns() as usize, n);
    assert_eq!(graph.number_of_defined_values(), n as u32);

    let last_row = (n - 1) as u32;
    let last_cols: Vec<u32> = graph.sparse_row(last_row).collect();
    assert_eq!(last_cols, vec![last_row]);
}

#[test]
fn modified_matching_peaks_supports_indices_above_u16() {
    let n = (u16::MAX as usize) + 10;
    let left = linear_spectrum(n, 1_000_000.0);
    let right = linear_spectrum(n, 1_000_000.0);

    let graph: RangedCSR2D<u32, u32, BiRange<u32>> = left
        .modified_matching_peaks(&right, 0.0, 0.0)
        .expect("matching graph construction should succeed");

    assert_eq!(graph.number_of_rows() as usize, n);
    assert_eq!(graph.number_of_columns() as usize, n);
    assert_eq!(graph.number_of_defined_values(), n as u32);
}

#[test]
fn matching_peaks_preserves_full_shape_without_edges() {
    let left = shifted_linear_spectrum(128, 1_000_000.0, 0.0);
    let right = shifted_linear_spectrum(257, 2_000_000.0, 10_000.0);

    let graph: RangedCSR2D<u32, u32, SimpleRange<u32>> = left
        .matching_peaks(&right, 0.0)
        .expect("matching graph construction should succeed");

    assert_eq!(graph.number_of_rows() as usize, left.len());
    assert_eq!(graph.number_of_columns() as usize, right.len());
    assert_eq!(graph.number_of_defined_values(), 0);

    let last_row = (left.len() - 1) as u32;
    assert_eq!(graph.sparse_row(last_row).count(), 0);
}

#[test]
fn modified_matching_peaks_preserves_full_shape_without_edges() {
    let left = shifted_linear_spectrum(64, 1_000_000.0, 0.0);
    let right = shifted_linear_spectrum(93, 2_000_000.0, 10_000.0);

    let graph: RangedCSR2D<u32, u32, BiRange<u32>> = left
        .modified_matching_peaks(&right, 0.0, 0.0)
        .expect("matching graph construction should succeed");

    assert_eq!(graph.number_of_rows() as usize, left.len());
    assert_eq!(graph.number_of_columns() as usize, right.len());
    assert_eq!(graph.number_of_defined_values(), 0);

    let last_row = (left.len() - 1) as u32;
    assert_eq!(graph.sparse_row(last_row).count(), 0);
}

#[test]
fn entropy_match_count_scales_past_u16() {
    let n = (u16::MAX as usize) + 10;
    let left = linear_spectrum(n, 1_000_000.0);
    let right = linear_spectrum(n, 1_000_000.0);

    let entropy = LinearEntropy::unweighted(0.0).expect("valid scorer config");
    let (_score, matches) = entropy
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    assert_eq!(matches, n);
}

#[test]
fn similarity_match_count_type_is_usize() {
    let left = linear_spectrum(4, 1_000.0);
    let right = linear_spectrum(4, 1_000.0);

    let exact = HungarianCosine::new(1.0_f32, 1.0_f32, 0.0_f32).expect("valid scorer config");
    let modified =
        ModifiedHungarianCosine::new(1.0_f32, 1.0_f32, 0.0_f32).expect("valid scorer config");
    let entropy = LinearEntropy::unweighted(0.0_f32).expect("valid scorer config");

    let (_s1, m1): (f32, usize) = exact
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (_s2, m2): (f32, usize) = modified
        .similarity(&left, &right)
        .expect("similarity computation should succeed");
    let (_s3, m3): (f32, usize) = entropy
        .similarity(&left, &right)
        .expect("similarity computation should succeed");

    assert_eq!(m1, 4);
    assert_eq!(m2, 4);
    assert_eq!(m3, 4);
}
