use geometric_traits::prelude::*;
use mass_spectrometry::prelude::*;
use multi_ranged::SimpleRange;

#[test]
fn count_edges() {
    let salicin: GenericSpectrum<f64, f64> = GenericSpectrum::salicin().expect("reference spectrum should build");
    let hc: GenericSpectrum<f64, f64> = GenericSpectrum::hydroxy_cholesterol().expect("reference spectrum should build");
    let epi: GenericSpectrum<f64, f64> = GenericSpectrum::epimeloscine().expect("reference spectrum should build");
    let tol = 0.1f64;

    type SpectrumPair<'a> = (
        &'a str,
        &'a GenericSpectrum<f64, f64>,
        &'a str,
        &'a GenericSpectrum<f64, f64>,
    );
    let pairs: Vec<SpectrumPair<'_>> = vec![
        ("HC", &hc, "salicin", &salicin),
        ("HC", &hc, "HC", &hc),
        ("salicin", &salicin, "salicin", &salicin),
        ("salicin", &salicin, "epi", &epi),
        ("HC", &hc, "epi", &epi),
        ("epi", &epi, "epi", &epi),
    ];

    for (ln, left, rn, right) in &pairs {
        let graph: RangedCSR2D<u32, u32, SimpleRange<u32>> = left
            .matching_peaks(*right, tol)
            .expect("matching graph construction should succeed");
        let n_left = left.len();
        let n_right = right.len();
        let n_rows = graph.number_of_rows() as usize;
        let n_cols = graph.number_of_columns() as usize;
        let n_edges = graph.number_of_defined_values();
        assert_eq!(
            n_rows, n_left,
            "row count must match left spectrum length for {ln} vs {rn}"
        );
        assert_eq!(
            n_cols, n_right,
            "column count must match right spectrum length for {ln} vs {rn}"
        );
        let padded = n_rows.max(n_cols);
        println!(
            "{ln}({n_left}) x {rn}({n_right}): graph={n_rows}x{n_cols}, edges={n_edges}, padded={padded}x{padded} = {} entries",
            (padded as u64) * (padded as u64)
        );
    }
}
