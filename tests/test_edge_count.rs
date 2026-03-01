use geometric_traits::prelude::*;
use mass_spectrometry::prelude::*;

#[test]
fn count_matching_edges() {
    let epi: GenericSpectrum<f64, f64> = GenericSpectrum::epimeloscine().unwrap();
    let salicin: GenericSpectrum<f64, f64> = GenericSpectrum::salicin().unwrap();
    let hc: GenericSpectrum<f64, f64> = GenericSpectrum::hydroxy_cholesterol().unwrap();

    for (name, left, right) in [
        ("salicin×salicin", &salicin, &salicin),
        ("HC×HC", &hc, &hc),
        ("HC×salicin", &hc, &salicin),
        ("salicin×epi", &salicin, &epi),
        ("HC×epi", &hc, &epi),
        ("epi×epi", &epi, &epi),
    ] {
        let matching = left.matching_peaks(right, 0.1f64).unwrap();
        let edges = matching.number_of_defined_values();
        let nr = matching.number_of_rows();
        let nc = matching.number_of_columns();

        // Count unique rows/cols that have at least one edge (compact dims)
        let mut unique_rows = std::collections::BTreeSet::new();
        let mut unique_cols = std::collections::BTreeSet::new();
        for row in 0..nr {
            for col in matching.sparse_row(row) {
                unique_rows.insert(row);
                unique_cols.insert(col);
            }
        }

        eprintln!(
            "{name:<20} {nr:>5}x{nc:>5} ({edges:>6} edges) compact: {}x{}",
            unique_rows.len(),
            unique_cols.len()
        );
    }
}
