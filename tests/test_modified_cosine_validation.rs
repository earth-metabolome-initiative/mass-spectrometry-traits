//! Validation of ModifiedCosine against matchms ModifiedCosine (greedy).
//!
//! Since matchms uses greedy assignment and we use optimal (Crouse LAPJV),
//! our scores should be >= matchms scores. We also verify symmetry.

use mass_spectrometry::prelude::*;
use support::spectrum_factory::build_spectrum;

mod support;

#[test]
fn validate_modified_cosine_against_matchms() {
    let csv_path = "tests/fixtures/expected_modified_cosine_similarities.csv";
    let mut reader = csv::Reader::from_path(csv_path)
        .unwrap_or_else(|e| panic!("Failed to open {csv_path}: {e}"));

    let mut tested = 0u32;
    let mut failures: Vec<String> = Vec::new();

    for result in reader.records() {
        let record = result.expect("Failed to read CSV record");

        let left_name = &record[0];
        let right_name = &record[1];
        let tolerance: f32 = record[2].parse().expect("bad tolerance");
        let mz_power: f32 = record[3].parse().expect("bad mz_power");
        let intensity_power: f32 = record[4].parse().expect("bad intensity_power");
        let matchms_score: f64 = record[5].parse().expect("bad score");

        let Some(left) = build_spectrum(left_name) else {
            continue;
        };
        let Some(right) = build_spectrum(right_name) else {
            continue;
        };

        let modified =
            ModifiedCosine::new(mz_power, intensity_power, tolerance).expect("valid scorer config");
        let (score, matches) = modified
            .similarity(&left, &right)
            .expect("similarity computation should succeed");
        let our_score = score as f64;

        // Our optimal assignment should produce scores >= matchms greedy
        // (minus small tolerance for floating point differences).
        if our_score < matchms_score - 1e-4 {
            failures.push(format!(
                "{left_name} vs {right_name} (tol={tolerance}, mz_pow={mz_power}, int_pow={intensity_power}): \
                 our_score={our_score:.12} < matchms_score={matchms_score:.12}"
            ));
        }

        // Also verify symmetry.
        let (score_ba, matches_ba) = modified
            .similarity(&right, &left)
            .expect("similarity computation should succeed");
        if (score as f64 - score_ba as f64).abs() >= 1e-6 {
            failures.push(format!(
                "{left_name} vs {right_name} (tol={tolerance}, mz_pow={mz_power}, int_pow={intensity_power}): \
                 symmetry violation: sim(A,B)={score:.12} != sim(B,A)={score_ba:.12}, \
                 matches(A,B)={matches} != matches(B,A)={matches_ba}"
            ));
        }

        tested += 1;
    }

    println!("Tested: {tested}, Failures: {}", failures.len());

    if !failures.is_empty() {
        let sample: Vec<&str> = failures.iter().take(100).map(|s| s.as_str()).collect();
        panic!(
            "{} / {tested} tests failed (showing first {}):\n{}",
            failures.len(),
            sample.len(),
            sample.join("\n")
        );
    }

    assert!(tested > 0, "No tests were run");
}
