mod support;

use std::collections::BTreeSet;

use support::spectrum_factory::{build_spectrum, canonical_spectrum_names, known_spectrum_names};

#[test]
fn registered_names_are_unique_and_resolve() {
    let mut seen = BTreeSet::new();

    for name in known_spectrum_names() {
        assert!(seen.insert(name), "duplicate registry name: {name}");
        assert!(
            build_spectrum(name).is_some(),
            "registry name does not resolve: {name}"
        );
    }

    assert!(
        canonical_spectrum_names().contains(&"hydroxy_cholesterol"),
        "canonical hydroxy_cholesterol entry must exist"
    );
    assert!(
        known_spectrum_names().contains(&"hydroxycholesterol"),
        "hydroxycholesterol alias must exist"
    );
}

#[test]
fn fixture_names_are_registered() {
    let mut fixture_names = BTreeSet::new();

    for path in [
        "tests/fixtures/expected_similarities.csv",
        "tests/fixtures/expected_modified_cosine_similarities.csv",
    ] {
        let mut reader =
            csv::Reader::from_path(path).unwrap_or_else(|e| panic!("failed to open {path}: {e}"));

        for result in reader.records() {
            let record = result.unwrap_or_else(|e| panic!("failed to read {path} record: {e}"));
            fixture_names.insert(record[0].to_string());
            fixture_names.insert(record[1].to_string());
        }
    }

    for name in fixture_names {
        assert!(
            build_spectrum(&name).is_some(),
            "fixture spectrum name missing from registry: {name}"
        );
    }
}

#[test]
fn unknown_name_returns_none() {
    assert!(build_spectrum("not_a_real_compound").is_none());
}
