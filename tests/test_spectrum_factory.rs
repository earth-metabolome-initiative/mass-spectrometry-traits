mod support;

use support::spectrum_factory::build_spectrum;

#[test]
fn known_names_resolve() {
    assert!(build_spectrum("aspirin").is_some());
    assert!(build_spectrum("cocaine").is_some());
}

#[test]
fn aliases_resolve() {
    assert!(build_spectrum("hydroxy_cholesterol").is_some());
    assert!(build_spectrum("hydroxycholesterol").is_some());
}

#[test]
fn unknown_name_returns_none() {
    assert!(build_spectrum("not_a_real_compound").is_none());
}
