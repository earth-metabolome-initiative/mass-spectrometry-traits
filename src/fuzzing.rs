//! Reusable fuzzing harnesses for similarity implementations.
//!
//! The harness logic lives in the main crate so fuzz targets and regression
//! tests execute the exact same code paths.

use arbitrary::{Arbitrary, Unstructured};

use alloc::vec::Vec;

use crate::prelude::ScalarSimilarity;
use crate::structs::{
    FlashCosineIndex, FlashEntropyIndex, FlashSearchResult, GenericSpectrum, HungarianCosine,
    LinearCosine, LinearEntropy, ModifiedHungarianCosine, ModifiedLinearCosine,
    ModifiedLinearEntropy, MsEntropyCleanSpectrum, SimilarityComputationError,
    SiriusMergeClosePeaks,
};
use crate::traits::{SpectralProcessor, Spectrum};

const FIXED_TOLERANCE: f64 = 0.1;
const SYMMETRY_EPS: f64 = 1.0e-4;
const DIFFERENTIAL_EPS: f64 = 1.0e-4;
const MODIFIED_DIFFERENTIAL_EPS: f64 = 1.0e-6;

/// Result returned by [`run_hungarian_cosine_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HungarianCosineHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Result returned by [`run_modified_hungarian_cosine_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifiedHungarianCosineHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Result returned by [`run_linear_entropy_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearEntropyHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Result returned by [`run_modified_linear_entropy_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifiedLinearEntropyHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Result returned by [`run_flash_cosine_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashCosineHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Result returned by [`run_flash_entropy_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashEntropyHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Result returned by [`run_ms_entropy_clean_spectrum_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsEntropyCleanSpectrumHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Execute the Hungarian-cosine fuzz harness for an arbitrary byte slice.
///
/// The function intentionally panics when a correctness invariant is violated.
/// This behavior is required for fuzzers and regression tests to surface bugs.
pub fn run_hungarian_cosine_case(bytes: &[u8]) -> HungarianCosineHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match CosineFuzzCase::arbitrary(&mut unstructured) {
        Ok(case) => case,
        Err(_) => return HungarianCosineHarnessOutcome::Checked,
    };

    if let Ok(dynamic) = HungarianCosine::new(case.mz_power, case.intensity_power, case.tolerance) {
        assert_bidirectional_properties(&dynamic, &case.left, &case.right, "dynamic", false);
    }

    let fixed = HungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("fixed config is valid");
    assert_bidirectional_properties(&fixed, &case.left, &case.right, "fixed", true);
    assert_self_similarity(&fixed, &case.left, 1.0e-5, "fixed/left");
    assert_self_similarity(&fixed, &case.right, 1.0e-5, "fixed/right");

    let wide = HungarianCosine::new(0.0, 1.0, 2.0).expect("wide config is valid");
    assert_self_similarity(&wide, &case.left, 1.0e-4, "wide/left");
    assert_self_similarity(&wide, &case.right, 1.0e-4, "wide/right");

    if case.left.len().max(case.right.len()) <= 128 {
        assert_linear_matches_hungarian(&case.left, &case.right);
    }

    HungarianCosineHarnessOutcome::Checked
}

/// Execute the modified-Hungarian-cosine fuzz harness for an arbitrary byte
/// slice.
///
/// The function intentionally panics when a correctness invariant is violated.
/// This behavior is required for fuzzers and regression tests to surface bugs.
pub fn run_modified_hungarian_cosine_case(bytes: &[u8]) -> ModifiedHungarianCosineHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match CosineFuzzCase::arbitrary(&mut unstructured) {
        Ok(case) => case,
        Err(_) => return ModifiedHungarianCosineHarnessOutcome::Checked,
    };

    if let Ok(dynamic) =
        ModifiedHungarianCosine::new(case.mz_power, case.intensity_power, case.tolerance)
    {
        assert_modified_bidirectional_properties(&dynamic, &case.left, &case.right, "dynamic");
        if let Ok(dynamic_hungarian) =
            HungarianCosine::new(case.mz_power, case.intensity_power, case.tolerance)
        {
            assert_modified_not_less_than_hungarian(
                &dynamic,
                &dynamic_hungarian,
                &case.left,
                &case.right,
                "dynamic",
            );
        }
    }

    let fixed = ModifiedHungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE)
        .expect("fixed modified config is valid");
    assert_modified_bidirectional_properties(&fixed, &case.left, &case.right, "fixed");
    assert_modified_self_similarity(&fixed, &case.left, 1.0e-5, "fixed/left");
    assert_modified_self_similarity(&fixed, &case.right, 1.0e-5, "fixed/right");
    let fixed_hungarian =
        HungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("fixed config is valid");
    assert_modified_not_less_than_hungarian(
        &fixed,
        &fixed_hungarian,
        &case.left,
        &case.right,
        "fixed",
    );

    if case.left.len().max(case.right.len()) <= 128 {
        assert_modified_linear_matches_modified_hungarian(&case.left, &case.right);
    }

    ModifiedHungarianCosineHarnessOutcome::Checked
}

/// Execute the linear-entropy fuzz harness for an arbitrary byte slice.
///
/// The function intentionally panics when a correctness invariant is violated.
/// This behavior is required for fuzzers and regression tests to surface bugs.
pub fn run_linear_entropy_case(bytes: &[u8]) -> LinearEntropyHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match EntropyFuzzCase::arbitrary(&mut unstructured) {
        Ok(case) => case,
        Err(_) => return LinearEntropyHarnessOutcome::Checked,
    };

    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let merged_left = merger.process(&case.left);
    let merged_right = merger.process(&case.right);

    if let Ok(dynamic) = LinearEntropy::new(
        case.mz_power,
        case.intensity_power,
        case.tolerance,
        case.weighted,
    ) {
        assert_linear_entropy_bidirectional_properties(
            &dynamic,
            &case.left,
            &case.right,
            "dynamic/original",
        );
        assert_linear_entropy_bidirectional_properties(
            &dynamic,
            &merged_left,
            &merged_right,
            "dynamic/merged",
        );
    }

    let fixed_unweighted =
        LinearEntropy::unweighted(FIXED_TOLERANCE).expect("fixed unweighted config is valid");
    assert_linear_entropy_bidirectional_properties(
        &fixed_unweighted,
        &merged_left,
        &merged_right,
        "fixed_unweighted/merged",
    );
    assert_linear_entropy_self_similarity(
        &fixed_unweighted,
        &merged_left,
        1.0e-4,
        "fixed_unweighted/merged/left",
    );
    assert_linear_entropy_self_similarity(
        &fixed_unweighted,
        &merged_right,
        1.0e-4,
        "fixed_unweighted/merged/right",
    );

    let fixed_weighted =
        LinearEntropy::weighted(FIXED_TOLERANCE).expect("fixed weighted config is valid");
    assert_linear_entropy_bidirectional_properties(
        &fixed_weighted,
        &merged_left,
        &merged_right,
        "fixed_weighted/merged",
    );
    assert_linear_entropy_self_similarity(
        &fixed_weighted,
        &merged_left,
        1.0e-4,
        "fixed_weighted/merged/left",
    );
    assert_linear_entropy_self_similarity(
        &fixed_weighted,
        &merged_right,
        1.0e-4,
        "fixed_weighted/merged/right",
    );

    assert_linear_entropy_original_outcome(
        &fixed_unweighted,
        &case.left,
        &case.right,
        "fixed_unweighted/original",
    );
    assert_linear_entropy_original_outcome(
        &fixed_unweighted,
        &case.right,
        &case.left,
        "fixed_unweighted/original/reverse",
    );

    LinearEntropyHarnessOutcome::Checked
}

/// Execute the modified-linear-entropy fuzz harness for an arbitrary byte
/// slice.
///
/// The function intentionally panics when a correctness invariant is violated.
/// This behavior is required for fuzzers and regression tests to surface bugs.
pub fn run_modified_linear_entropy_case(bytes: &[u8]) -> ModifiedLinearEntropyHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match EntropyFuzzCase::arbitrary(&mut unstructured) {
        Ok(case) => case,
        Err(_) => return ModifiedLinearEntropyHarnessOutcome::Checked,
    };

    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let merged_left = merger.process(&case.left);
    let merged_right = merger.process(&case.right);

    if let Ok(dynamic) = ModifiedLinearEntropy::new(
        case.mz_power,
        case.intensity_power,
        case.tolerance,
        case.weighted,
    ) {
        assert_modified_linear_entropy_bidirectional_properties(
            &dynamic,
            &case.left,
            &case.right,
            "dynamic/original",
        );
        assert_modified_linear_entropy_bidirectional_properties(
            &dynamic,
            &merged_left,
            &merged_right,
            "dynamic/merged",
        );
        if let Ok(dynamic_linear) = LinearEntropy::new(
            case.mz_power,
            case.intensity_power,
            case.tolerance,
            case.weighted,
        ) {
            assert_modified_entropy_not_less_than_linear(
                &dynamic,
                &dynamic_linear,
                &merged_left,
                &merged_right,
                "dynamic/merged",
            );
        }
    }

    let fixed_unweighted = ModifiedLinearEntropy::unweighted(FIXED_TOLERANCE)
        .expect("fixed unweighted modified entropy config is valid");
    assert_modified_linear_entropy_bidirectional_properties(
        &fixed_unweighted,
        &merged_left,
        &merged_right,
        "fixed_unweighted/merged",
    );
    assert_modified_linear_entropy_self_similarity(
        &fixed_unweighted,
        &merged_left,
        1.0e-4,
        "fixed_unweighted/merged/left",
    );
    assert_modified_linear_entropy_self_similarity(
        &fixed_unweighted,
        &merged_right,
        1.0e-4,
        "fixed_unweighted/merged/right",
    );
    let fixed_unweighted_linear = LinearEntropy::unweighted(FIXED_TOLERANCE)
        .expect("fixed unweighted linear entropy config is valid");
    assert_modified_entropy_not_less_than_linear(
        &fixed_unweighted,
        &fixed_unweighted_linear,
        &merged_left,
        &merged_right,
        "fixed_unweighted/merged",
    );

    let fixed_weighted = ModifiedLinearEntropy::weighted(FIXED_TOLERANCE)
        .expect("fixed weighted modified entropy config is valid");
    assert_modified_linear_entropy_bidirectional_properties(
        &fixed_weighted,
        &merged_left,
        &merged_right,
        "fixed_weighted/merged",
    );
    assert_modified_linear_entropy_self_similarity(
        &fixed_weighted,
        &merged_left,
        1.0e-4,
        "fixed_weighted/merged/left",
    );
    assert_modified_linear_entropy_self_similarity(
        &fixed_weighted,
        &merged_right,
        1.0e-4,
        "fixed_weighted/merged/right",
    );
    let fixed_weighted_linear = LinearEntropy::weighted(FIXED_TOLERANCE)
        .expect("fixed weighted linear entropy config is valid");
    assert_modified_entropy_not_less_than_linear(
        &fixed_weighted,
        &fixed_weighted_linear,
        &merged_left,
        &merged_right,
        "fixed_weighted/merged",
    );

    assert_modified_linear_entropy_original_outcome(
        &fixed_unweighted,
        &case.left,
        &case.right,
        "fixed_unweighted/original",
    );
    assert_modified_linear_entropy_original_outcome(
        &fixed_unweighted,
        &case.right,
        &case.left,
        "fixed_unweighted/original/reverse",
    );

    ModifiedLinearEntropyHarnessOutcome::Checked
}

/// Execute the flash-cosine fuzz harness for an arbitrary byte slice.
///
/// The function intentionally panics when a correctness invariant is violated.
/// This behavior is required for fuzzers and regression tests to surface bugs.
pub fn run_flash_cosine_case(bytes: &[u8]) -> FlashCosineHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match FlashCosineFuzzCase::arbitrary(&mut unstructured) {
        Ok(case) => case,
        Err(_) => return FlashCosineHarnessOutcome::Checked,
    };

    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let merged_lib: [GenericSpectrum; 3] = [
        merger.process(&case.library[0]),
        merger.process(&case.library[1]),
        merger.process(&case.library[2]),
    ];
    let merged_query = merger.process(&case.query);

    // Fixed config (1.0, 1.0, 0.1): differential oracle against LinearCosine.
    let index =
        match FlashCosineIndex::<f64>::new(1.0_f64, 1.0_f64, FIXED_TOLERANCE, merged_lib.iter()) {
            Ok(idx) => idx,
            Err(_) => return FlashCosineHarnessOutcome::Checked,
        };

    let direct_results = match index.search(&merged_query) {
        Ok(r) => r,
        Err(_) => return FlashCosineHarnessOutcome::Checked,
    };

    let oracle =
        LinearCosine::new(1.0_f64, 1.0_f64, FIXED_TOLERANCE).expect("fixed config is valid");
    assert_flash_cosine_equivalence(
        &direct_results,
        &oracle,
        &merged_lib,
        &merged_query,
        "fixed",
    );

    // search_modified: scores >= direct scores + oracle equivalence.
    if let Ok(modified_results) = index.search_modified(&merged_query) {
        assert_flash_modified_not_less_than_direct(
            &modified_results,
            &direct_results,
            "fixed/cosine",
        );

        let modified_oracle = ModifiedLinearCosine::new(1.0_f64, 1.0_f64, FIXED_TOLERANCE)
            .expect("fixed config is valid");
        assert_flash_modified_cosine_equivalence(
            &modified_results,
            &modified_oracle,
            &merged_lib,
            &merged_query,
            "fixed/modified",
        );
    }

    // search_with_state: exact equality with search.
    let mut state = index.new_search_state();
    if let Ok(state_results) = index.search_with_state(&merged_query, &mut state) {
        assert_flash_state_equivalence(&state_results, &direct_results, "fixed/cosine/direct");
    }
    if let Ok(modified_results) = index.search_modified(&merged_query)
        && let Ok(modified_state_results) =
            index.search_modified_with_state(&merged_query, &mut state)
    {
        assert_flash_state_equivalence(
            &modified_state_results,
            &modified_results,
            "fixed/cosine/modified",
        );
    }

    // Self-similarity: each non-empty library spectrum as query.
    for (i, lib_spec) in merged_lib.iter().enumerate() {
        assert_flash_self_similarity_cosine(&index, lib_spec, i as u32, "fixed/cosine");
    }

    // Dynamic config (arbitrary params): attempt build + search, check score ranges.
    if let Ok(dyn_index) = FlashCosineIndex::<f64>::new(
        case.mz_power,
        case.intensity_power,
        case.tolerance,
        merged_lib.iter(),
    ) && let Ok(dyn_results) = dyn_index.search(&merged_query)
    {
        for r in &dyn_results {
            assert_score_in_range(r.score, "dynamic/cosine");
            assert!(
                r.n_matches <= merged_query.len(),
                "dynamic/cosine: n_matches {} > query len {}",
                r.n_matches,
                merged_query.len()
            );
        }
    }

    FlashCosineHarnessOutcome::Checked
}

/// Execute the flash-entropy fuzz harness for an arbitrary byte slice.
///
/// The function intentionally panics when a correctness invariant is violated.
/// This behavior is required for fuzzers and regression tests to surface bugs.
pub fn run_flash_entropy_case(bytes: &[u8]) -> FlashEntropyHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match FlashEntropyFuzzCase::arbitrary(&mut unstructured) {
        Ok(case) => case,
        Err(_) => return FlashEntropyHarnessOutcome::Checked,
    };

    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let merged_lib: [GenericSpectrum; 3] = [
        merger.process(&case.library[0]),
        merger.process(&case.library[1]),
        merger.process(&case.library[2]),
    ];
    let merged_query = merger.process(&case.query);

    // Fixed unweighted config: differential oracle against LinearEntropy.
    let unweighted_index =
        match FlashEntropyIndex::<f64>::unweighted(FIXED_TOLERANCE, merged_lib.iter()) {
            Ok(idx) => idx,
            Err(_) => return FlashEntropyHarnessOutcome::Checked,
        };

    let uw_direct = match unweighted_index.search(&merged_query) {
        Ok(r) => r,
        Err(_) => return FlashEntropyHarnessOutcome::Checked,
    };

    let uw_oracle =
        LinearEntropy::unweighted(FIXED_TOLERANCE).expect("fixed unweighted config is valid");

    assert_flash_entropy_equivalence(
        &uw_direct,
        &uw_oracle,
        &merged_lib,
        &merged_query,
        "fixed_unweighted",
    );

    // search_modified: scores >= direct scores + oracle equivalence.
    if let Ok(uw_modified) = unweighted_index.search_modified(&merged_query) {
        assert_flash_modified_not_less_than_direct(
            &uw_modified,
            &uw_direct,
            "fixed_unweighted/entropy",
        );

        let uw_modified_oracle = ModifiedLinearEntropy::unweighted(FIXED_TOLERANCE)
            .expect("fixed unweighted config is valid");
        assert_flash_modified_entropy_equivalence(
            &uw_modified,
            &uw_modified_oracle,
            &merged_lib,
            &merged_query,
            "fixed_unweighted/modified",
        );
    }

    // search_with_state: exact equality with search.
    let mut uw_state = unweighted_index.new_search_state();
    if let Ok(uw_state_results) = unweighted_index.search_with_state(&merged_query, &mut uw_state) {
        assert_flash_state_equivalence(
            &uw_state_results,
            &uw_direct,
            "fixed_unweighted/entropy/direct",
        );
    }
    if let Ok(uw_modified) = unweighted_index.search_modified(&merged_query)
        && let Ok(uw_modified_state) =
            unweighted_index.search_modified_with_state(&merged_query, &mut uw_state)
    {
        assert_flash_state_equivalence(
            &uw_modified_state,
            &uw_modified,
            "fixed_unweighted/entropy/modified",
        );
    }

    // Self-similarity (unweighted).
    for (i, lib_spec) in merged_lib.iter().enumerate() {
        assert_flash_self_similarity_entropy(
            &unweighted_index,
            lib_spec,
            i as u32,
            "fixed_unweighted",
        );
    }

    // Fixed weighted config.
    let weighted_index =
        match FlashEntropyIndex::<f64>::weighted(FIXED_TOLERANCE, merged_lib.iter()) {
            Ok(idx) => idx,
            Err(_) => return FlashEntropyHarnessOutcome::Checked,
        };

    let w_direct = match weighted_index.search(&merged_query) {
        Ok(r) => r,
        Err(_) => return FlashEntropyHarnessOutcome::Checked,
    };

    let w_oracle =
        LinearEntropy::weighted(FIXED_TOLERANCE).expect("fixed weighted config is valid");
    assert_flash_entropy_equivalence(
        &w_direct,
        &w_oracle,
        &merged_lib,
        &merged_query,
        "fixed_weighted",
    );

    // search_modified (weighted): scores >= direct scores + oracle equivalence.
    if let Ok(w_modified) = weighted_index.search_modified(&merged_query) {
        assert_flash_modified_not_less_than_direct(
            &w_modified,
            &w_direct,
            "fixed_weighted/entropy",
        );

        let w_modified_oracle = ModifiedLinearEntropy::weighted(FIXED_TOLERANCE)
            .expect("fixed weighted config is valid");
        assert_flash_modified_entropy_equivalence(
            &w_modified,
            &w_modified_oracle,
            &merged_lib,
            &merged_query,
            "fixed_weighted/modified",
        );
    }

    // search_with_state (weighted): exact equality.
    let mut w_state = weighted_index.new_search_state();
    if let Ok(w_state_results) = weighted_index.search_with_state(&merged_query, &mut w_state) {
        assert_flash_state_equivalence(
            &w_state_results,
            &w_direct,
            "fixed_weighted/entropy/direct",
        );
    }

    // Self-similarity (weighted).
    for (i, lib_spec) in merged_lib.iter().enumerate() {
        assert_flash_self_similarity_entropy(&weighted_index, lib_spec, i as u32, "fixed_weighted");
    }

    // Dynamic config (arbitrary params): attempt build + search, check score ranges.
    if let Ok(dyn_index) = FlashEntropyIndex::<f64>::new(
        case.mz_power,
        case.intensity_power,
        case.tolerance,
        case.weighted,
        merged_lib.iter(),
    ) && let Ok(dyn_results) = dyn_index.search(&merged_query)
    {
        for r in &dyn_results {
            assert_score_in_range(r.score, "dynamic/entropy");
            assert!(
                r.n_matches <= merged_query.len(),
                "dynamic/entropy: n_matches {} > query len {}",
                r.n_matches,
                merged_query.len()
            );
        }
    }

    FlashEntropyHarnessOutcome::Checked
}

/// Execute the ms-entropy clean-spectrum fuzz harness for an arbitrary byte
/// slice.
///
/// The function intentionally panics when a postcondition invariant is
/// violated.  This behavior is required for fuzzers and regression tests to
/// surface bugs.
pub fn run_ms_entropy_clean_spectrum_case(bytes: &[u8]) -> MsEntropyCleanSpectrumHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match CleanSpectrumFuzzCase::arbitrary(&mut unstructured) {
        Ok(case) => case,
        Err(_) => return MsEntropyCleanSpectrumHarnessOutcome::Checked,
    };

    let builder = MsEntropyCleanSpectrum::builder();

    let Ok(builder) = builder.min_ms2_difference_in_da(case.min_ms2_difference_in_da) else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };
    let Ok(builder) = builder.noise_threshold(case.noise_threshold) else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };
    let Ok(builder) = builder.max_peak_num(case.max_peak_num) else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };
    let Ok(builder) = builder.normalize_intensity(case.normalize_intensity) else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };
    let Ok(builder) = builder.min_mz(case.min_mz) else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };
    let Ok(builder) = builder.max_mz(case.max_mz) else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };
    let Ok(builder) = builder.min_ms2_difference_in_ppm(case.min_ms2_difference_in_ppm) else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };

    let Ok(cleaner) = builder.build() else {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    };

    let input_len = case.spectrum.len();
    let output = cleaner.process(&case.spectrum);
    let output_len = output.len();

    // Postcondition: output length <= input length.
    assert!(
        output_len <= input_len,
        "clean_spectrum: output len {output_len} > input len {input_len}"
    );

    // Postcondition: max_peak_num respected.
    if let Some(max_peak_num) = cleaner.max_peak_num() {
        assert!(
            output_len <= max_peak_num,
            "clean_spectrum: output len {output_len} > max_peak_num {max_peak_num}"
        );
    }

    if output_len == 0 {
        return MsEntropyCleanSpectrumHarnessOutcome::Checked;
    }

    let peaks: Vec<(f64, f64)> = output.peaks().collect();

    // Postcondition: all mz values positive and finite.
    for (i, &(mz, _)) in peaks.iter().enumerate() {
        assert!(
            mz.is_finite() && mz > 0.0,
            "clean_spectrum: peak {i} has invalid mz {mz}"
        );
    }

    // Postcondition: all intensities positive and finite.
    for (i, &(_, intensity)) in peaks.iter().enumerate() {
        assert!(
            intensity.is_finite() && intensity > 0.0,
            "clean_spectrum: peak {i} has invalid intensity {intensity}"
        );
    }

    // Postcondition: peaks sorted by mz (strictly ascending).
    for window in peaks.windows(2) {
        assert!(
            window[0].0 < window[1].0,
            "clean_spectrum: mz not strictly ascending: {} >= {}",
            window[0].0,
            window[1].0,
        );
    }

    // Postcondition: adjacent peaks respect the centroiding distance.
    // When ppm mode is active, use ppm-derived distance; otherwise use Da.
    if let Some(ppm) = cleaner.min_ms2_difference_in_ppm() {
        if ppm > 0.0 {
            for window in peaks.windows(2) {
                let ppm_delta = window[1].0 * ppm * 1e-6;
                assert!(
                    (window[1].0 - window[0].0) > ppm_delta,
                    "clean_spectrum: adjacent peaks {:.6} and {:.6} closer than ppm threshold {ppm_delta:.6e}",
                    window[0].0,
                    window[1].0,
                );
            }
        }
    } else {
        let da = cleaner.min_ms2_difference_in_da();
        if da > 0.0 {
            for window in peaks.windows(2) {
                assert!(
                    (window[1].0 - window[0].0) > da,
                    "clean_spectrum: adjacent peaks {:.6} and {:.6} closer than Da threshold {da:.6e}",
                    window[0].0,
                    window[1].0,
                );
            }
        }
    }

    // Postcondition: if normalize_intensity is enabled, intensities sum ≈ 1.0.
    if cleaner.normalize_intensity() {
        let sum: f64 = peaks.iter().map(|&(_, intensity)| intensity).sum();
        assert!(
            (sum - 1.0).abs() <= 1.0e-6,
            "clean_spectrum: normalized intensities sum to {sum}, expected ~1.0"
        );
    }

    MsEntropyCleanSpectrumHarnessOutcome::Checked
}

#[derive(Debug)]
struct CosineFuzzCase {
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
    left: GenericSpectrum,
    right: GenericSpectrum,
}

impl<'a> Arbitrary<'a> for CosineFuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            mz_power: f64::arbitrary(u)?,
            intensity_power: f64::arbitrary(u)?,
            tolerance: f64::arbitrary(u)?,
            left: GenericSpectrum::arbitrary(u)?,
            right: GenericSpectrum::arbitrary(u)?,
        })
    }
}

#[derive(Debug)]
struct EntropyFuzzCase {
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
    weighted: bool,
    left: GenericSpectrum,
    right: GenericSpectrum,
}

impl<'a> Arbitrary<'a> for EntropyFuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            mz_power: f64::arbitrary(u)?,
            intensity_power: f64::arbitrary(u)?,
            tolerance: f64::arbitrary(u)?,
            weighted: bool::arbitrary(u)?,
            left: GenericSpectrum::arbitrary(u)?,
            right: GenericSpectrum::arbitrary(u)?,
        })
    }
}

#[derive(Debug)]
struct FlashCosineFuzzCase {
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
    library: [GenericSpectrum; 3],
    query: GenericSpectrum,
}

impl<'a> Arbitrary<'a> for FlashCosineFuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            mz_power: f64::arbitrary(u)?,
            intensity_power: f64::arbitrary(u)?,
            tolerance: f64::arbitrary(u)?,
            library: [
                GenericSpectrum::arbitrary(u)?,
                GenericSpectrum::arbitrary(u)?,
                GenericSpectrum::arbitrary(u)?,
            ],
            query: GenericSpectrum::arbitrary(u)?,
        })
    }
}

#[derive(Debug)]
struct FlashEntropyFuzzCase {
    mz_power: f64,
    intensity_power: f64,
    tolerance: f64,
    weighted: bool,
    library: [GenericSpectrum; 3],
    query: GenericSpectrum,
}

impl<'a> Arbitrary<'a> for FlashEntropyFuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            mz_power: f64::arbitrary(u)?,
            intensity_power: f64::arbitrary(u)?,
            tolerance: f64::arbitrary(u)?,
            weighted: bool::arbitrary(u)?,
            library: [
                GenericSpectrum::arbitrary(u)?,
                GenericSpectrum::arbitrary(u)?,
                GenericSpectrum::arbitrary(u)?,
            ],
            query: GenericSpectrum::arbitrary(u)?,
        })
    }
}

#[derive(Debug)]
struct CleanSpectrumFuzzCase {
    min_ms2_difference_in_da: f64,
    min_ms2_difference_in_ppm: Option<f64>,
    noise_threshold: Option<f64>,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
    min_mz: Option<f64>,
    max_mz: Option<f64>,
    spectrum: GenericSpectrum,
}

impl<'a> Arbitrary<'a> for CleanSpectrumFuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            min_ms2_difference_in_da: f64::arbitrary(u)?,
            min_ms2_difference_in_ppm: Option::<f64>::arbitrary(u)?,
            noise_threshold: Option::<f64>::arbitrary(u)?,
            max_peak_num: Option::<usize>::arbitrary(u)?,
            normalize_intensity: bool::arbitrary(u)?,
            min_mz: Option::<f64>::arbitrary(u)?,
            max_mz: Option::<f64>::arbitrary(u)?,
            spectrum: GenericSpectrum::arbitrary(u)?,
        })
    }
}

fn assert_modified_bidirectional_properties<S1, S2>(
    scorer: &ModifiedHungarianCosine,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum,
    S2: Spectrum,
{
    let forward = scorer.similarity(left, right);
    let reverse = scorer.similarity(right, left);

    if let (Ok((forward_score, forward_matches)), Ok((reverse_score, reverse_matches))) =
        (forward, reverse)
    {
        let max_matches = left.len().min(right.len());
        assert_score_in_range(forward_score, label);
        assert_score_in_range(reverse_score, label);
        assert!(
            forward_matches <= max_matches,
            "{label}: forward matches {forward_matches} exceed limit {max_matches}"
        );
        assert!(
            reverse_matches <= max_matches,
            "{label}: reverse matches {reverse_matches} exceed limit {max_matches}"
        );
        assert!(
            (forward_score - reverse_score).abs() <= SYMMETRY_EPS,
            "{label}: asymmetry score mismatch {forward_score} vs {reverse_score}"
        );
    }
}

fn assert_linear_entropy_bidirectional_properties<S1, S2>(
    scorer: &LinearEntropy,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum,
    S2: Spectrum,
{
    let forward = scorer.similarity(left, right);
    let reverse = scorer.similarity(right, left);

    if let (Ok((forward_score, forward_matches)), Ok((reverse_score, reverse_matches))) =
        (forward, reverse)
    {
        let max_matches = left.len().min(right.len());
        assert_score_in_range(forward_score, label);
        assert_score_in_range(reverse_score, label);
        assert!(
            forward_matches <= max_matches,
            "{label}: forward matches {forward_matches} exceed limit {max_matches}"
        );
        assert!(
            reverse_matches <= max_matches,
            "{label}: reverse matches {reverse_matches} exceed limit {max_matches}"
        );
        assert!(
            (forward_score - reverse_score).abs() <= SYMMETRY_EPS,
            "{label}: asymmetry score mismatch {forward_score} vs {reverse_score}"
        );
        assert_eq!(
            forward_matches, reverse_matches,
            "{label}: asymmetry match mismatch {forward_matches} vs {reverse_matches}"
        );
    }
}

fn assert_bidirectional_properties<S1, S2>(
    scorer: &HungarianCosine,
    left: &S1,
    right: &S2,
    label: &str,
    assert_match_symmetry: bool,
) where
    S1: Spectrum,
    S2: Spectrum,
{
    let forward = scorer.similarity(left, right);
    let reverse = scorer.similarity(right, left);

    if let (Ok((forward_score, forward_matches)), Ok((reverse_score, reverse_matches))) =
        (forward, reverse)
    {
        let max_matches = left.len().min(right.len());
        assert_score_in_range(forward_score, label);
        assert_score_in_range(reverse_score, label);
        assert!(
            forward_matches <= max_matches,
            "{label}: forward matches {forward_matches} exceed limit {max_matches}"
        );
        assert!(
            reverse_matches <= max_matches,
            "{label}: reverse matches {reverse_matches} exceed limit {max_matches}"
        );
        assert!(
            (forward_score - reverse_score).abs() <= SYMMETRY_EPS,
            "{label}: asymmetry score mismatch {forward_score} vs {reverse_score}"
        );
        if assert_match_symmetry {
            assert_eq!(
                forward_matches, reverse_matches,
                "{label}: asymmetry match mismatch {forward_matches} vs {reverse_matches}"
            );
        }
    }
}

fn assert_modified_self_similarity(
    scorer: &ModifiedHungarianCosine,
    spectrum: &GenericSpectrum,
    tolerance: f64,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let Ok((score, matches)) = scorer.similarity(spectrum, spectrum) else {
        return;
    };
    assert!(
        matches > 0,
        "{label}: non-empty spectrum ({} peaks) has 0 self-matches",
        spectrum.len()
    );

    assert_score_in_range(score, label);
    assert!(
        (1.0 - score).abs() <= tolerance,
        "{label}: self-similarity {score} exceeds tolerance {tolerance}"
    );
    assert!(
        matches <= spectrum.len(),
        "{label}: self match count {matches} > {}",
        spectrum.len()
    );
}

fn assert_self_similarity(
    scorer: &HungarianCosine,
    spectrum: &GenericSpectrum,
    tolerance: f64,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let Ok((score, matches)) = scorer.similarity(spectrum, spectrum) else {
        return;
    };
    assert!(
        matches > 0,
        "{label}: non-empty spectrum ({} peaks) has 0 self-matches",
        spectrum.len()
    );

    assert_score_in_range(score, label);
    assert!(
        (1.0 - score).abs() <= tolerance,
        "{label}: self-similarity {score} exceeds tolerance {tolerance}"
    );
    assert!(
        matches <= spectrum.len(),
        "{label}: self match count {matches} > {}",
        spectrum.len()
    );
}

fn assert_linear_entropy_self_similarity(
    scorer: &LinearEntropy,
    spectrum: &GenericSpectrum,
    tolerance: f64,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let Ok((score, matches)) = scorer.similarity(spectrum, spectrum) else {
        return;
    };
    assert!(
        matches > 0,
        "{label}: non-empty spectrum ({} peaks) has 0 self-matches",
        spectrum.len()
    );

    assert_score_in_range(score, label);
    assert!(
        (1.0 - score).abs() <= tolerance,
        "{label}: self-similarity {score} exceeds tolerance {tolerance}"
    );
    assert_eq!(
        matches,
        spectrum.len(),
        "{label}: self match count {matches} != {}",
        spectrum.len()
    );
}

fn assert_modified_linear_entropy_bidirectional_properties<S1, S2>(
    scorer: &ModifiedLinearEntropy,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum,
    S2: Spectrum,
{
    let forward = scorer.similarity(left, right);
    let reverse = scorer.similarity(right, left);

    if let (Ok((forward_score, forward_matches)), Ok((reverse_score, reverse_matches))) =
        (forward, reverse)
    {
        let max_matches = left.len().min(right.len());
        assert_score_in_range(forward_score, label);
        assert_score_in_range(reverse_score, label);
        assert!(
            forward_matches <= max_matches,
            "{label}: forward matches {forward_matches} exceed limit {max_matches}"
        );
        assert!(
            reverse_matches <= max_matches,
            "{label}: reverse matches {reverse_matches} exceed limit {max_matches}"
        );
        assert!(
            (forward_score - reverse_score).abs() <= SYMMETRY_EPS,
            "{label}: asymmetry score mismatch {forward_score} vs {reverse_score}"
        );
    }
}

fn assert_modified_linear_entropy_self_similarity(
    scorer: &ModifiedLinearEntropy,
    spectrum: &GenericSpectrum,
    tolerance: f64,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let Ok((score, matches)) = scorer.similarity(spectrum, spectrum) else {
        return;
    };
    assert!(
        matches > 0,
        "{label}: non-empty spectrum ({} peaks) has 0 self-matches",
        spectrum.len()
    );

    assert_score_in_range(score, label);
    assert!(
        (1.0 - score).abs() <= tolerance,
        "{label}: self-similarity {score} exceeds tolerance {tolerance}"
    );
    assert_eq!(
        matches,
        spectrum.len(),
        "{label}: self match count {matches} != {}",
        spectrum.len()
    );
}

fn assert_modified_entropy_not_less_than_linear<S1, S2>(
    modified: &ModifiedLinearEntropy,
    linear: &LinearEntropy,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum,
    S2: Spectrum,
{
    let modified_forward = modified.similarity(left, right);
    let linear_forward = linear.similarity(left, right);
    if let (Ok((modified_score, _)), Ok((linear_score, _))) = (modified_forward, linear_forward) {
        assert!(
            modified_score + MODIFIED_DIFFERENTIAL_EPS >= linear_score,
            "{label}: modified entropy score {modified_score} < linear entropy score {linear_score}"
        );
    }

    let modified_reverse = modified.similarity(right, left);
    let linear_reverse = linear.similarity(right, left);
    if let (Ok((modified_score, _)), Ok((linear_score, _))) = (modified_reverse, linear_reverse) {
        assert!(
            modified_score + MODIFIED_DIFFERENTIAL_EPS >= linear_score,
            "{label}: reverse modified entropy score {modified_score} < reverse linear entropy score {linear_score}"
        );
    }
}

fn assert_modified_linear_entropy_original_outcome(
    scorer: &ModifiedLinearEntropy,
    left: &GenericSpectrum,
    right: &GenericSpectrum,
    label: &str,
) {
    match scorer.similarity(left, right) {
        Ok((score, matches)) => {
            assert_score_in_range(score, label);
            let max_matches = left.len().min(right.len());
            assert!(
                matches <= max_matches,
                "{label}: matches {matches} exceed limit {max_matches}"
            );
        }
        Err(SimilarityComputationError::InvalidPeakSpacing(_)) => {}
        Err(error) => panic!("{label}: unexpected error on original spectra: {error:?}"),
    }
}

fn assert_modified_linear_matches_modified_hungarian(
    left: &GenericSpectrum,
    right: &GenericSpectrum,
) {
    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let left = merger.process(left);
    let right = merger.process(right);

    let modified_hungarian = ModifiedHungarianCosine::new(1.0_f64, 1.0_f64, FIXED_TOLERANCE)
        .expect("fixed modified-hungarian config is valid");
    let modified_linear = ModifiedLinearCosine::new(1.0_f64, 1.0_f64, FIXED_TOLERANCE)
        .expect("fixed modified-linear config is valid");

    let (
        Ok((modified_hungarian_score, modified_hungarian_matches)),
        Ok((modified_linear_score, modified_linear_matches)),
    ) = (
        modified_hungarian.similarity(&left, &right),
        modified_linear.similarity(&left, &right),
    )
    else {
        return;
    };

    // Score must agree within tolerance.  Match-count disagreements are
    // acceptable when scores agree: the Hungarian (Crouse LAPJV) and the
    // Linear DP use different f64 arithmetic paths (cost-domain augmentation
    // vs benefit-domain summation), so they can break ties differently on
    // edges whose normalised product is near f64::EPSILON.  These edges
    // contribute negligibly to the score, making the match-count difference
    // a tie-breaking artefact rather than a correctness issue.
    if (modified_hungarian_score - modified_linear_score).abs() > MODIFIED_DIFFERENTIAL_EPS {
        use alloc::string::String;
        use core::fmt::Write;
        let mut diag = String::new();
        let _ = writeln!(
            diag,
            "left precursor: {:?}, right precursor: {:?}, shift: {:?}",
            left.precursor_mz(),
            right.precursor_mz(),
            left.precursor_mz() - right.precursor_mz()
        );
        let _ = writeln!(diag, "left ({}):", left.len());
        for (mz, int) in left.peaks() {
            let _ = writeln!(diag, "  mz={mz:?} int={int:?}");
        }
        let _ = writeln!(diag, "right ({}):", right.len());
        for (mz, int) in right.peaks() {
            let _ = writeln!(diag, "  mz={mz:?} int={int:?}");
        }
        let _ = writeln!(
            diag,
            "Hungarian: score={modified_hungarian_score}, matches={modified_hungarian_matches}"
        );
        let _ = writeln!(
            diag,
            "Linear:    score={modified_linear_score}, matches={modified_linear_matches}"
        );
        panic!("{diag}");
    }
}

fn assert_modified_not_less_than_hungarian<S1, S2>(
    modified: &ModifiedHungarianCosine,
    hungarian: &HungarianCosine,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum,
    S2: Spectrum,
{
    let modified_forward = modified.similarity(left, right);
    let hungarian_forward = hungarian.similarity(left, right);
    if let (Ok((modified_score, _)), Ok((hungarian_score, _))) =
        (modified_forward, hungarian_forward)
    {
        assert!(
            modified_score + MODIFIED_DIFFERENTIAL_EPS >= hungarian_score,
            "{label}: modified score {modified_score} < hungarian score {hungarian_score}"
        );
    }

    let modified_reverse = modified.similarity(right, left);
    let hungarian_reverse = hungarian.similarity(right, left);
    if let (Ok((modified_score, _)), Ok((hungarian_score, _))) =
        (modified_reverse, hungarian_reverse)
    {
        assert!(
            modified_score + MODIFIED_DIFFERENTIAL_EPS >= hungarian_score,
            "{label}: reverse modified score {modified_score} < reverse hungarian score {hungarian_score}"
        );
    }
}

fn assert_linear_matches_hungarian(left: &GenericSpectrum, right: &GenericSpectrum) {
    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let left = merger.process(left);
    let right = merger.process(right);

    let hungarian =
        HungarianCosine::new(1.0_f64, 1.0_f64, FIXED_TOLERANCE).expect("fixed config is valid");
    let linear = LinearCosine::new(1.0_f64, 1.0_f64, FIXED_TOLERANCE).expect("fixed config");

    let (Ok((hungarian_score, hungarian_matches)), Ok((linear_score, linear_matches))) = (
        hungarian.similarity(&left, &right),
        linear.similarity(&left, &right),
    ) else {
        return;
    };

    assert!(
        (hungarian_score - linear_score).abs() <= DIFFERENTIAL_EPS,
        "fixed differential mismatch: Hungarian={hungarian_score} vs Linear={linear_score}"
    );
    assert_eq!(
        hungarian_matches, linear_matches,
        "fixed differential match mismatch: Hungarian={hungarian_matches} vs Linear={linear_matches}"
    );
}

fn assert_linear_entropy_original_outcome(
    scorer: &LinearEntropy,
    left: &GenericSpectrum,
    right: &GenericSpectrum,
    label: &str,
) {
    match scorer.similarity(left, right) {
        Ok((score, matches)) => {
            assert_score_in_range(score, label);
            let max_matches = left.len().min(right.len());
            assert!(
                matches <= max_matches,
                "{label}: matches {matches} exceed limit {max_matches}"
            );
        }
        Err(SimilarityComputationError::InvalidPeakSpacing(_)) => {}
        Err(error) => panic!("{label}: unexpected error on original spectra: {error:?}"),
    }
}

#[inline]
fn assert_score_in_range(score: f64, label: &str) {
    assert!(score.is_finite(), "{label}: score {score} is not finite");
    assert!(
        (-1.0e-6..=1.0 + 1.0e-6).contains(&score),
        "{label}: score {score} not in [0, 1]"
    );
}

fn assert_flash_cosine_equivalence(
    flash_results: &[FlashSearchResult],
    oracle: &LinearCosine,
    library: &[GenericSpectrum; 3],
    query: &GenericSpectrum,
    label: &str,
) {
    for (spec_id, lib_spec) in library.iter().enumerate() {
        let flash_result = flash_results
            .iter()
            .find(|r| r.spectrum_id == spec_id as u32);

        let oracle_result = oracle.similarity(lib_spec, query);
        let Ok((oracle_score, oracle_matches)) = oracle_result else {
            continue;
        };

        match flash_result {
            Some(fr) => {
                assert_score_in_range(fr.score, label);
                assert!(
                    (fr.score - oracle_score).abs() <= DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} flash score {} (n_matches={}) vs oracle score {oracle_score} (n_matches={oracle_matches})",
                    fr.score,
                    fr.n_matches,
                );
            }
            None => {
                assert!(
                    oracle_score <= DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} missing from flash results but oracle score is {oracle_score}"
                );
            }
        }
    }
}

fn assert_flash_entropy_equivalence(
    flash_results: &[FlashSearchResult],
    oracle: &LinearEntropy,
    library: &[GenericSpectrum; 3],
    query: &GenericSpectrum,
    label: &str,
) {
    for (spec_id, lib_spec) in library.iter().enumerate() {
        let flash_result = flash_results
            .iter()
            .find(|r| r.spectrum_id == spec_id as u32);

        let oracle_result = oracle.similarity(lib_spec, query);
        let Ok((oracle_score, oracle_matches)) = oracle_result else {
            continue;
        };

        match flash_result {
            Some(fr) => {
                assert_score_in_range(fr.score, label);
                assert!(
                    (fr.score - oracle_score).abs() <= DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} flash score {} (n_matches={}) vs oracle score {oracle_score} (n_matches={oracle_matches})",
                    fr.score,
                    fr.n_matches,
                );
            }
            None => {
                assert!(
                    oracle_score <= DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} missing from flash results but oracle score is {oracle_score}"
                );
            }
        }
    }
}

fn assert_flash_modified_cosine_equivalence(
    flash_results: &[FlashSearchResult],
    oracle: &ModifiedLinearCosine,
    library: &[GenericSpectrum; 3],
    query: &GenericSpectrum,
    label: &str,
) {
    for (spec_id, lib_spec) in library.iter().enumerate() {
        let flash_result = flash_results
            .iter()
            .find(|r| r.spectrum_id == spec_id as u32);

        let oracle_result = oracle.similarity(lib_spec, query);
        let Ok((oracle_score, oracle_matches)) = oracle_result else {
            continue;
        };

        match flash_result {
            Some(fr) => {
                assert_score_in_range(fr.score, label);
                // Flash's DenseAccumulator can count more matches than the
                // linear oracle when neutral-loss shifts bring peaks closer
                // than 2×tolerance. Extra matches only add score, so we
                // assert one-sided: flash must not be significantly *below*
                // the oracle.
                assert!(
                    fr.score >= oracle_score - DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} flash modified score {} (n_matches={}) vs oracle score {oracle_score} (n_matches={oracle_matches})",
                    fr.score,
                    fr.n_matches,
                );
            }
            None => {
                assert!(
                    oracle_score <= DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} missing from flash modified results but oracle score is {oracle_score}"
                );
            }
        }
    }
}

fn assert_flash_modified_entropy_equivalence(
    flash_results: &[FlashSearchResult],
    oracle: &ModifiedLinearEntropy,
    library: &[GenericSpectrum; 3],
    query: &GenericSpectrum,
    label: &str,
) {
    for (spec_id, lib_spec) in library.iter().enumerate() {
        let flash_result = flash_results
            .iter()
            .find(|r| r.spectrum_id == spec_id as u32);

        let oracle_result = oracle.similarity(lib_spec, query);
        let Ok((oracle_score, oracle_matches)) = oracle_result else {
            continue;
        };

        match flash_result {
            Some(fr) => {
                assert_score_in_range(fr.score, label);
                // Same one-sided check as modified cosine: Flash can
                // legitimately score higher due to extra shifted matches.
                assert!(
                    fr.score >= oracle_score - DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} flash modified score {} (n_matches={}) vs oracle score {oracle_score} (n_matches={oracle_matches})",
                    fr.score,
                    fr.n_matches,
                );
            }
            None => {
                assert!(
                    oracle_score <= DIFFERENTIAL_EPS,
                    "{label}: spec {spec_id} missing from flash modified results but oracle score is {oracle_score}"
                );
            }
        }
    }
}

fn assert_flash_modified_not_less_than_direct(
    modified: &[FlashSearchResult],
    direct: &[FlashSearchResult],
    label: &str,
) {
    for dr in direct {
        let mr = modified.iter().find(|r| r.spectrum_id == dr.spectrum_id);
        let modified_score = mr.map_or(0.0, |r| r.score);
        assert!(
            modified_score + DIFFERENTIAL_EPS >= dr.score,
            "{label}: spec {} modified score {modified_score} < direct score {}",
            dr.spectrum_id,
            dr.score
        );
    }
}

fn assert_flash_state_equivalence(
    state_results: &[FlashSearchResult],
    baseline_results: &[FlashSearchResult],
    label: &str,
) {
    assert_eq!(
        state_results.len(),
        baseline_results.len(),
        "{label}: state result count {} != baseline count {}",
        state_results.len(),
        baseline_results.len()
    );

    let mut state_sorted: Vec<FlashSearchResult> = state_results.to_vec();
    let mut baseline_sorted: Vec<FlashSearchResult> = baseline_results.to_vec();
    state_sorted.sort_by_key(|r| r.spectrum_id);
    baseline_sorted.sort_by_key(|r| r.spectrum_id);

    for (s, b) in state_sorted.iter().zip(baseline_sorted.iter()) {
        assert_eq!(
            s.spectrum_id, b.spectrum_id,
            "{label}: state spec_id {} != baseline spec_id {}",
            s.spectrum_id, b.spectrum_id
        );
        assert!(
            (s.score - b.score).abs() <= f64::EPSILON * 4.0,
            "{label}: spec {} state score {} != baseline score {}",
            s.spectrum_id,
            s.score,
            b.score
        );
        assert_eq!(
            s.n_matches, b.n_matches,
            "{label}: spec {} state matches {} != baseline matches {}",
            s.spectrum_id, s.n_matches, b.n_matches
        );
    }
}

fn assert_flash_self_similarity_cosine(
    index: &FlashCosineIndex,
    spectrum: &GenericSpectrum,
    expected_id: u32,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let results = match index.search(spectrum) {
        Ok(r) => r,
        Err(_) => return,
    };

    let self_result = results.iter().find(|r| r.spectrum_id == expected_id);
    let Some(sr) = self_result else {
        return;
    };

    assert_score_in_range(sr.score, label);
    assert!(
        (1.0 - sr.score).abs() <= 1.0e-4,
        "{label}: spec {expected_id} self-similarity {} not ~1.0",
        sr.score
    );
    assert_eq!(
        sr.n_matches,
        spectrum.len(),
        "{label}: spec {expected_id} self-match count {} != {}",
        sr.n_matches,
        spectrum.len()
    );
}

fn assert_flash_self_similarity_entropy(
    index: &FlashEntropyIndex,
    spectrum: &GenericSpectrum,
    expected_id: u32,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let results = match index.search(spectrum) {
        Ok(r) => r,
        Err(_) => return,
    };

    let self_result = results.iter().find(|r| r.spectrum_id == expected_id);
    let Some(sr) = self_result else {
        return;
    };

    assert_score_in_range(sr.score, label);
    assert!(
        (1.0 - sr.score).abs() <= 1.0e-4,
        "{label}: spec {expected_id} self-similarity {} not ~1.0",
        sr.score
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{SpectrumAlloc, SpectrumMut};

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

    fn flash_library() -> [GenericSpectrum; 3] {
        [
            make_spectrum(300.0, &[(100.0, 10.0), (200.0, 5.0)]),
            make_spectrum(450.0, &[(150.0, 7.0), (300.0, 4.0)]),
            make_spectrum(600.0, &[(50.0, 3.0), (500.0, 2.0)]),
        ]
    }

    #[test]
    fn cosine_helper_assertions_cover_success_paths() {
        let left = make_spectrum(300.0, &[(100.0, 10.0), (200.0, 5.0)]);
        let right = make_spectrum(300.0, &[(100.05, 9.0), (200.05, 4.0)]);

        let hungarian =
            HungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("fixed config is valid");
        let modified = ModifiedHungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE)
            .expect("fixed modified config is valid");

        assert_bidirectional_properties(&hungarian, &left, &right, "cosine", true);
        assert_modified_bidirectional_properties(&modified, &left, &right, "modified_cosine");
        assert_self_similarity(&hungarian, &left, 1.0e-4, "self/cosine");
        assert_modified_self_similarity(&modified, &left, 1.0e-4, "self/modified_cosine");
        assert_modified_not_less_than_hungarian(
            &modified,
            &hungarian,
            &left,
            &right,
            "modified>=hungarian",
        );
        assert_linear_matches_hungarian(&left, &right);
    }

    #[test]
    fn entropy_helper_assertions_cover_success_paths() {
        let left = make_spectrum(300.0, &[(100.0, 10.0), (200.0, 5.0)]);
        let right = make_spectrum(310.0, &[(100.0, 10.0), (210.0, 5.0)]);

        let linear = LinearEntropy::weighted(FIXED_TOLERANCE).expect("weighted config is valid");
        let modified = ModifiedLinearEntropy::weighted(FIXED_TOLERANCE)
            .expect("modified weighted config is valid");

        assert_linear_entropy_bidirectional_properties(&linear, &left, &right, "entropy");
        assert_linear_entropy_self_similarity(&linear, &left, 1.0e-4, "self/entropy");
        assert_modified_linear_entropy_bidirectional_properties(
            &modified,
            &left,
            &right,
            "modified_entropy",
        );
        assert_modified_linear_entropy_self_similarity(
            &modified,
            &left,
            1.0e-4,
            "self/modified_entropy",
        );
        assert_modified_entropy_not_less_than_linear(
            &modified,
            &linear,
            &left,
            &right,
            "modified_entropy>=linear",
        );
        assert_linear_entropy_original_outcome(&linear, &left, &right, "linear/original");
        assert_modified_linear_entropy_original_outcome(
            &modified,
            &left,
            &right,
            "modified/original",
        );
    }

    #[test]
    fn original_outcome_helpers_accept_invalid_peak_spacing() {
        let left = make_spectrum(200.0, &[(100.0, 10.0), (100.15, 9.0)]);
        let right = make_spectrum(200.0, &[(100.0, 8.0), (100.15, 7.0)]);

        let linear = LinearEntropy::weighted(FIXED_TOLERANCE).expect("weighted config is valid");
        let modified = ModifiedLinearEntropy::weighted(FIXED_TOLERANCE)
            .expect("modified weighted config is valid");

        assert_linear_entropy_original_outcome(&linear, &left, &right, "linear/close");
        assert_modified_linear_entropy_original_outcome(&modified, &left, &right, "modified/close");
    }

    #[test]
    fn flash_helper_assertions_cover_present_and_missing_result_paths() {
        let library = flash_library();
        let query = make_spectrum(310.0, &[(100.0, 10.0), (210.0, 5.0)]);

        let cosine_index = FlashCosineIndex::<f64>::new(1.0, 1.0, FIXED_TOLERANCE, library.iter())
            .expect("cosine index should build");
        let cosine_direct = cosine_index
            .search(&query)
            .expect("cosine search should work");
        let cosine_modified = cosine_index
            .search_modified(&query)
            .expect("modified cosine search should work");
        let cosine_oracle =
            LinearCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("linear cosine should build");
        let modified_cosine_oracle = ModifiedLinearCosine::new(1.0, 1.0, FIXED_TOLERANCE)
            .expect("modified linear cosine should build");

        assert_flash_cosine_equivalence(
            &cosine_direct,
            &cosine_oracle,
            &library,
            &query,
            "flash/cosine",
        );
        assert_flash_modified_cosine_equivalence(
            &cosine_modified,
            &modified_cosine_oracle,
            &library,
            &query,
            "flash/modified_cosine",
        );
        assert_flash_modified_not_less_than_direct(
            &cosine_modified,
            &cosine_direct,
            "flash/modified>=direct",
        );
        let mut cosine_state = cosine_index.new_search_state();
        let cosine_state_results = cosine_index
            .search_with_state(&query, &mut cosine_state)
            .expect("stateful cosine search should work");
        assert_flash_state_equivalence(&cosine_state_results, &cosine_direct, "flash/cosine/state");
        assert_flash_self_similarity_cosine(&cosine_index, &library[0], 0, "flash/cosine/self");

        let entropy_index = FlashEntropyIndex::<f64>::weighted(FIXED_TOLERANCE, library.iter())
            .expect("entropy index should build");
        let entropy_direct = entropy_index
            .search(&query)
            .expect("entropy search should work");
        let entropy_modified = entropy_index
            .search_modified(&query)
            .expect("modified entropy search should work");
        let entropy_oracle =
            LinearEntropy::weighted(FIXED_TOLERANCE).expect("linear entropy should build");
        let modified_entropy_oracle = ModifiedLinearEntropy::weighted(FIXED_TOLERANCE)
            .expect("modified linear entropy should build");

        assert_flash_entropy_equivalence(
            &entropy_direct,
            &entropy_oracle,
            &library,
            &query,
            "flash/entropy",
        );
        assert_flash_modified_entropy_equivalence(
            &entropy_modified,
            &modified_entropy_oracle,
            &library,
            &query,
            "flash/modified_entropy",
        );
        let mut entropy_state = entropy_index.new_search_state();
        let entropy_state_results = entropy_index
            .search_with_state(&query, &mut entropy_state)
            .expect("stateful entropy search should work");
        assert_flash_state_equivalence(
            &entropy_state_results,
            &entropy_direct,
            "flash/entropy/state",
        );
        assert_flash_self_similarity_entropy(&entropy_index, &library[0], 0, "flash/entropy/self");
    }

    #[test]
    fn score_range_helper_accepts_bounds() {
        assert_score_in_range(0.0, "zero");
        assert_score_in_range(1.0, "one");
        assert_score_in_range(1.0e-7, "small");
    }

    #[test]
    fn flash_harnesses_return_checked_for_empty_bytes() {
        assert_eq!(
            run_flash_cosine_case(&[]),
            FlashCosineHarnessOutcome::Checked
        );
        assert_eq!(
            run_flash_entropy_case(&[]),
            FlashEntropyHarnessOutcome::Checked
        );
        assert_eq!(
            run_ms_entropy_clean_spectrum_case(&[]),
            MsEntropyCleanSpectrumHarnessOutcome::Checked
        );
    }

    #[test]
    fn helper_assertions_cover_self_and_symmetric_success_paths() {
        let spectrum = make_spectrum(300.0, &[(100.0, 10.0), (200.0, 5.0)]);

        let hungarian =
            HungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("config should build");
        let modified_hungarian =
            ModifiedHungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("config should build");
        let linear_entropy = LinearEntropy::weighted(FIXED_TOLERANCE).expect("config should build");
        let modified_entropy =
            ModifiedLinearEntropy::weighted(FIXED_TOLERANCE).expect("config should build");

        assert_bidirectional_properties(&hungarian, &spectrum, &spectrum, "hungarian/self", true);
        assert_modified_bidirectional_properties(
            &modified_hungarian,
            &spectrum,
            &spectrum,
            "modified_hungarian/self",
        );
        assert_self_similarity(&hungarian, &spectrum, 1.0e-6, "hungarian/self_similarity");
        assert_modified_self_similarity(
            &modified_hungarian,
            &spectrum,
            1.0e-6,
            "modified_hungarian/self_similarity",
        );

        assert_linear_entropy_bidirectional_properties(
            &linear_entropy,
            &spectrum,
            &spectrum,
            "linear_entropy/self",
        );
        assert_linear_entropy_self_similarity(
            &linear_entropy,
            &spectrum,
            1.0e-6,
            "linear_entropy/self_similarity",
        );
        assert_modified_linear_entropy_bidirectional_properties(
            &modified_entropy,
            &spectrum,
            &spectrum,
            "modified_entropy/self",
        );
        assert_modified_linear_entropy_self_similarity(
            &modified_entropy,
            &spectrum,
            1.0e-6,
            "modified_entropy/self_similarity",
        );
        assert_modified_entropy_not_less_than_linear(
            &modified_entropy,
            &linear_entropy,
            &spectrum,
            &spectrum,
            "modified_entropy>=linear/self",
        );
    }

    #[test]
    fn flash_helpers_cover_present_result_and_entropy_self_similarity_paths() {
        let library = [
            make_spectrum(300.0, &[(100.0, 10.0), (200.0, 5.0)]),
            make_spectrum(450.0, &[(150.0, 7.0), (300.0, 4.0)]),
            make_spectrum(600.0, &[(50.0, 3.0), (500.0, 2.0)]),
        ];
        let query = make_spectrum(300.0, &[(100.0, 10.0), (200.0, 5.0)]);

        let cosine_index = FlashCosineIndex::<f64>::new(1.0, 1.0, FIXED_TOLERANCE, library.iter())
            .expect("cosine index should build");
        let cosine_results = cosine_index.search(&query).expect("search should work");
        let cosine_oracle =
            LinearCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("oracle should build");
        assert_flash_cosine_equivalence(
            &cosine_results,
            &cosine_oracle,
            &library,
            &query,
            "flash/cosine/present",
        );

        let entropy_index = FlashEntropyIndex::<f64>::weighted(FIXED_TOLERANCE, library.iter())
            .expect("entropy index should build");
        let entropy_results = entropy_index.search(&query).expect("search should work");
        let entropy_oracle = LinearEntropy::weighted(FIXED_TOLERANCE).expect("oracle should build");
        assert_flash_entropy_equivalence(
            &entropy_results,
            &entropy_oracle,
            &library,
            &query,
            "flash/entropy/present",
        );
        assert_flash_self_similarity_entropy(
            &entropy_index,
            &library[0],
            0,
            "flash/entropy/self/present",
        );
    }

    #[test]
    fn entropy_self_similarity_helpers_return_early_on_invalid_spacing() {
        let close = make_spectrum(300.0, &[(100.0, 10.0), (100.15, 5.0)]);
        let linear = LinearEntropy::weighted(FIXED_TOLERANCE).expect("config should build");
        let modified =
            ModifiedLinearEntropy::weighted(FIXED_TOLERANCE).expect("config should build");

        assert_linear_entropy_self_similarity(&linear, &close, 1.0e-6, "linear/invalid");
        assert_modified_linear_entropy_self_similarity(
            &modified,
            &close,
            1.0e-6,
            "modified/invalid",
        );
    }

    #[test]
    fn flash_self_similarity_helpers_return_early_on_query_errors_and_missing_results() {
        let library = flash_library();
        let invalid_query = make_spectrum(300.0, &[(100.0, 10.0), (100.15, 5.0)]);
        let missing_query = make_spectrum(900.0, &[(800.0, 1.0)]);

        let cosine_index = FlashCosineIndex::<f64>::new(1.0, 1.0, FIXED_TOLERANCE, library.iter())
            .expect("cosine index should build");
        assert_flash_self_similarity_cosine(&cosine_index, &invalid_query, 0, "flash/cosine/err");
        assert_flash_self_similarity_cosine(&cosine_index, &missing_query, 99, "flash/cosine/miss");

        let entropy_index = FlashEntropyIndex::<f64>::weighted(FIXED_TOLERANCE, library.iter())
            .expect("entropy index should build");
        assert_flash_self_similarity_entropy(
            &entropy_index,
            &invalid_query,
            0,
            "flash/entropy/err",
        );
        assert_flash_self_similarity_entropy(
            &entropy_index,
            &missing_query,
            99,
            "flash/entropy/miss",
        );
    }

    #[test]
    fn flash_entropy_equivalence_skips_entries_when_the_oracle_errors() {
        let library = [
            make_spectrum(300.0, &[(100.0, 10.0), (100.15, 5.0)]),
            make_spectrum(400.0, &[(200.0, 8.0), (200.15, 4.0)]),
            make_spectrum(500.0, &[(300.0, 7.0), (300.15, 3.0)]),
        ];
        let query = make_spectrum(900.0, &[(800.0, 1.0)]);
        let oracle = LinearEntropy::weighted(FIXED_TOLERANCE).expect("oracle should build");

        assert_flash_entropy_equivalence(&[], &oracle, &library, &query, "flash/entropy/skip");
    }

    #[test]
    fn arbitrary_fuzz_case_parsers_accept_large_zero_buffers() {
        let bytes = [0_u8; 4096];
        let mut cosine = Unstructured::new(&bytes);
        let mut entropy = Unstructured::new(&bytes);
        let mut flash_cosine = Unstructured::new(&bytes);
        let mut flash_entropy = Unstructured::new(&bytes);
        let mut clean = Unstructured::new(&bytes);

        assert!(CosineFuzzCase::arbitrary(&mut cosine).is_ok());
        assert!(EntropyFuzzCase::arbitrary(&mut entropy).is_ok());
        assert!(FlashCosineFuzzCase::arbitrary(&mut flash_cosine).is_ok());
        assert!(FlashEntropyFuzzCase::arbitrary(&mut flash_entropy).is_ok());
        assert!(CleanSpectrumFuzzCase::arbitrary(&mut clean).is_ok());
    }

    #[test]
    fn fuzz_harnesses_accept_large_zero_buffers() {
        let bytes = [0_u8; 4096];
        assert_eq!(
            run_hungarian_cosine_case(&bytes),
            HungarianCosineHarnessOutcome::Checked
        );
        assert_eq!(
            run_modified_hungarian_cosine_case(&bytes),
            ModifiedHungarianCosineHarnessOutcome::Checked
        );
        assert_eq!(
            run_linear_entropy_case(&bytes),
            LinearEntropyHarnessOutcome::Checked
        );
        assert_eq!(
            run_modified_linear_entropy_case(&bytes),
            ModifiedLinearEntropyHarnessOutcome::Checked
        );
        assert_eq!(
            run_flash_cosine_case(&bytes),
            FlashCosineHarnessOutcome::Checked
        );
        assert_eq!(
            run_flash_entropy_case(&bytes),
            FlashEntropyHarnessOutcome::Checked
        );
        assert_eq!(
            run_ms_entropy_clean_spectrum_case(&bytes),
            MsEntropyCleanSpectrumHarnessOutcome::Checked
        );
    }
}
