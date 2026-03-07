//! Reusable fuzzing harnesses for similarity implementations.
//!
//! The harness logic lives in the main crate so fuzz targets and regression
//! tests execute the exact same code paths.

use arbitrary::{Arbitrary, Unstructured};

use crate::prelude::ScalarSimilarity;
use crate::structs::{
    GenericSpectrum, HungarianCosine, LinearCosine, LinearEntropy, ModifiedHungarianCosine,
    ModifiedLinearCosine, SimilarityComputationError, SiriusMergeClosePeaks,
};
use crate::traits::{SpectralProcessor, Spectrum};

const FIXED_TOLERANCE: f32 = 0.1;
const SYMMETRY_EPS: f32 = 1.0e-4;
const DIFFERENTIAL_EPS: f32 = 1.0e-4;
const MODIFIED_DIFFERENTIAL_EPS: f32 = 1.0e-6;

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

#[derive(Debug)]
struct CosineFuzzCase {
    mz_power: f32,
    intensity_power: f32,
    tolerance: f32,
    left: GenericSpectrum<f32, f32>,
    right: GenericSpectrum<f32, f32>,
}

impl<'a> Arbitrary<'a> for CosineFuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            mz_power: f32::arbitrary(u)?,
            intensity_power: f32::arbitrary(u)?,
            tolerance: f32::arbitrary(u)?,
            left: GenericSpectrum::<f32, f32>::arbitrary(u)?,
            right: GenericSpectrum::<f32, f32>::arbitrary(u)?,
        })
    }
}

#[derive(Debug)]
struct EntropyFuzzCase {
    mz_power: f32,
    intensity_power: f32,
    tolerance: f32,
    weighted: bool,
    left: GenericSpectrum<f32, f32>,
    right: GenericSpectrum<f32, f32>,
}

impl<'a> Arbitrary<'a> for EntropyFuzzCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            mz_power: f32::arbitrary(u)?,
            intensity_power: f32::arbitrary(u)?,
            tolerance: f32::arbitrary(u)?,
            weighted: bool::arbitrary(u)?,
            left: GenericSpectrum::<f32, f32>::arbitrary(u)?,
            right: GenericSpectrum::<f32, f32>::arbitrary(u)?,
        })
    }
}

fn assert_modified_bidirectional_properties<S1, S2>(
    scorer: &ModifiedHungarianCosine<f32, f32>,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum<Mz = f32, Intensity = f32>,
    S2: Spectrum<Mz = f32, Intensity = f32>,
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
    scorer: &LinearEntropy<f32, f32>,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum<Mz = f32, Intensity = f32>,
    S2: Spectrum<Mz = f32, Intensity = f32>,
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
    scorer: &HungarianCosine<f32, f32>,
    left: &S1,
    right: &S2,
    label: &str,
    assert_match_symmetry: bool,
) where
    S1: Spectrum<Mz = f32, Intensity = f32>,
    S2: Spectrum<Mz = f32, Intensity = f32>,
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
    scorer: &ModifiedHungarianCosine<f32, f32>,
    spectrum: &GenericSpectrum<f32, f32>,
    tolerance: f32,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let Ok((score, matches)) = scorer.similarity(spectrum, spectrum) else {
        return;
    };
    if matches == 0 {
        return;
    }

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

fn assert_self_similarity(
    scorer: &HungarianCosine<f32, f32>,
    spectrum: &GenericSpectrum<f32, f32>,
    tolerance: f32,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let Ok((score, matches)) = scorer.similarity(spectrum, spectrum) else {
        return;
    };
    if matches == 0 {
        return;
    }

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

fn assert_linear_entropy_self_similarity(
    scorer: &LinearEntropy<f32, f32>,
    spectrum: &GenericSpectrum<f32, f32>,
    tolerance: f32,
    label: &str,
) {
    if spectrum.is_empty() {
        return;
    }

    let Ok((score, matches)) = scorer.similarity(spectrum, spectrum) else {
        return;
    };
    if matches == 0 {
        return;
    }

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

fn assert_modified_linear_matches_modified_hungarian(
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let left = merger.process(left);
    let right = merger.process(right);

    let modified_hungarian = ModifiedHungarianCosine::new(1.0_f32, 1.0_f32, FIXED_TOLERANCE)
        .expect("fixed modified-hungarian config is valid");
    let modified_linear = ModifiedLinearCosine::new(1.0_f32, 1.0_f32, FIXED_TOLERANCE)
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

    if (modified_hungarian_score - modified_linear_score).abs() > MODIFIED_DIFFERENTIAL_EPS
        || modified_hungarian_matches != modified_linear_matches
    {
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
    modified: &ModifiedHungarianCosine<f32, f32>,
    hungarian: &HungarianCosine<f32, f32>,
    left: &S1,
    right: &S2,
    label: &str,
) where
    S1: Spectrum<Mz = f32, Intensity = f32>,
    S2: Spectrum<Mz = f32, Intensity = f32>,
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

fn assert_linear_matches_hungarian(
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
) {
    let merger =
        SiriusMergeClosePeaks::new(FIXED_TOLERANCE).expect("fixed preprocess config is valid");
    let left = merger.process(left);
    let right = merger.process(right);

    let hungarian =
        HungarianCosine::new(1.0_f32, 1.0_f32, FIXED_TOLERANCE).expect("fixed config is valid");
    let linear = LinearCosine::new(1.0_f32, 1.0_f32, FIXED_TOLERANCE).expect("fixed config");

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
    scorer: &LinearEntropy<f32, f32>,
    left: &GenericSpectrum<f32, f32>,
    right: &GenericSpectrum<f32, f32>,
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
fn assert_score_in_range(score: f32, label: &str) {
    assert!(score.is_finite(), "{label}: score {score} is not finite");
    assert!(
        score >= -1.0e-6 && score <= 1.0 + 1.0e-6,
        "{label}: score {score} not in [0, 1]"
    );
}
