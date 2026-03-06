//! Reusable fuzzing harnesses for similarity implementations.
//!
//! The harness logic lives in the main crate so fuzz targets and regression
//! tests execute the exact same code paths.

use arbitrary::{Arbitrary, Unstructured};

use crate::prelude::ScalarSimilarity;
use crate::structs::{GenericSpectrum, HungarianCosine, LinearCosine, SiriusMergeClosePeaks};
use crate::traits::{SpectralProcessor, Spectrum};

const FIXED_TOLERANCE: f32 = 0.1;
const SYMMETRY_EPS: f32 = 1.0e-4;
const DIFFERENTIAL_EPS: f32 = 1.0e-4;

/// Result returned by [`run_hungarian_cosine_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HungarianCosineHarnessOutcome {
    /// All configured checks completed.
    Checked,
}

/// Execute the Hungarian-cosine fuzz harness for an arbitrary byte slice.
///
/// The function intentionally panics when a correctness invariant is violated.
/// This behavior is required for fuzzers and regression tests to surface bugs.
pub fn run_hungarian_cosine_case(bytes: &[u8]) -> HungarianCosineHarnessOutcome {
    let mut unstructured = Unstructured::new(bytes);
    let case = match HungarianCosineFuzzCase::arbitrary(&mut unstructured) {
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

#[derive(Debug)]
struct HungarianCosineFuzzCase {
    mz_power: f32,
    intensity_power: f32,
    tolerance: f32,
    left: GenericSpectrum<f32, f32>,
    right: GenericSpectrum<f32, f32>,
}

impl<'a> Arbitrary<'a> for HungarianCosineFuzzCase {
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

#[inline]
fn assert_score_in_range(score: f32, label: &str) {
    assert!(score.is_finite(), "{label}: score {score} is not finite");
    assert!(
        score >= -1.0e-6 && score <= 1.0 + 1.0e-6,
        "{label}: score {score} not in [0, 1]"
    );
}
