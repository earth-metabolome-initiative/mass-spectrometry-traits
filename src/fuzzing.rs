//! Reusable fuzzing harnesses for similarity implementations.
//!
//! The harness logic lives in the main crate so fuzz targets and regression
//! tests execute the exact same code paths.

use alloc::vec::Vec;

use crate::prelude::ScalarSimilarity;
use crate::structs::{GenericSpectrum, HungarianCosine, LinearCosine};
use crate::traits::{
    AspirinSpectrum, CocaineSpectrum, GlucoseSpectrum, Spectrum, SpectrumAlloc, SpectrumMut,
    StypolTrioneSpectrum,
};

const MAX_DECODED_PEAKS: usize = 48;
const MAX_SANITIZED_PEAKS: usize = 256;
const MIN_PEAK_VALUE: f32 = 1.0e-6;
const MAX_PEAK_VALUE: f32 = 1_000_000.0;
const MIN_PRECURSOR_MZ: f32 = 1.0e-3;
const MAX_PRECURSOR_MZ: f32 = 1_000_000.0;
const MIN_MZ_GAP: f32 = 1.0e-4;
const FIXED_TOLERANCE: f32 = 0.1;
const STRICT_MIN_GAP: f32 = (2.0 * FIXED_TOLERANCE) + 1.0e-4;
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
    let case = HungarianCosineFuzzCase::decode(bytes);
    let left_payload = case.left_payload();
    let right_payload = case.right_payload();

    let raw_left = RawSpectrum::from_payload(&left_payload);
    let raw_right = RawSpectrum::from_payload(&right_payload);
    let sanitized_left = sanitize_payload(&left_payload);
    let sanitized_right = sanitize_payload(&right_payload);

    if let Ok(dynamic) = HungarianCosine::new(case.mz_power, case.intensity_power, case.tolerance) {
        assert_bidirectional_properties(&dynamic, &raw_left, &raw_right, "dynamic/raw");
        assert_bidirectional_properties(
            &dynamic,
            &sanitized_left,
            &sanitized_right,
            "dynamic/sanitized",
        );
    }

    let fixed = HungarianCosine::new(1.0, 1.0, FIXED_TOLERANCE).expect("fixed config is valid");
    assert_bidirectional_properties(&fixed, &sanitized_left, &sanitized_right, "fixed/sanitized");
    assert_self_similarity(&fixed, &sanitized_left, 1.0e-5, "fixed/left");
    assert_self_similarity(&fixed, &sanitized_right, 1.0e-5, "fixed/right");

    let wide = HungarianCosine::new(0.0, 1.0, 2.0).expect("wide config is valid");
    assert_self_similarity(&wide, &sanitized_left, 1.0e-4, "wide/left");
    assert_self_similarity(&wide, &sanitized_right, 1.0e-4, "wide/right");

    if sanitized_left.len().max(sanitized_right.len()) <= 128 {
        assert_linear_matches_hungarian(&sanitized_left, &sanitized_right);
    }

    HungarianCosineHarnessOutcome::Checked
}

#[derive(Clone, Debug)]
struct HungarianCosineFuzzCase {
    flags: u8,
    mz_power: f32,
    intensity_power: f32,
    tolerance: f32,
    left_precursor_mz: f32,
    right_precursor_mz: f32,
    left_reference_selector: u8,
    right_reference_selector: u8,
    left_peaks: Vec<(f32, f32)>,
    right_peaks: Vec<(f32, f32)>,
}

impl HungarianCosineFuzzCase {
    fn decode(bytes: &[u8]) -> Self {
        let mut cursor = ByteCursor::new(bytes);
        let flags = cursor.next_u8();
        let mz_power = cursor.next_f32();
        let intensity_power = cursor.next_f32();
        let tolerance = cursor.next_f32();
        let left_precursor_mz = cursor.next_f32();
        let right_precursor_mz = cursor.next_f32();
        let left_reference_selector = cursor.next_u8();
        let right_reference_selector = cursor.next_u8();
        let left_len = (cursor.next_u8() as usize) % (MAX_DECODED_PEAKS + 1);
        let right_len = (cursor.next_u8() as usize) % (MAX_DECODED_PEAKS + 1);

        let mut left_peaks = Vec::with_capacity(left_len);
        for _ in 0..left_len {
            left_peaks.push((cursor.next_f32(), cursor.next_f32()));
        }

        let mut right_peaks = Vec::with_capacity(right_len);
        for _ in 0..right_len {
            right_peaks.push((cursor.next_f32(), cursor.next_f32()));
        }

        Self {
            flags,
            mz_power,
            intensity_power,
            tolerance,
            left_precursor_mz,
            right_precursor_mz,
            left_reference_selector,
            right_reference_selector,
            left_peaks,
            right_peaks,
        }
    }

    fn left_payload(&self) -> SpectrumPayload {
        let use_reference = self.flags & 0b0000_0001 != 0;
        if use_reference && let Some(payload) = reference_payload(self.left_reference_selector) {
            return payload;
        }
        SpectrumPayload {
            precursor_mz: self.left_precursor_mz,
            peaks: self.left_peaks.clone(),
        }
    }

    fn right_payload(&self) -> SpectrumPayload {
        let use_reference = self.flags & 0b0000_0010 != 0;
        if use_reference && let Some(payload) = reference_payload(self.right_reference_selector) {
            return payload;
        }
        SpectrumPayload {
            precursor_mz: self.right_precursor_mz,
            peaks: self.right_peaks.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct SpectrumPayload {
    precursor_mz: f32,
    peaks: Vec<(f32, f32)>,
}

fn reference_payload(selector: u8) -> Option<SpectrumPayload> {
    let spectrum = match selector {
        0 => <GenericSpectrum<f32, f32> as CocaineSpectrum>::cocaine().ok()?,
        1 => <GenericSpectrum<f32, f32> as GlucoseSpectrum>::glucose().ok()?,
        2 => <GenericSpectrum<f32, f32> as AspirinSpectrum>::aspirin().ok()?,
        255 => <GenericSpectrum<f32, f32> as StypolTrioneSpectrum>::stypoltrione().ok()?,
        _ => return None,
    };

    let peaks = spectrum.peaks().collect();
    Some(SpectrumPayload {
        precursor_mz: spectrum.precursor_mz(),
        peaks,
    })
}

fn sanitize_payload(payload: &SpectrumPayload) -> GenericSpectrum<f32, f32> {
    let precursor_mz = sanitize_precursor(payload.precursor_mz);
    let mut peaks: Vec<(f32, f32)> = payload
        .peaks
        .iter()
        .copied()
        .filter_map(|(mz, intensity)| sanitize_peak(mz, intensity))
        .collect();

    if peaks.len() > MAX_SANITIZED_PEAKS {
        peaks.truncate(MAX_SANITIZED_PEAKS);
    }
    peaks.sort_by(|(left_mz, _), (right_mz, _)| left_mz.total_cmp(right_mz));

    let mut spectrum =
        GenericSpectrum::with_capacity(precursor_mz, peaks.len().max(1)).expect("valid precursor");
    let mut last_mz: Option<f32> = None;

    for (mz, intensity) in peaks {
        let mut mz = mz;
        if let Some(previous_mz) = last_mz
            && mz <= previous_mz
        {
            let candidate = (previous_mz + MIN_MZ_GAP).min(MAX_PEAK_VALUE);
            if candidate <= previous_mz {
                continue;
            }
            mz = candidate;
        }

        spectrum
            .add_peak(mz, intensity)
            .expect("sanitized peak sequence must be valid");
        last_mz = Some(mz);
    }

    if spectrum.is_empty() {
        spectrum
            .add_peak(100.0, 1.0)
            .expect("fallback peak must be valid");
    }

    spectrum
}

fn sanitize_precursor(value: f32) -> f32 {
    if value.is_finite() {
        value.abs().clamp(MIN_PRECURSOR_MZ, MAX_PRECURSOR_MZ)
    } else {
        100.0
    }
}

fn sanitize_peak(mz: f32, intensity: f32) -> Option<(f32, f32)> {
    if !mz.is_finite() || !intensity.is_finite() {
        return None;
    }
    let sanitized_mz = mz.abs().clamp(MIN_PEAK_VALUE, MAX_PEAK_VALUE);
    let sanitized_intensity = intensity.abs().clamp(MIN_PEAK_VALUE, MAX_PEAK_VALUE);
    Some((sanitized_mz, sanitized_intensity))
}

fn assert_bidirectional_properties<S1, S2>(
    scorer: &HungarianCosine<f32, f32>,
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

fn assert_self_similarity(
    scorer: &HungarianCosine<f32, f32>,
    spectrum: &GenericSpectrum<f32, f32>,
    tolerance: f32,
    label: &str,
) {
    let (score, matches) = scorer
        .similarity(spectrum, spectrum)
        .expect("self-similarity should succeed for sanitized spectrum");

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
    let left = enforce_strict_spacing(left, STRICT_MIN_GAP);
    let right = enforce_strict_spacing(right, STRICT_MIN_GAP);

    let hungarian =
        HungarianCosine::new(1.0_f32, 1.0_f32, FIXED_TOLERANCE).expect("fixed config is valid");
    let linear = LinearCosine::new(1.0_f32, 1.0_f32, FIXED_TOLERANCE).expect("fixed config");

    let (hungarian_score, hungarian_matches) = hungarian
        .similarity(&left, &right)
        .expect("Hungarian similarity should succeed");
    let (linear_score, linear_matches) = linear
        .similarity(&left, &right)
        .expect("Linear similarity should succeed");

    assert!(
        (hungarian_score - linear_score).abs() <= DIFFERENTIAL_EPS,
        "fixed differential mismatch: Hungarian={hungarian_score} vs Linear={linear_score}"
    );
    assert_eq!(
        hungarian_matches, linear_matches,
        "fixed differential match mismatch: Hungarian={hungarian_matches} vs Linear={linear_matches}"
    );
}

fn enforce_strict_spacing(
    spectrum: &GenericSpectrum<f32, f32>,
    min_gap: f32,
) -> GenericSpectrum<f32, f32> {
    let mut output = GenericSpectrum::with_capacity(spectrum.precursor_mz(), spectrum.len())
        .expect("valid precursor");
    let mut last_mz: Option<f32> = None;

    for (mz, intensity) in spectrum.peaks() {
        let mut adjusted_mz = mz;
        if let Some(previous_mz) = last_mz {
            let required = previous_mz + min_gap;
            if adjusted_mz < required {
                adjusted_mz = required;
            }
        }

        output
            .add_peak(adjusted_mz, intensity.max(MIN_PEAK_VALUE))
            .expect("strict spacing transformation must stay valid");
        last_mz = Some(adjusted_mz);
    }

    output
}

#[inline]
fn assert_score_in_range(score: f32, label: &str) {
    assert!(score.is_finite(), "{label}: score {score} is not finite");
    assert!(
        score >= -1.0e-6 && score <= 1.0 + 1.0e-6,
        "{label}: score {score} not in [0, 1]"
    );
}

fn map_peak_mz(peak: &(f32, f32)) -> f32 {
    peak.0
}

fn map_peak_intensity(peak: &(f32, f32)) -> f32 {
    peak.1
}

#[derive(Clone, Debug)]
struct RawSpectrum {
    precursor_mz: f32,
    peaks: Vec<(f32, f32)>,
}

impl RawSpectrum {
    fn from_payload(payload: &SpectrumPayload) -> Self {
        Self {
            precursor_mz: payload.precursor_mz,
            peaks: payload.peaks.clone(),
        }
    }
}

impl Spectrum for RawSpectrum {
    type Intensity = f32;
    type Mz = f32;
    type SortedIntensitiesIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f32, f32)>, fn(&(f32, f32)) -> f32>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = core::iter::Map<core::slice::Iter<'a, (f32, f32)>, fn(&(f32, f32)) -> f32>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = core::iter::Copied<core::slice::Iter<'a, (f32, f32)>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.peaks.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.peaks.iter().map(map_peak_intensity)
    }

    fn intensity_nth(&self, n: usize) -> Self::Intensity {
        self.peaks[n].1
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.peaks.iter().map(map_peak_mz)
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.peaks[index..].iter().map(map_peak_mz)
    }

    fn mz_nth(&self, n: usize) -> Self::Mz {
        self.peaks[n].0
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.peaks.iter().copied()
    }

    fn peak_nth(&self, n: usize) -> (Self::Mz, Self::Intensity) {
        self.peaks[n]
    }

    fn precursor_mz(&self) -> Self::Mz {
        self.precursor_mz
    }
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn next_u8(&mut self) -> u8 {
        let value = self.bytes.get(self.offset).copied().unwrap_or(0);
        self.offset = self.offset.saturating_add(1);
        value
    }

    fn next_f32(&mut self) -> f32 {
        let bytes = [
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
        ];
        f32::from_le_bytes(bytes)
    }
}
