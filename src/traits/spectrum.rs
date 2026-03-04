//! Submodule defining a single Spectrum collection trait.

use alloc::vec::Vec;

use geometric_traits::prelude::*;
use multi_ranged::{BiRange, MultiRanged, SimpleRange};
use num_traits::ToPrimitive;

use crate::numeric_validation::{NumericValidationError, checked_to_f64};
use crate::prelude::Annotation;
use crate::structs::SimilarityComputationError;

#[inline]
fn to_matrix_index(index: usize) -> Result<u32, SimilarityComputationError> {
    u32::try_from(index).map_err(|_| SimilarityComputationError::IndexOverflow)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TolerancePosition {
    Below,
    Within,
    Above,
}

#[inline]
fn to_f64_checked<T: ToPrimitive>(
    value: T,
    name: &'static str,
) -> Result<f64, SimilarityComputationError> {
    checked_to_f64(value, name).map_err(|error| match error {
        NumericValidationError::NonRepresentable(name) => {
            SimilarityComputationError::ValueNotRepresentable(name)
        }
        NumericValidationError::NonFinite(name) => SimilarityComputationError::NonFiniteValue(name),
    })
}

#[inline]
fn tolerance_position_f64(target: f64, candidate: f64, tolerance: f64) -> TolerancePosition {
    if candidate > target {
        if candidate - target > tolerance {
            TolerancePosition::Above
        } else {
            TolerancePosition::Within
        }
    } else if target > candidate {
        if target - candidate > tolerance {
            TolerancePosition::Below
        } else {
            TolerancePosition::Within
        }
    } else {
        TolerancePosition::Within
    }
}

fn allocate_matching_peaks<R>(
    number_of_rows: usize,
    number_of_columns: usize,
) -> Result<RangedCSR2D<u32, u32, R>, SimilarityComputationError>
where
    R: MultiRanged<Step = u32>,
{
    let number_of_rows = to_matrix_index(number_of_rows)?;
    let number_of_columns = to_matrix_index(number_of_columns)?;
    let mut matching_peaks: RangedCSR2D<u32, u32, R> =
        SparseMatrixMut::with_sparse_shape((number_of_rows, number_of_columns));
    MatrixMut::increase_shape(&mut matching_peaks, (number_of_rows, number_of_columns))
        .map_err(|_| SimilarityComputationError::GraphConstructionFailed)?;
    Ok(matching_peaks)
}

fn collect_window_matches<S, F>(
    other: &S,
    lowest_other_index: usize,
    mz_f64: f64,
    mz_tolerance_f64: f64,
    mz_shift_f64: f64,
    non_finite_shift_label: &'static str,
    mut on_match: F,
) -> Result<usize, SimilarityComputationError>
where
    S: Spectrum,
    S::Mz: ToPrimitive,
    F: FnMut(u32),
{
    let mut new_lowest = lowest_other_index;
    for (j, other_mz) in other
        .mz_from(lowest_other_index)
        .enumerate()
        .map(|(j, mz)| (j + lowest_other_index, mz))
    {
        let other_mz_f64 = to_f64_checked(other_mz, "right_mz")?;
        let shifted_other_mz_f64 = other_mz_f64 + mz_shift_f64;
        if !shifted_other_mz_f64.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue(
                non_finite_shift_label,
            ));
        }

        match tolerance_position_f64(mz_f64, shifted_other_mz_f64, mz_tolerance_f64) {
            TolerancePosition::Above => break,
            TolerancePosition::Below => {
                new_lowest = j + 1;
                continue;
            }
            TolerancePosition::Within => {}
        }
        on_match(to_matrix_index(j)?);
    }
    Ok(new_lowest)
}

/// Trait for a single Spectrum.
pub trait Spectrum {
    /// The type of the Intensity.
    type Intensity: Number;
    /// The type of the mass over charge.
    type Mz: Number;
    /// Iterator over the intensities in the Spectrum, sorted by mass over
    /// charge.
    type SortedIntensitiesIter<'a>: Iterator<Item = Self::Intensity>
    where
        Self: 'a;
    /// Iterator over the sorted mass over charge values in the Spectrum.
    type SortedMzIter<'a>: Iterator<Item = Self::Mz>
    where
        Self: 'a;
    /// Iterator over the peaks in the Spectrum, sorted by mass over charge
    type SortedPeaksIter<'a>: Iterator<Item = (Self::Mz, Self::Intensity)>
    where
        Self: 'a;

    /// Returns the number of peaks in the Spectrum.
    fn len(&self) -> usize;

    /// Returns whether the Spectrum is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the intensities in the Spectrum, SORTED by mass
    /// over charge.
    fn intensities(&self) -> Self::SortedIntensitiesIter<'_>;

    /// Returns the nth-intensity value.
    ///
    /// # Arguments
    ///
    /// * `n`: The index of the intensity value.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    fn intensity_nth(&self, n: usize) -> Self::Intensity;

    /// Returns an iterator over the SORTED mass over charge values in the
    /// Spectrum.
    fn mz(&self) -> Self::SortedMzIter<'_>;

    /// Returns an iterator over the SORTED mass over charge values in the
    /// Spectrum, starting from the requested index.
    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_>;

    /// Returns the nth-mz value.
    ///
    /// # Arguments
    ///
    /// * `n`: The index of the mz value.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    fn mz_nth(&self, n: usize) -> Self::Mz;

    /// Returns an iterator over the peaks in the Spectrum, SORTED by mass over
    /// charge.
    fn peaks(&self) -> Self::SortedPeaksIter<'_>;

    /// Returns the nth-peak.
    ///
    /// # Arguments
    ///
    /// * `n`: The index of the peak.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    fn peak_nth(&self, n: usize) -> (Self::Mz, Self::Intensity);

    /// Returns the precursor mass over charge.
    fn precursor_mz(&self) -> Self::Mz;

    /// Returns the matching peaks graph for the provided tolerance.
    ///
    /// # Arguments
    ///
    /// * `other`: The other Spectrum.
    /// * `mz_tolerance`: The mass over charge
    fn matching_peaks<S: Spectrum<Mz = Self::Mz>>(
        &self,
        other: &S,
        mz_tolerance: Self::Mz,
    ) -> Result<RangedCSR2D<u32, u32, SimpleRange<u32>>, SimilarityComputationError>
    where
        Self::Mz: ToPrimitive,
    {
        let mz_tolerance_f64 = to_f64_checked(mz_tolerance, "mz_tolerance")?;
        if mz_tolerance_f64 < 0.0 {
            return Err(SimilarityComputationError::NegativeTolerance);
        }
        let mut matching_peaks =
            allocate_matching_peaks::<SimpleRange<u32>>(self.len(), other.len())?;
        let mut lowest_other_index = 0usize;
        let mut row_matches: Vec<u32> = Vec::new();
        for (i, mz) in self.mz().enumerate() {
            let row = to_matrix_index(i)?;
            let mz_f64 = to_f64_checked(mz, "left_mz")?;
            row_matches.clear();
            lowest_other_index = collect_window_matches(
                other,
                lowest_other_index,
                mz_f64,
                mz_tolerance_f64,
                0.0,
                "shifted_other_mz",
                |col| row_matches.push(col),
            )?;
            for &col in &row_matches {
                MatrixMut::add(&mut matching_peaks, (row, col))
                    .map_err(|_| SimilarityComputationError::GraphConstructionFailed)?;
            }
        }

        Ok(matching_peaks)
    }

    /// Returns the matching peaks graph for modified cosine similarity.
    ///
    /// Two windows are used per left peak: a direct window (same as
    /// `matching_peaks`) and, when `|mz_shift| > mz_tolerance`, a shifted
    /// window offset by `mz_shift` (`precursor_mz_self - precursor_mz_other`).
    /// This captures both direct matches and neutral-loss-related
    /// correspondences while remaining identical to non-modified matching for
    /// precursor differences within tolerance.
    ///
    /// # Arguments
    ///
    /// * `other`: The other Spectrum.
    /// * `mz_tolerance`: The mass over charge tolerance.
    /// * `mz_shift`: The precursor mass difference (`self - other`).
    fn modified_matching_peaks<S: Spectrum<Mz = Self::Mz>>(
        &self,
        other: &S,
        mz_tolerance: Self::Mz,
        mz_shift: Self::Mz,
    ) -> Result<RangedCSR2D<u32, u32, BiRange<u32>>, SimilarityComputationError>
    where
        Self::Mz: ToPrimitive,
    {
        let mz_tolerance_f64 = to_f64_checked(mz_tolerance, "mz_tolerance")?;
        if mz_tolerance_f64 < 0.0 {
            return Err(SimilarityComputationError::NegativeTolerance);
        }
        let mz_shift_f64 = to_f64_checked(mz_shift, "mz_shift")?;
        let use_shifted_window = mz_shift_f64.abs() > mz_tolerance_f64;
        let mut matching_peaks = allocate_matching_peaks::<BiRange<u32>>(self.len(), other.len())?;
        let mut lowest_direct = 0usize;
        let mut lowest_shifted = 0usize;
        let mut row_matches: Vec<u32> = Vec::new();

        for (i, mz) in self.mz().enumerate() {
            let row = to_matrix_index(i)?;
            let mz_f64 = to_f64_checked(mz, "left_mz")?;

            row_matches.clear();

            lowest_direct = collect_window_matches(
                other,
                lowest_direct,
                mz_f64,
                mz_tolerance_f64,
                0.0,
                "shifted_other_mz",
                |col| row_matches.push(col),
            )?;

            if use_shifted_window {
                lowest_shifted = collect_window_matches(
                    other,
                    lowest_shifted,
                    mz_f64,
                    mz_tolerance_f64,
                    mz_shift_f64,
                    "shifted_other_mz",
                    |col| row_matches.push(col),
                )?;

                row_matches.sort_unstable();
                row_matches.dedup();
            }
            for &col in &row_matches {
                MatrixMut::add(&mut matching_peaks, (row, col))
                    .map_err(|_| SimilarityComputationError::GraphConstructionFailed)?;
            }
        }

        Ok(matching_peaks)
    }
}

/// Trait for [`Spectrum`] with annotations.
pub trait AnnotatedSpectrum: Spectrum {
    /// The type of the annotation.
    type Annotation: Annotation;
}
