//! Submodule defining a single Spectrum collection trait.

use alloc::vec::Vec;

use geometric_traits::prelude::*;
use multi_ranged::{BiRange, SimpleRange};
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
        let number_of_rows =
            u32::try_from(self.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let number_of_columns =
            u32::try_from(other.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let mut matching_peaks: RangedCSR2D<u32, u32, SimpleRange<u32>> =
            SparseMatrixMut::with_sparse_shape((number_of_rows, number_of_columns));
        MatrixMut::increase_shape(&mut matching_peaks, (number_of_rows, number_of_columns))
            .map_err(|_| SimilarityComputationError::GraphConstructionFailed)?;
        let mut lowest_other_index = 0;
        for (i, mz) in self.mz().enumerate() {
            let row = to_matrix_index(i)?;
            let mz_f64 = to_f64_checked(mz, "mz")?;
            let mut new_lowest = lowest_other_index;
            for (j, other_mz) in other
                .mz_from(lowest_other_index)
                .enumerate()
                .map(|(j, mz)| (j + lowest_other_index, mz))
            {
                let other_mz_f64 = to_f64_checked(other_mz, "other_mz")?;
                match tolerance_position_f64(mz_f64, other_mz_f64, mz_tolerance_f64) {
                    TolerancePosition::Above => {
                        // other_mz is above the tolerance window.
                        break;
                    }
                    TolerancePosition::Below => {
                        // This peak is below the tolerance window. Since both
                        // spectra are sorted by m/z, no future left peak
                        // (with higher m/z) can match this right peak either.
                        new_lowest = j + 1;
                        continue;
                    }
                    TolerancePosition::Within => {}
                }
                let col = to_matrix_index(j)?;
                MatrixMut::add(&mut matching_peaks, (row, col))
                    .map_err(|_| SimilarityComputationError::GraphConstructionFailed)?;
            }
            lowest_other_index = new_lowest;
        }

        Ok(matching_peaks)
    }

    /// Returns the matching peaks graph for modified cosine similarity.
    ///
    /// Two windows are used per left peak: a direct window (same as
    /// `matching_peaks`) and a shifted window offset by `mz_shift`
    /// (`precursor_mz_self - precursor_mz_other`). This captures both
    /// direct matches and neutral-loss-related correspondences.
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

        let number_of_rows =
            u32::try_from(self.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let number_of_columns =
            u32::try_from(other.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let mut matching_peaks: RangedCSR2D<u32, u32, BiRange<u32>> =
            SparseMatrixMut::with_sparse_shape((number_of_rows, number_of_columns));
        MatrixMut::increase_shape(&mut matching_peaks, (number_of_rows, number_of_columns))
            .map_err(|_| SimilarityComputationError::GraphConstructionFailed)?;
        let mut lowest_direct = 0usize;
        let mut lowest_shifted = 0usize;

        for (i, mz) in self.mz().enumerate() {
            let row = to_matrix_index(i)?;
            let mz_f64 = to_f64_checked(mz, "mz")?;

            let mut row_matches: Vec<u32> = Vec::new();

            // --- Direct window ---
            let mut new_lowest_direct = lowest_direct;
            for (j, other_mz) in other
                .mz_from(lowest_direct)
                .enumerate()
                .map(|(j, mz)| (j + lowest_direct, mz))
            {
                let other_mz_f64 = to_f64_checked(other_mz, "other_mz")?;

                match tolerance_position_f64(mz_f64, other_mz_f64, mz_tolerance_f64) {
                    TolerancePosition::Above => break,
                    TolerancePosition::Below => {
                        new_lowest_direct = j + 1;
                        continue;
                    }
                    TolerancePosition::Within => {}
                }
                row_matches.push(to_matrix_index(j)?);
            }
            lowest_direct = new_lowest_direct;

            // --- Shifted window ---
            let mut new_lowest_shifted = lowest_shifted;
            for (j, other_mz) in other
                .mz_from(lowest_shifted)
                .enumerate()
                .map(|(j, mz)| (j + lowest_shifted, mz))
            {
                let other_mz_f64 = to_f64_checked(other_mz, "other_mz")?;

                let shifted_other_mz_f64 = other_mz_f64 + mz_shift_f64;
                if !shifted_other_mz_f64.is_finite() {
                    return Err(SimilarityComputationError::NonFiniteValue(
                        "shifted_other_mz",
                    ));
                }

                match tolerance_position_f64(mz_f64, shifted_other_mz_f64, mz_tolerance_f64) {
                    TolerancePosition::Above => break,
                    TolerancePosition::Below => {
                        new_lowest_shifted = j + 1;
                        continue;
                    }
                    TolerancePosition::Within => {}
                }
                row_matches.push(to_matrix_index(j)?);
            }
            lowest_shifted = new_lowest_shifted;

            row_matches.sort_unstable();
            row_matches.dedup();
            for col in row_matches {
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
