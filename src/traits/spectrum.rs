//! Submodule defining a single Spectrum collection trait.

use geometric_traits::prelude::*;
use multi_ranged::{BiRange, SimpleRange};
use num_traits::Zero;

use crate::prelude::Annotation;
use crate::structs::SimilarityComputationError;

#[inline]
fn to_matrix_index(index: usize) -> Result<u32, SimilarityComputationError> {
    u32::try_from(index).map_err(|_| SimilarityComputationError::IndexOverflow)
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
    ) -> Result<RangedCSR2D<u32, u32, SimpleRange<u32>>, SimilarityComputationError> {
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
            let mut new_lowest = lowest_other_index;
            for (j, other_mz) in other
                .mz_from(lowest_other_index)
                .enumerate()
                .map(|(j, mz)| (j + lowest_other_index, mz))
            {
                // Use a single canonical difference expression to avoid
                // tolerance-boundary asymmetry from mixed +/- comparisons.
                let diff = mz - other_mz;
                if diff < Self::Mz::zero() - mz_tolerance {
                    // other_mz is above the tolerance window.
                    break;
                }
                if diff > mz_tolerance {
                    // This peak is below the tolerance window. Since both
                    // spectra are sorted by m/z, no future left peak (with
                    // higher m/z) can match this right peak either, so we
                    // can safely skip past it for all subsequent iterations.
                    new_lowest = j + 1;
                    continue;
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
    ) -> Result<RangedCSR2D<u32, u32, BiRange<u32>>, SimilarityComputationError> {
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
            // The shifted window centre: we look for right peaks near
            // mz - shift, i.e. mz2 ∈ [mz - shift - tol, mz - shift + tol].
            let shifted_centre = mz - mz_shift;

            // Determine which window centre is lower so we insert column
            // indices in ascending order (required by BiRange).
            let (first_lowest, second_lowest, first_is_direct) = if mz <= shifted_centre {
                (&mut lowest_direct, &mut lowest_shifted, true)
            } else {
                (&mut lowest_shifted, &mut lowest_direct, false)
            };

            // --- First (lower-centre) window ---
            let mut new_first_lowest = *first_lowest;
            for (j, other_mz) in other
                .mz_from(*first_lowest)
                .enumerate()
                .map(|(j, mz)| (j + *first_lowest, mz))
            {
                // For shifted matches use (mz - other_mz) - shift so that
                // swapping spectra and negating shift is bit-symmetric.
                let diff = if first_is_direct {
                    mz - other_mz
                } else {
                    mz - other_mz - mz_shift
                };

                if diff < Self::Mz::zero() - mz_tolerance {
                    break;
                }
                if diff > mz_tolerance {
                    new_first_lowest = j + 1;
                    continue;
                }
                let col = to_matrix_index(j)?;
                MatrixMut::add(&mut matching_peaks, (row, col))
                    .map_err(|_| SimilarityComputationError::GraphConstructionFailed)?;
            }
            *first_lowest = new_first_lowest;

            // --- Second (higher-centre) window ---
            // Silently ignore duplicates: when |shift| < 2*tol the two
            // windows overlap and the same column index may already have
            // been inserted by the first window.
            let mut new_second_lowest = *second_lowest;
            for (j, other_mz) in other
                .mz_from(*second_lowest)
                .enumerate()
                .map(|(j, mz)| (j + *second_lowest, mz))
            {
                let diff = if first_is_direct {
                    mz - other_mz - mz_shift
                } else {
                    mz - other_mz
                };

                if diff < Self::Mz::zero() - mz_tolerance {
                    break;
                }
                if diff > mz_tolerance {
                    new_second_lowest = j + 1;
                    continue;
                }
                let col = to_matrix_index(j)?;
                match MatrixMut::add(&mut matching_peaks, (row, col)) {
                    Ok(()) | Err(geometric_traits::impls::MutabilityError::DuplicatedEntry(_)) => {}
                    Err(_) => {
                        return Err(SimilarityComputationError::GraphConstructionFailed);
                    }
                }
            }
            *second_lowest = new_second_lowest;
        }

        Ok(matching_peaks)
    }
}

/// Trait for [`Spectrum`] with annotations.
pub trait AnnotatedSpectrum: Spectrum {
    /// The type of the annotation.
    type Annotation: Annotation;
}
