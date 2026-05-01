//! Submodule defining a single Spectrum collection trait.

use alloc::vec::Vec;

use geometric_traits::prelude::*;
use multi_ranged::{BiRange, MultiRanged, SimpleRange};

use super::spectrum_annotation::Annotation;

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
fn tolerance_position_f64(target: f64, candidate: f64, tolerance: f64) -> TolerancePosition {
    if candidate > target + tolerance {
        TolerancePosition::Above
    } else if candidate < target - tolerance {
        TolerancePosition::Below
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
    F: FnMut(u32),
{
    let mut new_lowest = lowest_other_index;
    for (j, other_mz) in other
        .mz_from(lowest_other_index)
        .enumerate()
        .map(|(j, mz)| (j + lowest_other_index, mz))
    {
        let shifted_other_mz = other_mz + mz_shift_f64;
        if !shifted_other_mz.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue(
                non_finite_shift_label,
            ));
        }

        match tolerance_position_f64(mz_f64, shifted_other_mz, mz_tolerance_f64) {
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

#[inline]
fn validate_number_of_bins(number_of_bins: usize) -> Result<(), SimilarityComputationError> {
    if number_of_bins == 0 {
        return Err(SimilarityComputationError::InvalidParameter(
            "number_of_bins",
        ));
    }
    Ok(())
}

#[inline]
fn validate_binning_range(
    mz_min: f64,
    mz_max: f64,
    require_positive_min: bool,
) -> Result<(), SimilarityComputationError> {
    if !mz_min.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue("mz_min"));
    }
    if !mz_max.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue("mz_max"));
    }
    if require_positive_min && mz_min <= 0.0 {
        return Err(SimilarityComputationError::InvalidParameter("mz_min"));
    }
    if mz_min >= mz_max {
        return Err(SimilarityComputationError::InvalidParameter("mz_range"));
    }
    Ok(())
}

#[inline]
fn bin_position_to_index(position: f64, number_of_bins: usize) -> usize {
    let nearest_boundary = position.round();
    let tolerance = f64::EPSILON * position.abs().max(1.0) * 8.0;
    let position = if (position - nearest_boundary).abs() <= tolerance {
        nearest_boundary
    } else {
        position
    };
    (position.floor() as usize).min(number_of_bins - 1)
}

fn collect_binned_intensities<S, F>(
    spectrum: &S,
    number_of_bins: usize,
    mut bin_index: F,
) -> Result<Vec<f64>, SimilarityComputationError>
where
    S: Spectrum + ?Sized,
    F: FnMut(f64) -> Option<usize>,
{
    let mut binned = alloc::vec![0.0_f64; number_of_bins];
    for (mz, intensity) in spectrum.peaks() {
        if !mz.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue("mz"));
        }
        if !intensity.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue("intensity"));
        }

        if let Some(index) = bin_index(mz) {
            binned[index] += intensity;
            if !binned[index].is_finite() {
                return Err(SimilarityComputationError::NonFiniteValue(
                    "binned_intensity",
                ));
            }
        }
    }
    Ok(binned)
}

/// Trait for a single Spectrum.
pub trait Spectrum {
    /// Iterator over the intensities in the Spectrum, sorted by mass over
    /// charge.
    type SortedIntensitiesIter<'a>: Iterator<Item = f64>
    where
        Self: 'a;
    /// Iterator over the sorted mass over charge values in the Spectrum.
    type SortedMzIter<'a>: Iterator<Item = f64>
    where
        Self: 'a;
    /// Iterator over the peaks in the Spectrum, sorted by mass over charge
    type SortedPeaksIter<'a>: Iterator<Item = (f64, f64)>
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
    fn intensity_nth(&self, n: usize) -> f64;

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
    fn mz_nth(&self, n: usize) -> f64;

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
    fn peak_nth(&self, n: usize) -> (f64, f64);

    /// Returns the precursor mass over charge.
    fn precursor_mz(&self) -> f64;

    /// Returns a dense vector of intensities binned linearly over m/z.
    ///
    /// The interval is `[mz_min, mz_max]`, with `number_of_bins` equal-width
    /// bins. Values below `mz_min` or above `mz_max` are ignored. A peak exactly
    /// at `mz_max` is assigned to the last bin.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityComputationError`] when the range parameters are
    /// invalid, `number_of_bins` is zero, or a peak m/z/intensity is non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// use mass_spectrometry::prelude::*;
    ///
    /// let mut spectrum = GenericSpectrum::try_with_capacity(100.0, 3).unwrap();
    /// spectrum.add_peak(1.0, 2.0).unwrap();
    /// spectrum.add_peak(1.49, 3.0).unwrap();
    /// spectrum.add_peak(2.0, 7.0).unwrap();
    ///
    /// let bins = spectrum.linear_binned_intensities(1.0, 2.0, 2).unwrap();
    /// assert_eq!(bins, vec![5.0, 7.0]);
    /// ```
    fn linear_binned_intensities(
        &self,
        mz_min: f64,
        mz_max: f64,
        number_of_bins: usize,
    ) -> Result<Vec<f64>, SimilarityComputationError> {
        validate_number_of_bins(number_of_bins)?;
        validate_binning_range(mz_min, mz_max, false)?;

        let number_of_bins_f64 = number_of_bins as f64;
        let scale = number_of_bins_f64 / (mz_max - mz_min);
        collect_binned_intensities(self, number_of_bins, |mz| {
            if mz < mz_min || mz > mz_max {
                return None;
            }
            if mz == mz_max {
                return Some(number_of_bins - 1);
            }
            Some(bin_position_to_index((mz - mz_min) * scale, number_of_bins))
        })
    }

    /// Returns a dense vector of intensities binned logarithmically over m/z.
    ///
    /// The interval is `[mz_min, mz_max]`, split into equal-width bins in
    /// natural-log m/z space. Values below `mz_min` or above `mz_max` are
    /// ignored. A peak exactly at `mz_max` is assigned to the last bin.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityComputationError`] when the range parameters are
    /// invalid, `mz_min <= 0`, `number_of_bins` is zero, or a peak
    /// m/z/intensity is non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// use mass_spectrometry::prelude::*;
    ///
    /// let mut spectrum = GenericSpectrum::try_with_capacity(100.0, 3).unwrap();
    /// spectrum.add_peak(1.0, 2.0).unwrap();
    /// spectrum.add_peak(9.0, 3.0).unwrap();
    /// spectrum.add_peak(100.0, 7.0).unwrap();
    ///
    /// let bins = spectrum
    ///     .logarithmic_binned_intensities(1.0, 100.0, 2)
    ///     .unwrap();
    /// assert_eq!(bins, vec![5.0, 7.0]);
    /// ```
    fn logarithmic_binned_intensities(
        &self,
        mz_min: f64,
        mz_max: f64,
        number_of_bins: usize,
    ) -> Result<Vec<f64>, SimilarityComputationError> {
        validate_number_of_bins(number_of_bins)?;
        validate_binning_range(mz_min, mz_max, true)?;

        let log_min = mz_min.ln();
        let log_max = mz_max.ln();
        let scale = number_of_bins as f64 / (log_max - log_min);
        collect_binned_intensities(self, number_of_bins, |mz| {
            if mz < mz_min || mz > mz_max {
                return None;
            }
            if mz == mz_max {
                return Some(number_of_bins - 1);
            }
            Some(bin_position_to_index(
                (mz.ln() - log_min) * scale,
                number_of_bins,
            ))
        })
    }

    /// Returns the matching peaks graph for the provided tolerance.
    ///
    /// # Arguments
    ///
    /// * `other`: The other Spectrum.
    /// * `mz_tolerance`: The mass over charge
    fn matching_peaks<S: Spectrum>(
        &self,
        other: &S,
        mz_tolerance: f64,
    ) -> Result<RangedCSR2D<u32, u32, SimpleRange<u32>>, SimilarityComputationError> {
        if !mz_tolerance.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue("mz_tolerance"));
        }
        if mz_tolerance < 0.0 {
            return Err(SimilarityComputationError::NegativeTolerance);
        }
        let mut matching_peaks =
            allocate_matching_peaks::<SimpleRange<u32>>(self.len(), other.len())?;
        let mut lowest_other_index = 0usize;
        let mut row_matches: Vec<u32> = Vec::new();
        for (i, mz) in self.mz().enumerate() {
            if !mz.is_finite() {
                return Err(SimilarityComputationError::NonFiniteValue("left_mz"));
            }
            let row = to_matrix_index(i)?;
            row_matches.clear();
            lowest_other_index = collect_window_matches(
                other,
                lowest_other_index,
                mz,
                mz_tolerance,
                0.0,
                "right_mz",
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
    /// `matching_peaks`) and, when the precursor difference exceeds
    /// `mz_tolerance`, a shifted window that matches neutral losses
    /// (`self_mz - self_precursor` vs `other_mz - other_precursor`).
    /// This captures both direct matches and neutral-loss-related
    /// correspondences while remaining identical to non-modified matching for
    /// precursor differences within tolerance.
    ///
    /// # Arguments
    ///
    /// * `other`: The other Spectrum.
    /// * `mz_tolerance`: The mass over charge tolerance.
    /// * `self_precursor`: The precursor m/z of `self`.
    /// * `other_precursor`: The precursor m/z of `other`.
    fn modified_matching_peaks<S: Spectrum>(
        &self,
        other: &S,
        mz_tolerance: f64,
        self_precursor: f64,
        other_precursor: f64,
    ) -> Result<RangedCSR2D<u32, u32, BiRange<u32>>, SimilarityComputationError> {
        if !mz_tolerance.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue("mz_tolerance"));
        }
        if mz_tolerance < 0.0 {
            return Err(SimilarityComputationError::NegativeTolerance);
        }
        let use_shifted_window = other_precursor < self_precursor - mz_tolerance
            || other_precursor > self_precursor + mz_tolerance;
        let mut matching_peaks = allocate_matching_peaks::<BiRange<u32>>(self.len(), other.len())?;
        let mut lowest_direct = 0usize;
        let mut lowest_shifted = 0usize;
        let mut row_matches: Vec<u32> = Vec::new();

        for (i, mz) in self.mz().enumerate() {
            if !mz.is_finite() {
                return Err(SimilarityComputationError::NonFiniteValue("left_mz"));
            }
            let row = to_matrix_index(i)?;

            row_matches.clear();

            lowest_direct = collect_window_matches(
                other,
                lowest_direct,
                mz,
                mz_tolerance,
                0.0,
                "right_mz",
                |col| row_matches.push(col),
            )?;

            if use_shifted_window {
                // Compare neutral losses directly to avoid floating-point
                // absorption when the shift magnitude dwarfs peak mz values.
                // self_nl = self_mz - self_prec, other_nl = other_mz - other_prec.
                // Passing -other_prec as the shift makes collect_window_matches
                // compute other_mz + (-other_prec) = other_nl for each right peak.
                lowest_shifted = collect_window_matches(
                    other,
                    lowest_shifted,
                    mz - self_precursor,
                    mz_tolerance,
                    -other_precursor,
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

#[cfg(test)]
mod tests {
    use geometric_traits::prelude::*;
    use multi_ranged::SimpleRange;

    use super::*;

    #[derive(Clone)]
    struct RawSpectrum {
        precursor_mz: f64,
        peaks: Vec<(f64, f64)>,
    }

    impl Spectrum for RawSpectrum {
        type SortedIntensitiesIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedMzIter<'a>
            = core::iter::Map<core::slice::Iter<'a, (f64, f64)>, fn(&(f64, f64)) -> f64>
        where
            Self: 'a;
        type SortedPeaksIter<'a>
            = core::iter::Copied<core::slice::Iter<'a, (f64, f64)>>
        where
            Self: 'a;

        fn len(&self) -> usize {
            self.peaks.len()
        }

        fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
            self.peaks.iter().map(|peak| peak.1)
        }

        fn intensity_nth(&self, n: usize) -> f64 {
            self.peaks[n].1
        }

        fn mz(&self) -> Self::SortedMzIter<'_> {
            self.peaks.iter().map(|peak| peak.0)
        }

        fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
            self.peaks[index..].iter().map(|peak| peak.0)
        }

        fn mz_nth(&self, n: usize) -> f64 {
            self.peaks[n].0
        }

        fn peaks(&self) -> Self::SortedPeaksIter<'_> {
            self.peaks.iter().copied()
        }

        fn peak_nth(&self, n: usize) -> (f64, f64) {
            self.peaks[n]
        }

        fn precursor_mz(&self) -> f64 {
            self.precursor_mz
        }
    }

    #[test]
    fn tolerance_position_f64_classifies_offsets() {
        assert_eq!(
            tolerance_position_f64(10.0, 9.7, 0.1),
            TolerancePosition::Below
        );
        assert_eq!(
            tolerance_position_f64(10.0, 10.0, 0.1),
            TolerancePosition::Within
        );
        assert_eq!(
            tolerance_position_f64(10.0, 10.3, 0.1),
            TolerancePosition::Above
        );
    }

    #[test]
    fn allocate_matching_peaks_preserves_empty_shape() {
        let left_empty: RangedCSR2D<u32, u32, SimpleRange<u32>> =
            allocate_matching_peaks::<SimpleRange<u32>>(0, 2).expect("shape should allocate");
        assert_eq!(left_empty.number_of_rows(), 0);
        assert_eq!(left_empty.number_of_columns(), 2);
        assert_eq!(left_empty.number_of_defined_values(), 0);

        let right_empty: RangedCSR2D<u32, u32, SimpleRange<u32>> =
            allocate_matching_peaks::<SimpleRange<u32>>(2, 0).expect("shape should allocate");
        assert_eq!(right_empty.number_of_rows(), 2);
        assert_eq!(right_empty.number_of_columns(), 0);
        assert_eq!(right_empty.number_of_defined_values(), 0);
    }

    #[test]
    fn collect_window_matches_advances_lowest_index_and_reports_non_finite_shift() {
        let other = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(0.8, 1.0), (1.0, 1.0), (1.2, 1.0)],
        };
        let mut matched_cols = Vec::new();
        let new_lowest = collect_window_matches(&other, 0, 1.0, 0.05, 0.0, "right_mz", |col| {
            matched_cols.push(col)
        })
        .expect("window collection should succeed");
        assert_eq!(new_lowest, 1);
        assert_eq!(matched_cols, alloc::vec![1]);

        let overflowing = RawSpectrum {
            precursor_mz: -f64::MAX,
            peaks: alloc::vec![(f64::MAX, 1.0)],
        };
        let error = collect_window_matches(
            &overflowing,
            0,
            100.0,
            0.1,
            f64::MAX,
            "shifted_other_mz",
            |_| {},
        )
        .expect_err("non-finite shifted value should be rejected");
        assert_eq!(
            error,
            SimilarityComputationError::NonFiniteValue("shifted_other_mz")
        );
    }

    #[test]
    fn spectrum_is_empty_reflects_len() {
        let empty = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![],
        };
        let nonempty = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(1.0, 1.0)],
        };

        assert!(empty.is_empty());
        assert!(!nonempty.is_empty());
    }

    #[test]
    fn matching_peaks_collects_multiple_columns_in_single_row() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(1.0, 1.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(0.95, 1.0), (1.05, 1.0), (1.3, 1.0)],
        };

        let graph = left
            .matching_peaks(&right, 0.1)
            .expect("matching graph should build");
        let cols: Vec<u32> = graph.sparse_row(0).collect();

        assert_eq!(cols, alloc::vec![0, 1]);
        assert_eq!(graph.number_of_defined_values(), 2);
    }

    #[test]
    fn linear_binned_intensities_accumulates_equal_width_bins() {
        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![
                (0.5, 10.0),
                (1.0, 2.0),
                (1.49, 3.0),
                (1.5, 5.0),
                (2.0, 7.0),
                (2.1, 11.0),
            ],
        };

        let bins = spectrum
            .linear_binned_intensities(1.0, 2.0, 2)
            .expect("valid linear binning config should succeed");

        assert_eq!(bins, alloc::vec![5.0, 12.0]);
    }

    #[test]
    fn logarithmic_binned_intensities_accumulates_log_width_bins() {
        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![
                (0.5, 10.0),
                (1.0, 2.0),
                (9.0, 3.0),
                (10.0, 5.0),
                (100.0, 7.0),
                (101.0, 11.0),
            ],
        };

        let bins = spectrum
            .logarithmic_binned_intensities(1.0, 100.0, 2)
            .expect("valid logarithmic binning config should succeed");

        assert_eq!(bins, alloc::vec![5.0, 12.0]);
    }

    #[test]
    fn binning_rejects_invalid_parameters() {
        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(1.0, 1.0)],
        };

        assert_eq!(
            spectrum
                .linear_binned_intensities(1.0, 2.0, 0)
                .expect_err("zero bins should be rejected"),
            SimilarityComputationError::InvalidParameter("number_of_bins")
        );
        assert_eq!(
            spectrum
                .linear_binned_intensities(f64::NAN, 2.0, 2)
                .expect_err("non-finite min should be rejected"),
            SimilarityComputationError::NonFiniteValue("mz_min")
        );
        assert_eq!(
            spectrum
                .linear_binned_intensities(2.0, 2.0, 2)
                .expect_err("empty range should be rejected"),
            SimilarityComputationError::InvalidParameter("mz_range")
        );
        assert_eq!(
            spectrum
                .logarithmic_binned_intensities(0.0, 2.0, 2)
                .expect_err("non-positive logarithmic minimum should be rejected"),
            SimilarityComputationError::InvalidParameter("mz_min")
        );
    }

    #[test]
    fn binning_rejects_non_finite_peak_values() {
        let non_finite_mz = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(f64::NAN, 1.0)],
        };
        assert_eq!(
            non_finite_mz
                .linear_binned_intensities(1.0, 2.0, 2)
                .expect_err("non-finite peak mz should be rejected"),
            SimilarityComputationError::NonFiniteValue("mz")
        );

        let non_finite_intensity = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(1.0, f64::INFINITY)],
        };
        assert_eq!(
            non_finite_intensity
                .logarithmic_binned_intensities(1.0, 2.0, 2)
                .expect_err("non-finite intensity should be rejected"),
            SimilarityComputationError::NonFiniteValue("intensity")
        );
    }

    #[test]
    fn modified_matching_peaks_uses_shifted_window_only_when_needed() {
        let left = RawSpectrum {
            precursor_mz: 210.0,
            peaks: alloc::vec![(100.0, 1.0), (110.5, 1.0)],
        };
        let right = RawSpectrum {
            precursor_mz: 200.0,
            peaks: alloc::vec![(100.0, 1.0)],
        };

        let shifted = left
            .modified_matching_peaks(&right, 1.0, 210.0, 200.0)
            .expect("shifted matching graph should build");
        assert_eq!(shifted.number_of_defined_values(), 2);

        let unshifted = left
            .modified_matching_peaks(&right, 10.0, 210.0, 200.0)
            .expect("within-tolerance precursor difference should disable shift");
        assert_eq!(unshifted.number_of_defined_values(), 1);
    }

    #[test]
    fn collect_window_matches_can_advance_past_below_values_without_matching() {
        let other = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(0.8, 1.0)],
        };
        let mut matched_cols = Vec::new();
        let new_lowest = collect_window_matches(&other, 0, 1.0, 0.05, 0.0, "right_mz", |col| {
            matched_cols.push(col)
        })
        .expect("window collection should succeed");

        assert_eq!(new_lowest, 1);
        assert!(matched_cols.is_empty());
    }

    #[test]
    fn modified_matching_peaks_rejects_non_finite_tolerance() {
        let spectrum = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(1.0, 1.0)],
        };

        let error = spectrum
            .modified_matching_peaks(&spectrum, f64::NAN, 100.0, 100.0)
            .expect_err("non-finite tolerance should be rejected");
        assert_eq!(
            error,
            SimilarityComputationError::NonFiniteValue("mz_tolerance")
        );
    }

    #[test]
    fn modified_matching_peaks_propagates_shifted_non_finite_values() {
        let left = RawSpectrum {
            precursor_mz: 100.0,
            peaks: alloc::vec![(1.0, 1.0)],
        };
        let right = RawSpectrum {
            precursor_mz: -f64::MAX,
            peaks: alloc::vec![(f64::MAX, 1.0)],
        };

        let error = left
            .modified_matching_peaks(&right, 0.1, 100.0, -f64::MAX)
            .expect_err("non-finite shifted values should be rejected");
        assert_eq!(
            error,
            SimilarityComputationError::NonFiniteValue("shifted_other_mz")
        );
    }
}
