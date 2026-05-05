//! Shared internals for the Flash inverted m/z index.
//!
//! All items are `pub(crate)` — individual variant modules (`flash_cosine_index`,
//! `flash_entropy_index`) expose only their public wrappers.

use alloc::vec::Vec;
use core::{cmp::Ordering, ops::Range};

use geometric_traits::prelude::{CSR2D, MatrixMut, SparseMatrixMut, ValuedCSR2D};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::similarity_errors::{SimilarityComputationError, SimilarityConfigError};
use crate::traits::{Spectrum, SpectrumFloat};

const PREFIX_PRUNING_MIN_THRESHOLD: f64 = 0.85;
pub(crate) const DEFAULT_COSINE_SPECTRUM_BLOCK_SIZE: usize = 1024;
pub(crate) const DEFAULT_ENTROPY_SPECTRUM_BLOCK_SIZE: usize = 256;
const PRECURSOR_INDEX_PROGRESS_BATCH: usize = 8192;

type PostingCsr = CSR2D<u32, u32, u32>;
type UpperBoundCsr = ValuedCSR2D<u32, u32, u32, f64>;

/// Build a row-compressed posting matrix from sorted `(block, posting)` pairs.
///
/// Rows are spectrum or precursor blocks. Columns are indices into the
/// corresponding flat posting arrays, so the CSR owns only row structure and
/// keeps peak data in the compact metric-specific buffers.
fn posting_csr(
    number_of_rows: u32,
    number_of_columns: u32,
    entries: Vec<(u32, u32)>,
) -> Result<PostingCsr, SimilarityComputationError> {
    posting_csr_from_entries(number_of_rows, number_of_columns, entries, || {})
}

fn posting_csr_with_progress<G>(
    number_of_rows: u32,
    number_of_columns: u32,
    entries: Vec<(u32, u32)>,
    progress: &mut ProgressTicker<'_, G>,
) -> Result<PostingCsr, SimilarityComputationError>
where
    G: FlashIndexBuildProgress + ?Sized,
{
    posting_csr_from_entries(number_of_rows, number_of_columns, entries, || {
        progress.tick_one();
    })
}

fn posting_csr_from_entries(
    number_of_rows: u32,
    number_of_columns: u32,
    entries: Vec<(u32, u32)>,
    mut after_insert: impl FnMut(),
) -> Result<PostingCsr, SimilarityComputationError> {
    let number_of_entries =
        u32::try_from(entries.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
    let mut csr = PostingCsr::with_sparse_shaped_capacity(
        (number_of_rows, number_of_columns),
        number_of_entries,
    );
    for entry in entries {
        csr.add(entry)
            .map_err(|_| SimilarityComputationError::IndexConstructionFailed)?;
        after_insert();
    }
    Ok(csr)
}

/// Build a valued CSR for `(m/z-bin, spectrum-block) -> score upper bound`.
fn upper_bound_csr(
    number_of_rows: u32,
    number_of_columns: u32,
    entries: Vec<(u32, u32, f64)>,
) -> Result<UpperBoundCsr, SimilarityComputationError> {
    let number_of_entries =
        u32::try_from(entries.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
    let mut csr = UpperBoundCsr::with_sparse_shaped_capacity(
        (number_of_rows, number_of_columns),
        number_of_entries,
    );
    for entry in entries {
        csr.add(entry)
            .map_err(|_| SimilarityComputationError::IndexConstructionFailed)?;
    }
    Ok(csr)
}

pub(crate) struct PreparedFlashSpectrum<P: SpectrumFloat> {
    pub(crate) precursor_mz: P,
    pub(crate) mz: Vec<P>,
    pub(crate) data: Vec<P>,
}

pub(crate) type PreparedFlashSpectra<P> = Vec<PreparedFlashSpectrum<P>>;

/// Coarse-grained phases reported while building a Flash index.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FlashIndexBuildPhase {
    /// Convert user spectra into the normalized representation stored by the index.
    PrepareSpectra,
    /// Reorder spectra by the deterministic peak-signature layout used internally.
    ReorderSpectra,
    /// Build threshold/top-k spectrum-block upper bounds.
    BuildBlockUpperBounds,
    /// Build threshold/top-k block-local product-ion postings.
    BuildBlockProductIndex,
    /// Pack spectrum peaks into flat arrays for the global Flash index.
    PackFlashPeaks,
    /// Sort the global product-ion postings by product m/z.
    SortProductIndex,
    /// Sort the global neutral-loss postings by neutral-loss value.
    SortNeutralLossIndex,
    /// Build the 2D precursor/product posting index used by PEPMASS filtering.
    BuildPrecursorIndex,
    /// Index construction has completed.
    Finish,
}

impl FlashIndexBuildPhase {
    /// Human-readable phase label suitable for progress bars.
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::PrepareSpectra => "preparing spectra",
            Self::ReorderSpectra => "reordering spectra",
            Self::BuildBlockUpperBounds => "building block upper bounds",
            Self::BuildBlockProductIndex => "building block product index",
            Self::PackFlashPeaks => "packing flash peaks",
            Self::SortProductIndex => "sorting product index",
            Self::SortNeutralLossIndex => "sorting neutral-loss index",
            Self::BuildPrecursorIndex => "building precursor index",
            Self::Finish => "index ready",
        }
    }
}

/// Progress sink used by Flash index builders.
///
/// Implement this trait to receive coarse build phases and per-item ticks. With
/// the `indicatif` feature enabled, [`indicatif::ProgressBar`] implements this
/// trait directly and can be passed to [`crate::traits::SpectraIndexBuilder::progress`].
pub trait FlashIndexBuildProgress {
    /// Start a new build phase. `len` is present when the phase has a known
    /// number of ticks.
    fn start_phase(&self, phase: FlashIndexBuildPhase, len: Option<u64>);

    /// Advance the current phase by `delta` ticks.
    fn inc(&self, delta: u64);

    /// Mark construction as complete.
    fn finish(&self);
}

/// Shared construction options used by Flash index builders.
///
/// This keeps execution mode, construction progress, and optional precursor
/// filtering in one place instead of multiplying public construction APIs for
/// every combination of those choices.
#[derive(Clone, Copy)]
pub struct FlashIndexBuildOptions<'a> {
    parallel: bool,
    progress: Option<&'a (dyn FlashIndexBuildProgress + Sync + 'a)>,
    pepmass_filter: PepmassFilter,
}

impl<'a> Default for FlashIndexBuildOptions<'a> {
    fn default() -> Self {
        Self {
            parallel: false,
            progress: None,
            pepmass_filter: PepmassFilter::disabled(),
        }
    }
}

impl<'a> FlashIndexBuildOptions<'a> {
    /// Returns whether construction should use Rayon-backed preparation and
    /// sorting where available.
    #[inline]
    #[must_use]
    pub const fn parallel(&self) -> bool {
        self.parallel
    }

    /// Sets whether construction should use Rayon-backed preparation and
    /// sorting where available.
    #[inline]
    pub(crate) const fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Returns the progress sink configured for index construction.
    #[inline]
    pub(crate) fn progress(&self) -> &(dyn FlashIndexBuildProgress + Sync + 'a) {
        static NOOP_PROGRESS: NoopFlashIndexBuildProgress = NoopFlashIndexBuildProgress;
        self.progress.unwrap_or(&NOOP_PROGRESS)
    }

    /// Sets the progress sink used during construction.
    #[inline]
    pub(crate) fn set_progress(&mut self, progress: &'a (dyn FlashIndexBuildProgress + Sync + 'a)) {
        self.progress = Some(progress);
    }

    /// Returns the precursor-mass filter requested at construction time.
    #[inline]
    #[must_use]
    pub const fn pepmass_filter(&self) -> PepmassFilter {
        self.pepmass_filter
    }

    /// Sets the precursor-mass filter requested at construction time.
    #[inline]
    pub(crate) const fn set_pepmass_filter(&mut self, pepmass_filter: PepmassFilter) {
        self.pepmass_filter = pepmass_filter;
    }
}

/// Progress sink used by row-oriented Flash searches.
///
/// Implement this trait to receive one tick per completed query row. With the
/// `indicatif` feature enabled, [`indicatif::ProgressBar`] implements this
/// trait directly and can be passed to row selections through
/// `FlashCosineSelfSimilarityIndex::rows().progress(...)`.
pub trait FlashRowSearchProgress {
    /// Start row search over `len` known rows.
    fn start(&self, len: u64);

    /// Advance the row search progress by `delta` completed rows.
    fn inc(&self, delta: u64);

    /// Mark row search as complete.
    fn finish(&self);
}

/// Progress sink that deliberately ignores all build progress events.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopFlashIndexBuildProgress;

/// Progress sink that deliberately ignores all row search progress events.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopFlashRowSearchProgress;

impl FlashIndexBuildProgress for NoopFlashIndexBuildProgress {
    #[inline]
    fn start_phase(&self, _phase: FlashIndexBuildPhase, _len: Option<u64>) {}

    #[inline]
    fn inc(&self, _delta: u64) {}

    #[inline]
    fn finish(&self) {}
}

impl FlashRowSearchProgress for NoopFlashRowSearchProgress {
    #[inline]
    fn start(&self, _len: u64) {}

    #[inline]
    fn inc(&self, _delta: u64) {}

    #[inline]
    fn finish(&self) {}
}

#[cfg(feature = "indicatif")]
impl FlashIndexBuildProgress for indicatif::ProgressBar {
    fn start_phase(&self, phase: FlashIndexBuildPhase, len: Option<u64>) {
        self.reset();
        self.set_message(phase.message());
        self.set_position(0);
        if let Some(len) = len {
            self.set_length(len);
        }
    }

    #[inline]
    fn inc(&self, delta: u64) {
        self.inc(delta);
    }

    fn finish(&self) {
        self.finish_with_message(FlashIndexBuildPhase::Finish.message());
    }
}

#[cfg(feature = "indicatif")]
impl FlashRowSearchProgress for indicatif::ProgressBar {
    fn start(&self, len: u64) {
        self.reset();
        self.set_message("searching rows");
        self.set_position(0);
        self.set_length(len);
    }

    #[inline]
    fn inc(&self, delta: u64) {
        self.inc(delta);
    }

    fn finish(&self) {
        self.finish_with_message("rows ready");
    }
}

pub(crate) fn progress_len_from_size_hint(size_hint: (usize, Option<usize>)) -> Option<u64> {
    let (lower, upper) = size_hint;
    upper
        .filter(|&upper| upper == lower)
        .and_then(|len| u64::try_from(len).ok())
}

/// Batches fine-grained progress ticks so large inner loops can report useful
/// progress without calling the sink once per posting.
struct ProgressTicker<'a, G: FlashIndexBuildProgress + ?Sized> {
    progress: &'a G,
    pending: usize,
    batch_size: usize,
}

impl<'a, G> ProgressTicker<'a, G>
where
    G: FlashIndexBuildProgress + ?Sized,
{
    fn new(progress: &'a G, batch_size: usize) -> Self {
        Self {
            progress,
            pending: 0,
            batch_size: batch_size.max(1),
        }
    }

    #[inline]
    fn tick_one(&mut self) {
        self.tick(1);
    }

    fn tick(&mut self, delta: usize) {
        if delta == 0 {
            return;
        }

        self.pending = self.pending.saturating_add(delta);
        if self.pending >= self.batch_size {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.pending == 0 {
            return;
        }

        self.progress
            .inc(u64::try_from(self.pending).unwrap_or(u64::MAX));
        self.pending = 0;
    }
}

/// Return the number of coarse work ticks reported while building the
/// PEPMASS 2D index.
fn pepmass_index_progress_len(
    n_spectra: usize,
    total_product_peaks: usize,
    total_neutral_loss_peaks: usize,
) -> Option<u64> {
    let spectrum_work = n_spectra.checked_mul(2)?;
    let product_work = total_product_peaks.checked_mul(3)?;
    let neutral_loss_work = total_neutral_loss_peaks.checked_mul(3)?;
    let total = spectrum_work
        .checked_add(product_work)?
        .checked_add(neutral_loss_work)?;
    u64::try_from(total).ok()
}

/// Choose the precursor-bin width for the PEPMASS 2D index.
///
/// A bin spans the full query PEPMASS window width, so any query overlaps only
/// a small number of occupied bins. Zero-tolerance filters use a finite
/// fallback width and rely on the final exact precursor check.
fn pepmass_precursor_bin_width(filter: PepmassFilter) -> Option<f64> {
    filter.tolerance().map(|tolerance| {
        if tolerance > 0.0 {
            2.0 * tolerance
        } else {
            1.0
        }
    })
}

/// Convert a precursor m/z value into the row bin used by the PEPMASS 2D index.
#[inline]
fn pepmass_precursor_bin_id(precursor_mz: f64, bin_width: f64) -> i64 {
    (precursor_mz / bin_width).floor() as i64
}

/// Build `(precursor-bin-row, posting)` CSR entries by counting postings per
/// occupied precursor bin. `posting_spec_id` is already in product-m/z or
/// neutral-loss order, so stable bucketing preserves the searchable order
/// inside each precursor bin without a full posting tuple sort.
fn posting_entries_by_precursor_bin<G>(
    spectrum_bin_row: &[u32],
    posting_spec_id: &[u32],
    n_rows: usize,
    progress: &mut ProgressTicker<'_, G>,
) -> Result<Vec<(u32, u32)>, SimilarityComputationError>
where
    G: FlashIndexBuildProgress + ?Sized,
{
    let mut counts = alloc::vec![0usize; n_rows];
    for &spec_id in posting_spec_id {
        let &row_id = spectrum_bin_row
            .get(spec_id as usize)
            .ok_or(SimilarityComputationError::IndexOverflow)?;
        let count = counts
            .get_mut(row_id as usize)
            .ok_or(SimilarityComputationError::IndexOverflow)?;
        *count = count
            .checked_add(1)
            .ok_or(SimilarityComputationError::IndexOverflow)?;
        progress.tick_one();
    }

    let mut offsets = Vec::with_capacity(n_rows + 1);
    offsets.push(0usize);
    for &count in &counts {
        let next = offsets
            .last()
            .copied()
            .and_then(|offset| offset.checked_add(count))
            .ok_or(SimilarityComputationError::IndexOverflow)?;
        offsets.push(next);
    }

    let mut next_offsets = offsets[..n_rows].to_vec();
    let mut entries = alloc::vec![(0u32, 0u32); posting_spec_id.len()];
    for (posting_index, &spec_id) in posting_spec_id.iter().enumerate() {
        let &row_id = spectrum_bin_row
            .get(spec_id as usize)
            .ok_or(SimilarityComputationError::IndexOverflow)?;
        let entry_offset = next_offsets
            .get_mut(row_id as usize)
            .ok_or(SimilarityComputationError::IndexOverflow)?;
        let output_index = *entry_offset;
        *entry_offset = entry_offset
            .checked_add(1)
            .ok_or(SimilarityComputationError::IndexOverflow)?;
        entries[output_index] = (
            row_id,
            u32::try_from(posting_index).map_err(|_| SimilarityComputationError::IndexOverflow)?,
        );
        progress.tick_one();
    }

    Ok(entries)
}

/// Optional precursor-mass filter for Flash index searches.
///
/// When enabled, searches only score library spectra whose precursor m/z is
/// within `tolerance` of the query precursor m/z. This is useful for MGF
/// `PEPMASS`-style workflows where product-ion similarity should only be
/// considered inside a precursor mass window.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PepmassFilter {
    tolerance: Option<f64>,
}

impl PepmassFilter {
    /// Create a disabled precursor-mass filter.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { tolerance: None }
    }

    /// Create a filter that accepts candidates within `tolerance` Da.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityConfigError`] if `tolerance` is negative or
    /// non-finite.
    pub fn within_tolerance(tolerance: f64) -> Result<Self, SimilarityConfigError> {
        if !tolerance.is_finite() {
            return Err(SimilarityConfigError::NonFiniteParameter(
                "pepmass_tolerance",
            ));
        }
        if tolerance < 0.0 {
            return Err(SimilarityConfigError::InvalidParameter("pepmass_tolerance"));
        }
        Ok(Self {
            tolerance: Some(tolerance),
        })
    }

    /// Returns whether this filter rejects candidates by precursor mass.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        self.tolerance.is_some()
    }

    /// Returns the configured tolerance, if the filter is enabled.
    #[must_use]
    pub const fn tolerance(self) -> Option<f64> {
        self.tolerance
    }

    /// Returns whether `library_precursor_mz` is accepted for `query_precursor_mz`.
    #[inline]
    pub(crate) fn allows(self, query_precursor_mz: Option<f64>, library_precursor_mz: f64) -> bool {
        let Some(tolerance) = self.tolerance else {
            return true;
        };
        let Some(query_precursor_mz) = query_precursor_mz else {
            return false;
        };
        (query_precursor_mz - library_precursor_mz).abs() <= tolerance
    }
}

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Default)]
pub(crate) struct SpectrumIdMap {
    internal_to_public: Vec<u32>,
    public_to_internal: Vec<u32>,
}

impl SpectrumIdMap {
    pub(crate) fn identity() -> Self {
        Self::default()
    }

    fn from_internal_to_public(
        internal_to_public: Vec<u32>,
    ) -> Result<Self, SimilarityComputationError> {
        if internal_to_public.is_empty() {
            return Ok(Self::identity());
        }

        let mut public_to_internal = alloc::vec![u32::MAX; internal_to_public.len()];
        for (internal_id, &public_id) in internal_to_public.iter().enumerate() {
            let public_index = usize::try_from(public_id)
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
            if public_index >= public_to_internal.len()
                || public_to_internal[public_index] != u32::MAX
            {
                return Err(SimilarityComputationError::IndexOverflow);
            }
            public_to_internal[public_index] = u32::try_from(internal_id)
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
        }

        Ok(Self {
            internal_to_public,
            public_to_internal,
        })
    }

    #[inline]
    pub(crate) fn public_to_internal(&self, public_id: u32) -> Option<u32> {
        if self.public_to_internal.is_empty() {
            return Some(public_id);
        }
        self.public_to_internal.get(public_id as usize).copied()
    }

    #[inline]
    pub(crate) fn internal_to_public(&self, internal_id: u32) -> u32 {
        if self.internal_to_public.is_empty() {
            return internal_id;
        }
        self.internal_to_public[internal_id as usize]
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct SpectrumReorderKey {
    top_mz_bins: [i64; 8],
    precursor_bin: i64,
    n_peaks: u32,
    public_id: u32,
}

pub(crate) fn reorder_prepared_spectra_by_signature<P: SpectrumFloat>(
    spectra: PreparedFlashSpectra<P>,
    tolerance: f64,
) -> Result<(PreparedFlashSpectra<P>, SpectrumIdMap), SimilarityComputationError> {
    let n_spectra: u32 =
        u32::try_from(spectra.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
    if n_spectra <= 1 {
        return Ok((spectra, SpectrumIdMap::identity()));
    }

    let bin_width = if tolerance > 0.0 {
        (10.0 * tolerance).max(tolerance)
    } else {
        1.0
    };
    let mut keyed = Vec::with_capacity(spectra.len());
    for (public_id, spectrum) in spectra.into_iter().enumerate() {
        let public_id =
            u32::try_from(public_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let key = spectrum_reorder_key(&spectrum, public_id, bin_width)?;
        keyed.push((key, spectrum));
    }

    keyed.sort_unstable_by_key(|(key, _)| *key);

    let mut internal_to_public = Vec::with_capacity(keyed.len());
    let mut reordered = Vec::with_capacity(keyed.len());
    for (key, spectrum) in keyed {
        internal_to_public.push(key.public_id);
        reordered.push(spectrum);
    }
    let id_map = SpectrumIdMap::from_internal_to_public(internal_to_public)?;

    Ok((reordered, id_map))
}

fn spectrum_reorder_key<P: SpectrumFloat>(
    spectrum: &PreparedFlashSpectrum<P>,
    public_id: u32,
    bin_width: f64,
) -> Result<SpectrumReorderKey, SimilarityComputationError> {
    let precursor_bin = mz_bin(spectrum.precursor_mz.to_f64(), bin_width)?;
    let n_peaks =
        u32::try_from(spectrum.mz.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
    let mut top_mz_bins = [i64::MAX; 8];
    for (target, &mz) in top_mz_bins.iter_mut().zip(spectrum.mz.iter()) {
        *target = mz_bin(mz.to_f64(), bin_width)?;
    }

    Ok(SpectrumReorderKey {
        top_mz_bins,
        precursor_bin,
        n_peaks,
        public_id,
    })
}

fn mz_bin(mz: f64, bin_width: f64) -> Result<i64, SimilarityComputationError> {
    if !mz.is_finite() || !bin_width.is_finite() || bin_width <= 0.0 {
        return Err(SimilarityComputationError::NonFiniteValue("mz"));
    }
    Ok((mz / bin_width).floor() as i64)
}

#[derive(Clone, Copy)]
enum SortBackend {
    Sequential,
    #[cfg(feature = "rayon")]
    Parallel,
}

pub(crate) fn convert_flash_value<P: SpectrumFloat>(
    value: f64,
    name: &'static str,
) -> Result<P, SimilarityComputationError> {
    if !value.is_finite() {
        return Err(SimilarityComputationError::NonFiniteValue(name));
    }
    P::from_f64(value).ok_or(SimilarityComputationError::NonFiniteValue(name))
}

pub(crate) fn convert_flash_values<P: SpectrumFloat>(
    values: impl IntoIterator<Item = f64>,
    name: &'static str,
) -> Result<Vec<P>, SimilarityComputationError> {
    values
        .into_iter()
        .map(|value| convert_flash_value(value, name))
        .collect()
}

pub(crate) fn flash_values_to_f64<P: SpectrumFloat>(values: &[P]) -> Vec<f64> {
    values.iter().map(|value| value.to_f64()).collect()
}

fn compare_indexed_values<P: SpectrumFloat>(values: &[P], left: u32, right: u32) -> Ordering {
    values[left as usize]
        .to_f64()
        .total_cmp(&values[right as usize].to_f64())
        .then_with(|| left.cmp(&right))
}

fn sort_permutation_by_values<P>(perm: &mut [u32], values: &[P], backend: SortBackend)
where
    P: SpectrumFloat + Sync,
{
    match backend {
        SortBackend::Sequential => {
            perm.sort_unstable_by(|&left, &right| compare_indexed_values(values, left, right));
        }
        #[cfg(feature = "rayon")]
        SortBackend::Parallel => {
            perm.par_sort_unstable_by(|&left, &right| compare_indexed_values(values, left, right));
        }
    }
}

struct SearchBitVec {
    words: Vec<usize>,
    len: usize,
}

impl SearchBitVec {
    const WORD_BITS: usize = usize::BITS as usize;

    fn new() -> Self {
        Self {
            words: Vec::new(),
            len: 0,
        }
    }

    fn zeros(len: usize) -> Self {
        Self {
            words: alloc::vec![0; len.div_ceil(Self::WORD_BITS)],
            len,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.len);
        let word = self.words[index / Self::WORD_BITS];
        let mask = 1usize << (index % Self::WORD_BITS);
        word & mask != 0
    }

    #[inline]
    fn set(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.len);
        let word = &mut self.words[index / Self::WORD_BITS];
        let mask = 1usize << (index % Self::WORD_BITS);
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    #[cfg(feature = "mem_size")]
    fn stored_bytes(&self, flags: mem_dbg::SizeFlags) -> usize {
        let words = if flags.contains(mem_dbg::SizeFlags::CAPACITY) {
            self.words.capacity()
        } else {
            self.words.len()
        };
        words * core::mem::size_of::<usize>()
    }
}

#[cfg(feature = "mem_size")]
impl mem_dbg::FlatType for SearchBitVec {
    type Flat = mem_dbg::False;
}

#[cfg(feature = "mem_size")]
impl mem_dbg::MemSize for SearchBitVec {
    fn mem_size_rec(
        &self,
        flags: mem_dbg::SizeFlags,
        _refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        core::mem::size_of::<Self>() + self.stored_bytes(flags)
    }
}

#[cfg(feature = "mem_dbg")]
impl mem_dbg::MemDbgImpl for SearchBitVec {}

// ---------------------------------------------------------------------------
// FlashKernel — scoring kernel abstraction
// ---------------------------------------------------------------------------

/// Scoring kernel abstraction for the Flash index.
///
/// Each kernel defines how to prepare per-peak scoring data, compute per-spectrum
/// metadata (e.g. a norm), score a single matched pair, and finalize the
/// accumulated raw score into a `[0, 1]` similarity.
pub(crate) trait FlashKernel {
    /// Per-spectrum metadata (norm for cosine, `()` for entropy).
    type SpectrumMeta: Copy + Default;

    /// Compute per-spectrum metadata from all prepared peak data values.
    fn spectrum_meta<P: SpectrumFloat>(peak_data: &[P]) -> Self::SpectrumMeta;

    /// Score contribution of a single matched pair.
    fn pair_score(query: f64, library: f64) -> f64;

    /// Finalize accumulated raw score into `[0, 1]`.
    fn finalize(
        raw: f64,
        n_matches: usize,
        query_meta: &Self::SpectrumMeta,
        lib_meta: &Self::SpectrumMeta,
    ) -> f64;
}

// ---------------------------------------------------------------------------
// FlashSearchResult
// ---------------------------------------------------------------------------

/// A single search result from a Flash index query.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlashSearchResult {
    /// Index of the library spectrum (0-based, insertion order).
    pub spectrum_id: u32,
    /// Similarity score in `[0, 1]`.
    pub score: f64,
    /// Number of matched peak pairs.
    pub n_matches: usize,
}

/// Per-query diagnostic counters for Flash index searches.
///
/// These counters are intended for benchmarks and profiling. They describe the
/// most recent search run with a [`SearchState`] and are reset by public search
/// entry points before executing a query.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(flat))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlashSearchDiagnostics {
    /// Product-ion postings visited while scanning m/z windows.
    pub product_postings_visited: usize,
    /// Unique primary candidates marked for exact scoring.
    pub candidates_marked: usize,
    /// Candidates passed to exact scoring.
    pub candidates_rescored: usize,
    /// Results emitted after threshold/top-k filtering.
    pub results_emitted: usize,
    /// Spectrum blocks considered by a threshold upper-bound filter.
    pub spectrum_blocks_evaluated: usize,
    /// Spectrum blocks whose upper bound was high enough to keep.
    pub spectrum_blocks_allowed: usize,
    /// Spectrum blocks rejected before candidate marking.
    pub spectrum_blocks_pruned: usize,
}

pub(crate) fn compare_search_results_by_rank(
    left: &FlashSearchResult,
    right: &FlashSearchResult,
) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| right.n_matches.cmp(&left.n_matches))
        .then_with(|| left.spectrum_id.cmp(&right.spectrum_id))
}

/// Reusable scratch space for top-k Flash index searches.
///
/// Use one `TopKSearchState` per worker together with [`SearchState`] when
/// running many top-k queries, so the bounded candidate buffer can be reused
/// instead of allocated for every query.
///
/// # Example
///
/// ```
/// use mass_spectrometry::prelude::*;
///
/// let mut left: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
/// left.add_peaks([(100.0, 10.0), (200.0, 20.0)]).unwrap();
/// let mut right: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 2).unwrap();
/// right.add_peaks([(100.05, 10.0), (200.05, 20.0)]).unwrap();
///
/// let spectra = vec![left, right];
/// let index = FlashCosineIndex::<f64>::builder()
///     .mz_power(0.0)
///     .intensity_power(1.0)
///     .mz_tolerance(0.1)
///     .build(&spectra)
///     .unwrap();
/// let mut search_state = index.new_search_state();
/// let mut top_k_state = TopKSearchState::new();
/// let mut hits = Vec::new();
///
/// index
///     .for_each_top_k_threshold_with_state(
///         &spectra[0],
///         2,
///         0.8,
///         &mut search_state,
///         &mut top_k_state,
///         |hit| hits.push(hit),
///     )
///     .unwrap();
///
/// assert_eq!(hits[0].spectrum_id, 0);
/// assert!(hits.iter().any(|hit| hit.spectrum_id == 1));
/// ```
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Debug, Default)]
pub struct TopKSearchState {
    results: Vec<FlashSearchResult>,
}

impl TopKSearchState {
    /// Create an empty top-k scratch state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) struct TopKSearchResults<'a> {
    k: usize,
    min_score: f64,
    results: &'a mut Vec<FlashSearchResult>,
    worst_index: Option<usize>,
    pruning_score: f64,
}

impl<'a> TopKSearchResults<'a> {
    /// Prepare the bounded result buffer for one query.
    pub(crate) fn new(k: usize, min_score: f64, state: &'a mut TopKSearchState) -> Self {
        state.results.clear();
        if state.results.capacity() < k {
            state.results.reserve(k - state.results.capacity());
        }
        Self {
            k,
            min_score,
            results: &mut state.results,
            worst_index: None,
            pruning_score: min_score,
        }
    }

    /// Insert a result if it is good enough for the current top-k set.
    #[inline]
    pub(crate) fn push(&mut self, result: FlashSearchResult) -> bool {
        if self.k == 0 || result.score < self.min_score {
            return false;
        }
        if self.results.len() < self.k {
            self.results.push(result);
            if self.results.len() == self.k {
                self.refresh_worst();
            }
            return true;
        }

        let worst_index = self
            .worst_index
            .expect("full top-k buffer should have a cached worst index");
        let worst_result = self.results[worst_index];

        if compare_search_results_by_rank(&result, &worst_result).is_lt() {
            self.results[worst_index] = result;
            self.refresh_worst();
            return true;
        }
        false
    }

    /// Return the score that a future candidate must reach to matter.
    #[inline]
    pub(crate) fn pruning_score(&self) -> f64 {
        self.pruning_score
    }

    /// Return the number of results currently retained.
    pub(crate) fn len(&self) -> usize {
        self.results.len()
    }

    /// Sort selected results into the public deterministic rank order.
    fn finish(&mut self) {
        self.results.sort_by(compare_search_results_by_rank);
    }

    /// Refresh the cached replacement position and pruning score.
    fn refresh_worst(&mut self) {
        if self.results.len() < self.k {
            self.worst_index = None;
            self.pruning_score = self.min_score;
            return;
        }

        let Some((worst_index, worst_result)) = self
            .results
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| compare_search_results_by_rank(left, right))
        else {
            self.worst_index = None;
            self.pruning_score = self.min_score;
            return;
        };

        self.worst_index = Some(worst_index);
        self.pruning_score = worst_result.score.max(self.min_score);
    }

    /// Emit selected results after final ranking.
    pub(crate) fn emit<Emit>(mut self, mut emit: Emit)
    where
        Emit: FnMut(FlashSearchResult),
    {
        self.finish();
        for &result in self.results.iter() {
            emit(result);
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DirectThresholdSearch<'a, K: FlashKernel, Q: SpectrumFloat> {
    pub(crate) query_mz: &'a [Q],
    pub(crate) query_data: &'a [Q],
    pub(crate) query_meta: &'a K::SpectrumMeta,
    pub(crate) score_threshold: f64,
    pub(crate) query_precursor_mz: Option<f64>,
}

/// Product-ion postings partitioned by contiguous spectrum-id blocks.
///
/// The global product index is optimal when every spectrum is eligible. Once a
/// query-level block bound has selected a tiny subset of spectrum blocks, this
/// layout lets candidate generation binary-search only those block-local peak
/// lists and skip postings from every rejected block.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct SpectrumBlockProductIndex<P: SpectrumFloat = f64> {
    block_size: u32,
    n_blocks: u32,
    postings: PostingCsr,
    mz: Vec<P>,
    data: Vec<P>,
    spec_id: Vec<u32>,
}

impl<P: SpectrumFloat> SpectrumBlockProductIndex<P> {
    pub(crate) fn build(
        spectra: &[PreparedFlashSpectrum<P>],
        block_size: usize,
    ) -> Result<Self, SimilarityComputationError> {
        let block_size = block_size.max(1);
        let block_size_u32 =
            u32::try_from(block_size).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let n_blocks = spectra.len().div_ceil(block_size);
        let n_blocks_u32 =
            u32::try_from(n_blocks).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let total_peaks: usize = spectra.iter().map(|spectrum| spectrum.mz.len()).sum();
        let total_peaks_u32 =
            u32::try_from(total_peaks).map_err(|_| SimilarityComputationError::IndexOverflow)?;

        let mut entries: Vec<(u32, P, P, u32)> = Vec::with_capacity(total_peaks);
        for (spec_id, spectrum) in spectra.iter().enumerate() {
            let spec_id_u32 =
                u32::try_from(spec_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            let block_id = u32::try_from(spec_id / block_size)
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
            for (&mz, &data) in spectrum.mz.iter().zip(spectrum.data.iter()) {
                entries.push((block_id, mz, data, spec_id_u32));
            }
        }
        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.to_f64().total_cmp(&right.1.to_f64()))
                .then_with(|| left.3.cmp(&right.3))
        });

        let mut mz = Vec::with_capacity(total_peaks);
        let mut data = Vec::with_capacity(total_peaks);
        let mut spec_id = Vec::with_capacity(total_peaks);
        let mut posting_entries = Vec::with_capacity(total_peaks);
        for (block_id, entry_mz, entry_data, entry_spec_id) in entries {
            let entry_id =
                u32::try_from(mz.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            posting_entries.push((block_id, entry_id));
            mz.push(entry_mz);
            data.push(entry_data);
            spec_id.push(entry_spec_id);
        }
        let postings = posting_csr(n_blocks_u32, total_peaks_u32, posting_entries)?;

        Ok(Self {
            block_size: block_size_u32,
            n_blocks: n_blocks_u32,
            postings,
            mz,
            data,
            spec_id,
        })
    }

    #[inline]
    pub(crate) fn spectrum_block_id(&self, spec_id: u32) -> u32 {
        spec_id / self.block_size
    }

    pub(crate) fn for_each_peak_in_window<Q: SpectrumFloat>(
        &self,
        block_id: u32,
        mz: Q,
        tolerance: f64,
        mut emit: impl FnMut(u32, P),
    ) -> usize {
        if block_id >= self.n_blocks {
            return 0;
        }

        let query_mz = mz.to_f64();
        let lo = query_mz - tolerance;
        let hi = query_mz + tolerance;
        let row_entries = self.postings.sparse_row_slice(block_id);
        let first =
            row_entries.partition_point(|&entry_id| self.mz[entry_id as usize].to_f64() < lo);

        let mut visited = 0usize;
        for &entry_id in &row_entries[first..] {
            let entry_id = entry_id as usize;
            let product_mz = self.mz[entry_id].to_f64();
            if product_mz > hi {
                break;
            }
            visited = visited.saturating_add(1);
            emit(self.spec_id[entry_id], self.data[entry_id]);
        }

        visited
    }
}

/// Product and neutral-loss postings grouped by precursor-mass bins.
///
/// This index is only used when a [`PepmassFilter`] is enabled. It answers the
/// two-dimensional query "precursor m/z within the PEPMASS window and
/// product/neutral-loss m/z within the peak window" by first selecting occupied
/// precursor bins and then binary-searching the value-sorted posting ids inside
/// those bins. Boundary bins may over-include spectra, so callers still apply
/// the exact precursor check before scoring.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct Pepmass2DPostingIndex {
    bin_width: f64,
    precursor_bins: Vec<i64>,
    product_bins: PostingCsr,
    neutral_loss_bins: PostingCsr,
}

#[derive(Clone, Copy)]
struct ProductPostingSlices<'a, P: SpectrumFloat> {
    mz: &'a [P],
    spec_id: &'a [u32],
    data: &'a [P],
}

#[derive(Clone, Copy)]
struct NeutralLossPostingSlices<'a, P: SpectrumFloat> {
    value: &'a [P],
    spec_id: &'a [u32],
    data: &'a [P],
    to_product: &'a [u32],
}

impl Pepmass2DPostingIndex {
    pub(crate) fn build<P, G>(
        spectrum_precursor_mz: &[P],
        product_spec_id: &[u32],
        neutral_loss_spec_id: &[u32],
        filter: PepmassFilter,
        progress: &G,
    ) -> Result<Self, SimilarityComputationError>
    where
        P: SpectrumFloat,
        G: FlashIndexBuildProgress + ?Sized,
    {
        let bin_width = pepmass_precursor_bin_width(filter).ok_or(
            SimilarityComputationError::InvalidParameter("pepmass_2d_index"),
        )?;
        let total_product_peaks = product_spec_id.len();
        let total_product_peaks_u32 = u32::try_from(total_product_peaks)
            .map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let total_neutral_loss_peaks = neutral_loss_spec_id.len();
        let total_neutral_loss_peaks_u32 = u32::try_from(total_neutral_loss_peaks)
            .map_err(|_| SimilarityComputationError::IndexOverflow)?;

        progress.start_phase(
            FlashIndexBuildPhase::BuildPrecursorIndex,
            pepmass_index_progress_len(
                spectrum_precursor_mz.len(),
                total_product_peaks,
                total_neutral_loss_peaks,
            ),
        );
        let mut progress_ticker = ProgressTicker::new(progress, PRECURSOR_INDEX_PROGRESS_BATCH);

        let mut precursor_bins = Vec::with_capacity(spectrum_precursor_mz.len());
        for &precursor_mz in spectrum_precursor_mz {
            let precursor_mz = precursor_mz.to_f64();
            if !precursor_mz.is_finite() {
                return Err(SimilarityComputationError::NonFiniteValue("precursor_mz"));
            }
            precursor_bins.push(pepmass_precursor_bin_id(precursor_mz, bin_width));
            progress_ticker.tick_one();
        }

        precursor_bins.sort_unstable();
        precursor_bins.dedup();
        let n_rows = precursor_bins.len();
        let n_rows_u32 =
            u32::try_from(n_rows).map_err(|_| SimilarityComputationError::IndexOverflow)?;

        let mut spectrum_bin_row = Vec::with_capacity(spectrum_precursor_mz.len());
        for &precursor_mz in spectrum_precursor_mz {
            let bin_id = pepmass_precursor_bin_id(precursor_mz.to_f64(), bin_width);
            let row_id = precursor_bins
                .binary_search(&bin_id)
                .map_err(|_| SimilarityComputationError::IndexConstructionFailed)?;
            spectrum_bin_row.push(
                u32::try_from(row_id).map_err(|_| SimilarityComputationError::IndexOverflow)?,
            );
            progress_ticker.tick_one();
        }

        // Product posting ids are already ordered by product m/z globally, so
        // stable precursor-bin bucketing preserves the m/z order inside each
        // row while satisfying CSR insertion order.
        let product_entries = posting_entries_by_precursor_bin(
            &spectrum_bin_row,
            product_spec_id,
            n_rows,
            &mut progress_ticker,
        )?;

        // Neutral-loss posting ids follow the same invariant, but sorted by
        // neutral-loss value instead of product m/z.
        let neutral_loss_entries = posting_entries_by_precursor_bin(
            &spectrum_bin_row,
            neutral_loss_spec_id,
            n_rows,
            &mut progress_ticker,
        )?;

        debug_assert!(u32::try_from(total_product_peaks).is_ok());
        debug_assert!(u32::try_from(total_neutral_loss_peaks).is_ok());
        debug_assert!(u32::try_from(spectrum_precursor_mz.len()).is_ok());

        let index = Self {
            bin_width,
            precursor_bins,
            product_bins: posting_csr_with_progress(
                n_rows_u32,
                total_product_peaks_u32,
                product_entries,
                &mut progress_ticker,
            )?,
            neutral_loss_bins: posting_csr_with_progress(
                n_rows_u32,
                total_neutral_loss_peaks_u32,
                neutral_loss_entries,
                &mut progress_ticker,
            )?,
        };
        progress_ticker.flush();
        Ok(index)
    }

    #[inline]
    fn matches_filter(&self, filter: PepmassFilter) -> bool {
        pepmass_precursor_bin_width(filter)
            .is_some_and(|bin_width| self.bin_width.to_bits() == bin_width.to_bits())
    }

    #[inline]
    fn matching_row_range(
        &self,
        filter: PepmassFilter,
        query_precursor_mz: Option<f64>,
    ) -> Range<usize> {
        let Some(tolerance) = filter.tolerance() else {
            return 0..self.precursor_bins.len();
        };
        let Some(query_precursor_mz) = query_precursor_mz else {
            return 0..0;
        };

        let lo = query_precursor_mz - tolerance;
        let hi = query_precursor_mz + tolerance;
        let lo_bin = pepmass_precursor_bin_id(lo, self.bin_width);
        let hi_bin = pepmass_precursor_bin_id(hi, self.bin_width);
        let start = self
            .precursor_bins
            .partition_point(|&bin_id| bin_id < lo_bin);
        let end = self
            .precursor_bins
            .partition_point(|&bin_id| bin_id <= hi_bin);
        start..end
    }

    fn for_each_product_peak_in_window<P, Q>(
        &self,
        rows: Range<usize>,
        mz: Q,
        tolerance: f64,
        postings: ProductPostingSlices<'_, P>,
        mut emit: impl FnMut(usize, u32, P),
    ) -> usize
    where
        P: SpectrumFloat,
        Q: SpectrumFloat,
    {
        let query_mz = mz.to_f64();
        let lo = query_mz - tolerance;
        let hi = query_mz + tolerance;
        let mut visited = 0usize;

        for row_id in rows {
            let row_id = row_id as u32;
            let row_indices = self.product_bins.sparse_row_slice(row_id);
            let first = row_indices.partition_point(|&product_index| {
                postings.mz[product_index as usize].to_f64() < lo
            });

            for &product_index in &row_indices[first..] {
                let product_index = product_index as usize;
                let indexed_mz = postings.mz[product_index].to_f64();
                if indexed_mz > hi {
                    break;
                }
                visited = visited.saturating_add(1);
                emit(
                    product_index,
                    postings.spec_id[product_index],
                    postings.data[product_index],
                );
            }
        }

        visited
    }

    fn for_each_neutral_loss_in_window<P>(
        &self,
        rows: Range<usize>,
        neutral_loss: f64,
        tolerance: f64,
        postings: NeutralLossPostingSlices<'_, P>,
        mut emit: impl FnMut(usize, u32, P, usize),
    ) where
        P: SpectrumFloat,
    {
        let lo = neutral_loss - tolerance;
        let hi = neutral_loss + tolerance;

        for row_id in rows {
            let row_id = row_id as u32;
            let row_indices = self.neutral_loss_bins.sparse_row_slice(row_id);
            let first = row_indices.partition_point(|&neutral_loss_index| {
                postings.value[neutral_loss_index as usize].to_f64() < lo
            });

            for &neutral_loss_index in &row_indices[first..] {
                let neutral_loss_index = neutral_loss_index as usize;
                let indexed_neutral_loss = postings.value[neutral_loss_index].to_f64();
                if indexed_neutral_loss > hi {
                    break;
                }
                emit(
                    neutral_loss_index,
                    postings.spec_id[neutral_loss_index],
                    postings.data[neutral_loss_index],
                    postings.to_product[neutral_loss_index] as usize,
                );
            }
        }
    }
}

/// Sparse spectrum-block upper bounds over coarse m/z bins.
///
/// This is a shared pruning-only index. Each m/z bin stores, per contiguous
/// spectrum-id block, the maximum metric-specific peak value observed in that
/// block. Query code supplies the metric-specific function that turns a query
/// peak and a stored block maximum into a safe score upper-bound contribution.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct SpectrumBlockUpperBoundIndex {
    bin_width: f64,
    min_bin: i64,
    bins: UpperBoundCsr,
    n_bins: usize,
    n_blocks: usize,
}

impl SpectrumBlockUpperBoundIndex {
    pub(crate) fn build<P, Values>(
        spectra: &[PreparedFlashSpectrum<P>],
        tolerance: f64,
        block_size: usize,
        mut peak_values: Values,
    ) -> Result<Self, SimilarityComputationError>
    where
        P: SpectrumFloat,
        Values:
            FnMut(u32, &PreparedFlashSpectrum<P>) -> Result<Vec<f64>, SimilarityComputationError>,
    {
        let block_size = block_size.max(1);
        let n_blocks = spectra.len().div_ceil(block_size);
        let n_blocks_u32 =
            u32::try_from(n_blocks).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let bin_width = if tolerance > 0.0 {
            2.0 * tolerance
        } else {
            1.0
        };

        let mut min_bin = i64::MAX;
        let mut max_bin = i64::MIN;
        for spectrum in spectra {
            for &mz in &spectrum.mz {
                let bin = Self::bin_id(mz.to_f64(), bin_width);
                min_bin = min_bin.min(bin);
                max_bin = max_bin.max(bin);
            }
        }

        if min_bin == i64::MAX {
            return Ok(Self {
                bin_width,
                min_bin: 0,
                bins: upper_bound_csr(0, n_blocks_u32, Vec::new())?,
                n_bins: 0,
                n_blocks,
            });
        }

        let n_bins = usize::try_from(max_bin - min_bin + 1)
            .map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let n_bins_u32 =
            u32::try_from(n_bins).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let total_peaks = spectra.iter().map(|spectrum| spectrum.mz.len()).sum();
        let mut entries: Vec<(usize, u32, f64)> = Vec::with_capacity(total_peaks);

        for (spec_id, spectrum) in spectra.iter().enumerate() {
            let spec_id_u32 =
                u32::try_from(spec_id).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            let values = peak_values(spec_id_u32, spectrum)?;
            debug_assert_eq!(values.len(), spectrum.mz.len());

            let block_id = u32::try_from(spec_id / block_size)
                .map_err(|_| SimilarityComputationError::IndexOverflow)?;
            for (&mz, &value) in spectrum.mz.iter().zip(values.iter()) {
                if !value.is_finite() {
                    return Err(SimilarityComputationError::NonFiniteValue(
                        "block_upper_bound",
                    ));
                }
                if value <= 0.0 {
                    continue;
                }

                let bin_index = usize::try_from(Self::bin_id(mz.to_f64(), bin_width) - min_bin)
                    .map_err(|_| SimilarityComputationError::IndexOverflow)?;
                entries.push((bin_index, block_id, value));
            }
        }

        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.total_cmp(&right.2))
        });

        let mut upper_bound_entries: Vec<(u32, u32, f64)> = Vec::with_capacity(entries.len());
        for (bin_index, block_id, value) in entries {
            let bin_index =
                u32::try_from(bin_index).map_err(|_| SimilarityComputationError::IndexOverflow)?;
            if let Some(last) = upper_bound_entries.last_mut()
                && last.0 == bin_index
                && last.1 == block_id
            {
                last.2 = f64::max(last.2, value);
                continue;
            }
            upper_bound_entries.push((bin_index, block_id, value));
        }

        Ok(Self {
            bin_width,
            min_bin,
            bins: upper_bound_csr(n_bins_u32, n_blocks_u32, upper_bound_entries)?,
            n_bins,
            n_blocks,
        })
    }

    pub(crate) fn prepare_allowed_blocks<Q, Contribution>(
        &self,
        query_mz: &[Q],
        tolerance: f64,
        minimum_bound: f64,
        state: &mut SearchState,
        mut contribution: Contribution,
    ) where
        Q: SpectrumFloat,
        Contribution: FnMut(usize, f64) -> f64,
    {
        state.prepare_spectrum_block_scratch(self.n_blocks);
        if self.n_blocks == 0 || self.n_bins == 0 {
            state.add_spectrum_block_filter_stats(self.n_blocks, 0);
            return;
        }

        for (query_index, &mz) in query_mz.iter().enumerate() {
            for bin_index in self.bin_indices_for_window(mz.to_f64(), tolerance) {
                let bin_index = bin_index as u32;
                let (block_ids, max_values) = self.bins.sparse_row_entries_slice(bin_index);
                for (&block_id, &max_value) in block_ids.iter().zip(max_values) {
                    state.add_spectrum_block_upper_bound(
                        block_id,
                        contribution(query_index, max_value),
                    );
                }
            }
        }

        let allowed_blocks = state.mark_allowed_spectrum_blocks(minimum_bound);
        state.add_spectrum_block_filter_stats(self.n_blocks, allowed_blocks);
    }

    fn bin_indices_for_window(&self, mz: f64, tolerance: f64) -> impl Iterator<Item = usize> + '_ {
        let lo_bin = Self::bin_id(mz - tolerance, self.bin_width);
        let hi_bin = Self::bin_id(mz + tolerance, self.bin_width);
        let start = lo_bin.max(self.min_bin);
        let end = hi_bin.min(self.min_bin + self.n_bins as i64 - 1);

        let (start, end) = if start <= end {
            (
                (start - self.min_bin) as usize,
                (end - self.min_bin) as usize,
            )
        } else {
            (1, 0)
        };

        start..=end
    }

    #[inline]
    fn bin_id(mz: f64, bin_width: f64) -> i64 {
        (mz / bin_width).floor() as i64
    }
}

// ---------------------------------------------------------------------------
// DenseAccumulator
// ---------------------------------------------------------------------------

/// Per-query score accumulator. Uses dense arrays (one slot per library
/// spectrum) with a `touched` list for efficient reset.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
struct DenseAccumulator {
    scores: Vec<f64>,
    counts: Vec<u32>,
    touched: Vec<u32>,
}

impl DenseAccumulator {
    fn new() -> Self {
        Self {
            scores: Vec::new(),
            counts: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, n_spectra: usize) {
        if self.scores.len() == n_spectra {
            return;
        }

        self.scores.clear();
        self.counts.clear();
        self.touched.clear();
        self.scores.resize(n_spectra, 0.0);
        self.counts.resize(n_spectra, 0);
    }

    #[inline]
    fn accumulate(&mut self, spec_id: u32, score: f64) {
        let idx = spec_id as usize;
        if self.counts[idx] == 0 {
            self.touched.push(spec_id);
        }
        self.scores[idx] += score;
        self.counts[idx] = self.counts[idx].saturating_add(1);
    }

    #[inline]
    fn touched_len(&self) -> usize {
        self.touched.len()
    }

    /// Replace a previously accumulated score with a better one, without
    /// changing the match count.
    #[inline]
    fn upgrade(&mut self, spec_id: u32, old_score: f64, new_score: f64) {
        self.scores[spec_id as usize] += new_score - old_score;
    }

    /// Drain the accumulator, calling `emit` for each spectrum that received
    /// at least one match. Resets all touched slots for reuse.
    fn drain(&mut self, mut emit: impl FnMut(u32, f64, u32)) {
        for &id in &self.touched {
            let idx = id as usize;
            emit(id, self.scores[idx], self.counts[idx]);
            self.scores[idx] = 0.0;
            self.counts[idx] = 0;
        }
        self.touched.clear();
    }
}

// ---------------------------------------------------------------------------
// SearchState — reusable per-query scratch space
// ---------------------------------------------------------------------------

/// Reusable scratch space for Flash index searches.
///
/// Create one `SearchState` per worker and pass it to repeated search calls to
/// reuse scratch buffers. Large buffers are allocated lazily by the search mode
/// that needs them, so direct threshold searches do not pay for modified-search
/// scratch.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub struct SearchState {
    acc: DenseAccumulator,
    matched_products: SearchBitVec,
    direct_scores: Vec<f64>,
    candidate_spectra: SearchBitVec,
    candidate_touched: Vec<u32>,
    block_upper_bounds: Vec<f64>,
    block_upper_bound_touched: Vec<u32>,
    allowed_spectrum_blocks: SearchBitVec,
    allowed_block_touched: Vec<u32>,
    query_order: Vec<usize>,
    query_suffix_bound: Vec<f64>,
    diagnostics: FlashSearchDiagnostics,
}

impl SearchState {
    /// Create a new lazily-sized `SearchState`.
    fn new() -> Self {
        Self {
            acc: DenseAccumulator::new(),
            matched_products: SearchBitVec::new(),
            direct_scores: Vec::new(),
            candidate_spectra: SearchBitVec::new(),
            candidate_touched: Vec::new(),
            block_upper_bounds: Vec::new(),
            block_upper_bound_touched: Vec::new(),
            allowed_spectrum_blocks: SearchBitVec::new(),
            allowed_block_touched: Vec::new(),
            query_order: Vec::new(),
            query_suffix_bound: Vec::new(),
            diagnostics: FlashSearchDiagnostics::default(),
        }
    }

    /// Return diagnostics for the most recent query executed with this state.
    #[inline]
    pub fn diagnostics(&self) -> FlashSearchDiagnostics {
        self.diagnostics
    }

    /// Reset query diagnostics.
    #[inline]
    pub fn reset_diagnostics(&mut self) {
        self.diagnostics = FlashSearchDiagnostics::default();
    }

    #[inline]
    pub(crate) fn add_product_postings_visited(&mut self, count: usize) {
        self.diagnostics.product_postings_visited = self
            .diagnostics
            .product_postings_visited
            .saturating_add(count);
    }

    #[inline]
    pub(crate) fn add_candidates_rescored(&mut self, count: usize) {
        self.diagnostics.candidates_rescored =
            self.diagnostics.candidates_rescored.saturating_add(count);
    }

    #[inline]
    pub(crate) fn add_candidates_marked(&mut self, count: usize) {
        self.diagnostics.candidates_marked =
            self.diagnostics.candidates_marked.saturating_add(count);
    }

    #[inline]
    pub(crate) fn add_results_emitted(&mut self, count: usize) {
        self.diagnostics.results_emitted = self.diagnostics.results_emitted.saturating_add(count);
    }

    /// Records how many spectrum blocks were kept by a query-level upper-bound filter.
    #[inline]
    pub(crate) fn add_spectrum_block_filter_stats(
        &mut self,
        blocks_evaluated: usize,
        blocks_allowed: usize,
    ) {
        self.diagnostics.spectrum_blocks_evaluated = self
            .diagnostics
            .spectrum_blocks_evaluated
            .saturating_add(blocks_evaluated);
        self.diagnostics.spectrum_blocks_allowed = self
            .diagnostics
            .spectrum_blocks_allowed
            .saturating_add(blocks_allowed);
        self.diagnostics.spectrum_blocks_pruned = self
            .diagnostics
            .spectrum_blocks_pruned
            .saturating_add(blocks_evaluated.saturating_sub(blocks_allowed));
    }

    pub(crate) fn ensure_candidate_capacity(&mut self, n_spectra: usize) {
        if self.candidate_spectra.len() == n_spectra {
            return;
        }

        self.candidate_spectra = SearchBitVec::zeros(n_spectra);
        self.candidate_touched.clear();
    }

    fn ensure_modified_capacity(&mut self, n_products: usize) {
        if self.matched_products.len() == n_products && self.direct_scores.len() == n_products {
            return;
        }

        self.matched_products = SearchBitVec::zeros(n_products);
        self.direct_scores.clear();
        self.direct_scores.resize(n_products, 0.0);
    }

    pub(crate) fn prepare_threshold_order<P: SpectrumFloat>(&mut self, query_data: &[P]) {
        self.query_order.clear();
        self.query_order.extend(0..query_data.len());
        self.query_order.sort_unstable_by(|&left, &right| {
            query_data[right]
                .to_f64()
                .abs()
                .total_cmp(&query_data[left].to_f64().abs())
                .then_with(|| left.cmp(&right))
        });

        self.query_suffix_bound.clear();
        self.query_suffix_bound.resize(query_data.len() + 1, 0.0);
        for order_index in (0..self.query_order.len()).rev() {
            let query_index = self.query_order[order_index];
            let query_value = query_data[query_index].to_f64();
            self.query_suffix_bound[order_index] =
                self.query_suffix_bound[order_index + 1] + query_value * query_value;
        }
        for value in &mut self.query_suffix_bound {
            *value = value.sqrt();
        }
    }

    pub(crate) fn prepare_additive_threshold_order<P: SpectrumFloat>(
        &mut self,
        query_data: &[P],
        mut upper_bound: impl FnMut(f64) -> f64,
    ) {
        self.query_order.clear();
        self.query_order.extend(0..query_data.len());
        self.query_order.sort_unstable_by(|&left, &right| {
            upper_bound(query_data[right].to_f64())
                .total_cmp(&upper_bound(query_data[left].to_f64()))
                .then_with(|| left.cmp(&right))
        });

        self.query_suffix_bound.clear();
        self.query_suffix_bound.resize(query_data.len() + 1, 0.0);
        for order_index in (0..self.query_order.len()).rev() {
            let query_index = self.query_order[order_index];
            self.query_suffix_bound[order_index] = self.query_suffix_bound[order_index + 1]
                + upper_bound(query_data[query_index].to_f64());
        }
    }

    pub(crate) fn threshold_prefix_len_by_target(&self, target: f64) -> usize {
        self.query_suffix_bound
            .iter()
            .position(|&remaining_bound| remaining_bound < target)
            .unwrap_or(self.query_order.len())
    }

    pub(crate) fn query_order(&self) -> &[usize] {
        &self.query_order
    }

    pub(crate) fn query_suffix_bound_at(&self, index: usize) -> f64 {
        self.query_suffix_bound[index]
    }

    #[inline]
    pub(crate) fn mark_candidate(&mut self, spec_id: u32) {
        let idx = spec_id as usize;
        if !self.candidate_spectra.get(idx) {
            self.candidate_spectra.set(idx, true);
            self.candidate_touched.push(spec_id);
            self.diagnostics.candidates_marked =
                self.diagnostics.candidates_marked.saturating_add(1);
        }
    }

    #[inline]
    pub(crate) fn is_candidate(&self, spec_id: u32) -> bool {
        self.candidate_spectra.get(spec_id as usize)
    }

    pub(crate) fn candidate_touched(&self) -> &[u32] {
        &self.candidate_touched
    }

    pub(crate) fn reset_candidates(&mut self) {
        for &spec_id in &self.candidate_touched {
            self.candidate_spectra.set(spec_id as usize, false);
        }
        self.candidate_touched.clear();
    }

    /// Ensures that block-pruning scratch space is sized for the current index.
    pub(crate) fn prepare_spectrum_block_scratch(&mut self, n_blocks: usize) {
        if self.block_upper_bounds.len() != n_blocks {
            self.block_upper_bounds.clear();
            self.block_upper_bounds.resize(n_blocks, 0.0);
            self.block_upper_bound_touched.clear();
            self.allowed_spectrum_blocks = SearchBitVec::zeros(n_blocks);
            self.allowed_block_touched.clear();
            return;
        }

        self.block_upper_bound_touched.clear();
        self.reset_allowed_spectrum_blocks();
    }

    /// Adds one query contribution to a block-level maximum possible score.
    #[inline]
    pub(crate) fn add_spectrum_block_upper_bound(&mut self, block_id: u32, contribution: f64) {
        if contribution <= 0.0 {
            return;
        }

        let block_index = block_id as usize;
        if self.block_upper_bounds[block_index] == 0.0 {
            self.block_upper_bound_touched.push(block_id);
        }
        self.block_upper_bounds[block_index] += contribution;
    }

    /// Marks blocks whose accumulated bound can still reach `minimum_score`.
    pub(crate) fn mark_allowed_spectrum_blocks(&mut self, minimum_score: f64) -> usize {
        const BOUND_EPSILON: f64 = 1e-12;

        let mut allowed_blocks = 0usize;
        for &block_id in &self.block_upper_bound_touched {
            let block_index = block_id as usize;
            if self.block_upper_bounds[block_index] + BOUND_EPSILON >= minimum_score
                && !self.allowed_spectrum_blocks.get(block_index)
            {
                self.allowed_spectrum_blocks.set(block_index, true);
                self.allowed_block_touched.push(block_id);
                allowed_blocks += 1;
            }
            self.block_upper_bounds[block_index] = 0.0;
        }
        self.block_upper_bound_touched.clear();
        allowed_blocks
    }

    /// Returns the number of spectrum blocks allowed by the current block filter.
    #[inline]
    pub(crate) fn n_allowed_spectrum_blocks(&self) -> usize {
        self.allowed_block_touched.len()
    }

    /// Returns the `index`th allowed spectrum block id.
    #[inline]
    pub(crate) fn allowed_spectrum_block_id(&self, index: usize) -> u32 {
        self.allowed_block_touched[index]
    }

    /// Returns whether `block_id` is currently allowed by block pruning.
    #[inline]
    pub(crate) fn is_allowed_spectrum_block(&self, block_id: u32) -> bool {
        let block_index = block_id as usize;
        block_index < self.allowed_spectrum_blocks.len()
            && self.allowed_spectrum_blocks.get(block_index)
    }

    /// Clears the current query's allowed block bitset.
    pub(crate) fn reset_allowed_spectrum_blocks(&mut self) {
        for &block_id in &self.allowed_block_touched {
            self.allowed_spectrum_blocks.set(block_id as usize, false);
        }
        self.allowed_block_touched.clear();
    }
}

// ---------------------------------------------------------------------------
// FlashIndex<K> — the inverted m/z index
// ---------------------------------------------------------------------------

/// Inverted m/z index shared by all Flash search variants.
#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
pub(crate) struct FlashIndex<K: FlashKernel, P: SpectrumFloat = f64> {
    // Product ion index (sorted by m/z).
    pub(crate) product_mz: Vec<P>,
    product_spec_id: Vec<u32>,
    pub(crate) product_data: Vec<P>,
    pepmass_index: Option<Pepmass2DPostingIndex>,

    // Product peaks in per-spectrum order for exact candidate rescoring.
    spectrum_offsets: Vec<u32>,
    spectrum_mz: Vec<P>,
    spectrum_data: Vec<P>,
    spectrum_precursor_mz: Vec<P>,

    // Neutral loss index (sorted by neutral loss value).
    nl_value: Vec<P>,
    nl_spec_id: Vec<u32>,
    nl_data: Vec<P>,
    /// Maps neutral-loss entry → product entry for anti-double-counting.
    nl_to_product: Vec<u32>,

    // Per-spectrum metadata.
    spectrum_meta: Vec<K::SpectrumMeta>,
    spectrum_id_map: SpectrumIdMap,
    pub(crate) n_spectra: u32,

    // Config.
    pub(crate) tolerance: f64,
    pepmass_filter: PepmassFilter,
}

impl<K: FlashKernel, P: SpectrumFloat + Sync> FlashIndex<K, P> {
    #[cfg(test)]
    pub(crate) fn build_with_spectrum_id_map(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
        spectrum_id_map: SpectrumIdMap,
    ) -> Result<Self, SimilarityComputationError> {
        let progress = NoopFlashIndexBuildProgress;
        Self::build_with_spectrum_id_map_and_progress(
            tolerance,
            spectra,
            spectrum_id_map,
            &progress,
        )
    }

    pub(crate) fn build_with_spectrum_id_map_and_progress<G>(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
        spectrum_id_map: SpectrumIdMap,
        progress: &G,
    ) -> Result<Self, SimilarityComputationError>
    where
        G: FlashIndexBuildProgress + ?Sized,
    {
        Self::build_with_sort(
            tolerance,
            spectra,
            spectrum_id_map,
            SortBackend::Sequential,
            progress,
        )
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn build_parallel_with_spectrum_id_map_and_progress<G>(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
        spectrum_id_map: SpectrumIdMap,
        progress: &G,
    ) -> Result<Self, SimilarityComputationError>
    where
        G: FlashIndexBuildProgress + Sync + ?Sized,
    {
        Self::build_with_sort(
            tolerance,
            spectra,
            spectrum_id_map,
            SortBackend::Parallel,
            progress,
        )
    }

    fn build_with_sort<G>(
        tolerance: f64,
        spectra: PreparedFlashSpectra<P>,
        spectrum_id_map: SpectrumIdMap,
        sort_backend: SortBackend,
        progress: &G,
    ) -> Result<Self, SimilarityComputationError>
    where
        G: FlashIndexBuildProgress + ?Sized,
    {
        let n_spectra: u32 =
            u32::try_from(spectra.len()).map_err(|_| SimilarityComputationError::IndexOverflow)?;
        let total_peaks: usize = spectra.iter().map(|spectrum| spectrum.mz.len()).sum();
        let _: u32 =
            u32::try_from(total_peaks).map_err(|_| SimilarityComputationError::IndexOverflow)?;

        let mut spectrum_meta_vec: Vec<K::SpectrumMeta> = Vec::with_capacity(spectra.len());

        // Collect sort-key indices. We sort a permutation array by m/z rather
        // than sorting the full (u32, PeakEntry) tuples to reduce sort
        // bandwidth.
        let mut peak_mz_flat: Vec<P> = Vec::with_capacity(total_peaks);
        let mut peak_spec_id_flat: Vec<u32> = Vec::with_capacity(total_peaks);
        let mut peak_data_flat: Vec<P> = Vec::with_capacity(total_peaks);
        let mut peak_nl_flat: Vec<P> = Vec::with_capacity(total_peaks);
        let mut spectrum_offsets: Vec<u32> = Vec::with_capacity(spectra.len() + 1);
        let mut spectrum_mz: Vec<P> = Vec::with_capacity(total_peaks);
        let mut spectrum_data: Vec<P> = Vec::with_capacity(total_peaks);
        let mut spectrum_precursor_mz: Vec<P> = Vec::with_capacity(spectra.len());

        progress.start_phase(
            FlashIndexBuildPhase::PackFlashPeaks,
            Some(u64::from(n_spectra)),
        );
        for (spec_id, spectrum) in spectra.iter().enumerate() {
            let spec_id = spec_id as u32; // safe: checked above
            spectrum_meta_vec.push(K::spectrum_meta(&spectrum.data));
            spectrum_precursor_mz.push(spectrum.precursor_mz);
            spectrum_offsets.push(spectrum_mz.len() as u32);

            for (&mz, &data) in spectrum.mz.iter().zip(spectrum.data.iter()) {
                peak_mz_flat.push(mz);
                peak_spec_id_flat.push(spec_id);
                peak_data_flat.push(data);
                peak_nl_flat.push(convert_flash_value(
                    spectrum.precursor_mz.to_f64() - mz.to_f64(),
                    "neutral_loss",
                )?);
                spectrum_mz.push(mz);
                spectrum_data.push(data);
            }
            progress.inc(1);
        }
        spectrum_offsets.push(spectrum_mz.len() as u32);

        // Build a permutation array and sort it by m/z.
        progress.start_phase(FlashIndexBuildPhase::SortProductIndex, Some(1));
        let mut product_perm: Vec<u32> = (0..total_peaks as u32).collect();
        sort_permutation_by_values(&mut product_perm, &peak_mz_flat, sort_backend);
        progress.inc(1);

        // Build old insertion index → new sorted index mapping.
        let mut old_to_new = alloc::vec![0u32; total_peaks];
        for (new_idx, &old_idx) in product_perm.iter().enumerate() {
            old_to_new[old_idx as usize] = new_idx as u32;
        }

        // Scatter into sorted product arrays.
        let mut product_mz = Vec::with_capacity(total_peaks);
        let mut product_spec_id = Vec::with_capacity(total_peaks);
        let mut product_data = Vec::with_capacity(total_peaks);
        for &old_idx in &product_perm {
            let i = old_idx as usize;
            product_mz.push(peak_mz_flat[i]);
            product_spec_id.push(peak_spec_id_flat[i]);
            product_data.push(peak_data_flat[i]);
        }

        // Build a permutation for NL entries sorted by neutral loss value.
        progress.start_phase(FlashIndexBuildPhase::SortNeutralLossIndex, Some(1));
        let mut nl_perm: Vec<u32> = (0..total_peaks as u32).collect();
        sort_permutation_by_values(&mut nl_perm, &peak_nl_flat, sort_backend);
        progress.inc(1);

        // Scatter into sorted NL arrays, remapping product index.
        let mut nl_value = Vec::with_capacity(total_peaks);
        let mut nl_spec_id = Vec::with_capacity(total_peaks);
        let mut nl_data = Vec::with_capacity(total_peaks);
        let mut nl_to_product = Vec::with_capacity(total_peaks);
        for &old_idx in &nl_perm {
            let i = old_idx as usize;
            nl_value.push(peak_nl_flat[i]);
            nl_spec_id.push(peak_spec_id_flat[i]);
            nl_data.push(peak_data_flat[i]);
            nl_to_product.push(old_to_new[i]);
        }

        Ok(FlashIndex {
            product_mz,
            product_spec_id,
            product_data,
            pepmass_index: None,
            spectrum_offsets,
            spectrum_mz,
            spectrum_data,
            spectrum_precursor_mz,
            nl_value,
            nl_spec_id,
            nl_data,
            nl_to_product,
            spectrum_meta: spectrum_meta_vec,
            spectrum_id_map,
            n_spectra,
            tolerance,
            pepmass_filter: PepmassFilter::disabled(),
        })
    }

    /// Create a [`SearchState`] sized for this index, suitable for reuse
    /// across multiple queries.
    pub(crate) fn new_search_state(&self) -> SearchState {
        SearchState::new()
    }

    pub(crate) fn spectrum_slices(&self, spec_id: u32) -> (&[P], &[P]) {
        let offset_start = self.spectrum_offsets[spec_id as usize] as usize;
        let offset_end = self.spectrum_offsets[spec_id as usize + 1] as usize;
        (
            &self.spectrum_mz[offset_start..offset_end],
            &self.spectrum_data[offset_start..offset_end],
        )
    }

    pub(crate) fn spectrum_meta(&self, spec_id: u32) -> &K::SpectrumMeta {
        &self.spectrum_meta[spec_id as usize]
    }

    #[inline]
    pub(crate) fn spectrum_precursor_mz(&self, spec_id: u32) -> f64 {
        self.spectrum_precursor_mz[spec_id as usize].to_f64()
    }

    #[inline]
    pub(crate) fn pepmass_filter(&self) -> PepmassFilter {
        self.pepmass_filter
    }

    pub(crate) fn set_pepmass_filter_with_progress<G>(
        &mut self,
        pepmass_filter: PepmassFilter,
        progress: &G,
    ) -> Result<(), SimilarityComputationError>
    where
        G: FlashIndexBuildProgress + ?Sized,
    {
        if pepmass_filter.is_enabled() {
            self.ensure_pepmass_index(pepmass_filter, progress)?;
            self.pepmass_filter = pepmass_filter;
        } else {
            self.clear_pepmass_filter();
        }
        Ok(())
    }

    pub(crate) fn clear_pepmass_filter(&mut self) {
        self.pepmass_index = None;
        self.pepmass_filter = PepmassFilter::disabled();
    }

    fn ensure_pepmass_index<G>(
        &mut self,
        pepmass_filter: PepmassFilter,
        progress: &G,
    ) -> Result<bool, SimilarityComputationError>
    where
        G: FlashIndexBuildProgress + ?Sized,
    {
        if self
            .pepmass_index
            .as_ref()
            .is_some_and(|index| index.matches_filter(pepmass_filter))
        {
            return Ok(false);
        }

        self.pepmass_index = Some(Pepmass2DPostingIndex::build(
            &self.spectrum_precursor_mz,
            &self.product_spec_id,
            &self.nl_spec_id,
            pepmass_filter,
            progress,
        )?);
        Ok(true)
    }

    pub(crate) fn query_precursor_mz_for_filter<S>(
        &self,
        query: &S,
    ) -> Result<Option<f64>, SimilarityComputationError>
    where
        S: Spectrum,
    {
        if !self.pepmass_filter.is_enabled() {
            return Ok(None);
        }
        let query_precursor_mz = query.precursor_mz().to_f64();
        if !query_precursor_mz.is_finite() {
            return Err(SimilarityComputationError::NonFiniteValue(
                "query_precursor_mz",
            ));
        }
        Ok(Some(query_precursor_mz))
    }

    #[inline]
    fn allows_precursor(&self, query_precursor_mz: Option<f64>, spec_id: u32) -> bool {
        if !self.pepmass_filter.is_enabled() {
            return true;
        }
        self.pepmass_filter
            .allows(query_precursor_mz, self.spectrum_precursor_mz(spec_id))
    }

    #[inline]
    pub(crate) fn public_spectrum_id(&self, internal_id: u32) -> u32 {
        self.spectrum_id_map.internal_to_public(internal_id)
    }

    #[inline]
    pub(crate) fn internal_spectrum_id(
        &self,
        public_id: u32,
    ) -> Result<u32, SimilarityComputationError> {
        self.spectrum_id_map
            .public_to_internal(public_id)
            .filter(|&internal_id| internal_id < self.n_spectra)
            .ok_or(SimilarityComputationError::IndexOverflow)
    }

    fn pepmass_row_range(&self, query_precursor_mz: Option<f64>) -> Range<usize> {
        self.pepmass_index
            .as_ref()
            .map(|pepmass_index| {
                pepmass_index.matching_row_range(self.pepmass_filter, query_precursor_mz)
            })
            .unwrap_or(0..0)
    }

    fn for_each_product_peak_in_window<Q: SpectrumFloat>(
        &self,
        mz: Q,
        query_precursor_mz: Option<f64>,
        mut emit: impl FnMut(usize, u32, P),
    ) -> usize {
        if self.pepmass_filter.is_enabled() {
            let Some(pepmass_index) = self.pepmass_index.as_ref() else {
                debug_assert!(false, "PEPMASS filter enabled without PEPMASS index");
                return 0;
            };
            let rows = self.pepmass_row_range(query_precursor_mz);
            return pepmass_index.for_each_product_peak_in_window(
                rows,
                mz,
                self.tolerance,
                ProductPostingSlices {
                    mz: &self.product_mz,
                    spec_id: &self.product_spec_id,
                    data: &self.product_data,
                },
                |product_index, spec_id, product_data| {
                    if self.allows_precursor(query_precursor_mz, spec_id) {
                        emit(product_index, spec_id, product_data);
                    }
                },
            );
        }

        let mz = mz.to_f64();
        let lo = mz - self.tolerance;
        let hi = mz + self.tolerance;
        let start = self.product_mz.partition_point(|&v| v.to_f64() < lo);

        let mut visited = 0usize;
        for idx in start..self.product_mz.len() {
            let product_mz = self.product_mz[idx].to_f64();
            if product_mz > hi {
                break;
            }
            visited += 1;
            let spec_id = self.product_spec_id[idx];
            if self.allows_precursor(query_precursor_mz, spec_id) {
                emit(idx, spec_id, self.product_data[idx]);
            }
        }
        visited
    }

    pub(crate) fn for_each_product_spectrum_in_window<Q: SpectrumFloat>(
        &self,
        mz: Q,
        query_precursor_mz: Option<f64>,
        mut emit: impl FnMut(u32),
    ) -> usize {
        self.for_each_product_peak_in_window(mz, query_precursor_mz, |_, spec_id, _| {
            emit(spec_id);
        })
    }

    pub(crate) fn mark_candidates_from_query_order_prefix(
        &self,
        query_mz: &[impl SpectrumFloat],
        prefix_len: usize,
        query_precursor_mz: Option<f64>,
        state: &mut SearchState,
    ) {
        state.ensure_candidate_capacity(self.n_spectra as usize);
        let prefix_len = prefix_len.min(state.query_order.len());
        for order_position in 0..prefix_len {
            let query_index = state.query_order[order_position];
            let visited = self.for_each_product_spectrum_in_window(
                query_mz[query_index],
                query_precursor_mz,
                |spec_id| {
                    state.mark_candidate(spec_id);
                },
            );
            state.add_product_postings_visited(visited);
        }
    }

    pub(crate) fn for_each_allowed_block_raw_score<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_precursor_mz: Option<f64>,
        block_products: &SpectrumBlockProductIndex<P>,
        state: &mut SearchState,
        mut emit: impl FnMut(u32, f64, usize),
    ) {
        state.acc.ensure_capacity(self.n_spectra as usize);
        if state.n_allowed_spectrum_blocks() == 0 {
            state.reset_allowed_spectrum_blocks();
            return;
        }

        let mut product_postings_visited = 0usize;
        if self.pepmass_filter.is_enabled() {
            for (query_index, &mz) in query_mz.iter().enumerate() {
                product_postings_visited =
                    product_postings_visited.saturating_add(self.for_each_product_peak_in_window(
                        mz,
                        query_precursor_mz,
                        |_, spec_id, library_data| {
                            let block_id = block_products.spectrum_block_id(spec_id);
                            if !state.is_allowed_spectrum_block(block_id) {
                                return;
                            }
                            let score = K::pair_score(
                                query_data[query_index].to_f64(),
                                library_data.to_f64(),
                            );
                            if score != 0.0 {
                                state.acc.accumulate(spec_id, score);
                            }
                        },
                    ));
            }

            state.add_product_postings_visited(product_postings_visited);
            state.add_candidates_marked(state.acc.touched_len());
            state.acc.drain(|spec_id, raw, count| {
                emit(spec_id, raw, count as usize);
            });
            state.reset_allowed_spectrum_blocks();
            return;
        }

        for (query_index, &mz) in query_mz.iter().enumerate() {
            for block_index in 0..state.n_allowed_spectrum_blocks() {
                let block_id = state.allowed_spectrum_block_id(block_index);
                product_postings_visited = product_postings_visited.saturating_add(
                    block_products.for_each_peak_in_window(
                        block_id,
                        mz,
                        self.tolerance,
                        |spec_id, library_data| {
                            let score = K::pair_score(
                                query_data[query_index].to_f64(),
                                library_data.to_f64(),
                            );
                            if score != 0.0 {
                                state.acc.accumulate(spec_id, score);
                            }
                        },
                    ),
                );
            }
        }

        state.add_product_postings_visited(product_postings_visited);
        state.add_candidates_marked(state.acc.touched_len());
        state.acc.drain(|spec_id, raw, count| {
            emit(spec_id, raw, count as usize);
        });
        state.reset_allowed_spectrum_blocks();
    }

    /// Visit candidates from the currently allowed spectrum blocks in query
    /// suffix-bound order.
    ///
    /// The caller must prepare `state.query_order` and `state.query_suffix_bound`
    /// in the metric's bound units before calling this method. Each spectrum is
    /// passed to `visit_candidate` at most once, where the caller can exact-score
    /// it and return the updated suffix stop bound derived from the current top-k
    /// floor. Once the remaining query suffix falls below that bound, unseen
    /// spectra cannot enter the top-k set and the block scan stops.
    pub(crate) fn score_allowed_block_candidates_by_query_order<Q, Visit>(
        &self,
        query_mz: &[Q],
        query_precursor_mz: Option<f64>,
        block_products: &SpectrumBlockProductIndex<P>,
        state: &mut SearchState,
        mut suffix_stop_bound: f64,
        mut visit_candidate: Visit,
    ) where
        Q: SpectrumFloat,
        Visit: FnMut(u32) -> f64,
    {
        state.ensure_candidate_capacity(self.n_spectra as usize);
        if state.n_allowed_spectrum_blocks() == 0 {
            state.reset_allowed_spectrum_blocks();
            return;
        }

        let mut product_postings_visited = 0usize;
        let mut candidates_rescored = 0usize;
        let query_order_len = state.query_order().len();
        if self.pepmass_filter.is_enabled() {
            for order_position in 0..query_order_len {
                let query_index = state.query_order()[order_position];
                let query_mz = query_mz[query_index];
                product_postings_visited =
                    product_postings_visited.saturating_add(self.for_each_product_peak_in_window(
                        query_mz,
                        query_precursor_mz,
                        |_, spec_id, _| {
                            let block_id = block_products.spectrum_block_id(spec_id);
                            if !state.is_allowed_spectrum_block(block_id) {
                                return;
                            }
                            if state.is_candidate(spec_id) {
                                return;
                            }
                            state.mark_candidate(spec_id);
                            candidates_rescored = candidates_rescored.saturating_add(1);
                            suffix_stop_bound = visit_candidate(spec_id);
                        },
                    ));

                if state.query_suffix_bound_at(order_position + 1) < suffix_stop_bound {
                    break;
                }
            }

            state.add_product_postings_visited(product_postings_visited);
            state.add_candidates_rescored(candidates_rescored);
            state.reset_candidates();
            state.reset_allowed_spectrum_blocks();
            return;
        }

        for order_position in 0..query_order_len {
            let query_index = state.query_order()[order_position];
            let query_mz = query_mz[query_index];

            for block_index in 0..state.n_allowed_spectrum_blocks() {
                let block_id = state.allowed_spectrum_block_id(block_index);
                product_postings_visited = product_postings_visited.saturating_add(
                    block_products.for_each_peak_in_window(
                        block_id,
                        query_mz,
                        self.tolerance,
                        |spec_id, _| {
                            if state.is_candidate(spec_id) {
                                return;
                            }
                            state.mark_candidate(spec_id);
                            candidates_rescored = candidates_rescored.saturating_add(1);
                            suffix_stop_bound = visit_candidate(spec_id);
                        },
                    ),
                );
            }

            if state.query_suffix_bound_at(order_position + 1) < suffix_stop_bound {
                break;
            }
        }

        state.add_product_postings_visited(product_postings_visited);
        state.add_candidates_rescored(candidates_rescored);
        state.reset_candidates();
        state.reset_allowed_spectrum_blocks();
    }

    /// Direct (unshifted) search: for each query peak, binary-search the
    /// product index and accumulate scores.
    pub(crate) fn search_direct<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: Option<f64>,
    ) -> Vec<FlashSearchResult> {
        let mut state = self.new_search_state();
        self.search_direct_with_state(
            query_mz,
            query_data,
            query_meta,
            query_precursor_mz,
            &mut state,
        )
    }

    /// Direct search using a caller-provided [`SearchState`] to avoid
    /// per-query allocation.
    pub(crate) fn search_direct_with_state<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: Option<f64>,
        state: &mut SearchState,
    ) -> Vec<FlashSearchResult> {
        let mut results = Vec::new();
        self.for_each_direct_with_state(
            query_mz,
            query_data,
            query_meta,
            query_precursor_mz,
            state,
            |result| {
                results.push(result);
            },
        );
        results
    }

    pub(crate) fn for_each_direct_with_state<Q: SpectrumFloat, Emit>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: Option<f64>,
        state: &mut SearchState,
        mut emit: Emit,
    ) where
        Emit: FnMut(FlashSearchResult),
    {
        state.reset_diagnostics();
        if self.n_spectra == 0 || query_mz.is_empty() {
            return;
        }

        state.acc.ensure_capacity(self.n_spectra as usize);
        let mut product_postings_visited = 0usize;

        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            product_postings_visited =
                product_postings_visited.saturating_add(self.for_each_product_peak_in_window(
                    qmz,
                    query_precursor_mz,
                    |_, spec_id, library_data| {
                        let score =
                            K::pair_score(query_data[q_idx].to_f64(), library_data.to_f64());
                        state.acc.accumulate(spec_id, score);
                    },
                ));
        }

        state.add_product_postings_visited(product_postings_visited);
        let mut results_emitted = 0usize;
        state.acc.drain(|spec_id, raw, count| {
            if !self.allows_precursor(query_precursor_mz, spec_id) {
                return;
            }
            let score = K::finalize(
                raw,
                count as usize,
                query_meta,
                &self.spectrum_meta[spec_id as usize],
            );
            if score > 0.0 {
                results_emitted = results_emitted.saturating_add(1);
                emit(FlashSearchResult {
                    spectrum_id: self.public_spectrum_id(spec_id),
                    score,
                    n_matches: count as usize,
                });
            }
        });
        state.add_results_emitted(results_emitted);
    }

    pub(crate) fn direct_score_for_spectrum<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        spec_id: u32,
    ) -> (f64, usize) {
        let offset_start = self.spectrum_offsets[spec_id as usize] as usize;
        let offset_end = self.spectrum_offsets[spec_id as usize + 1] as usize;
        let library_mz = &self.spectrum_mz[offset_start..offset_end];
        let library_data = &self.spectrum_data[offset_start..offset_end];

        let mut raw = 0.0_f64;
        let mut n_matches = 0usize;
        let mut library_index = 0usize;

        for (query_index, &qmz) in query_mz.iter().enumerate() {
            let qmz = qmz.to_f64();
            while library_index < library_mz.len()
                && library_mz[library_index].to_f64() < qmz - self.tolerance
            {
                library_index += 1;
            }
            if library_index < library_mz.len()
                && library_mz[library_index].to_f64() >= qmz - self.tolerance
                && library_mz[library_index].to_f64() <= qmz + self.tolerance
            {
                let score = K::pair_score(
                    query_data[query_index].to_f64(),
                    library_data[library_index].to_f64(),
                );
                if score != 0.0 {
                    raw += score;
                    n_matches += 1;
                }
                library_index += 1;
            }
        }

        (raw, n_matches)
    }

    fn for_each_neutral_loss_peak_in_window(
        &self,
        neutral_loss: f64,
        query_precursor_mz: Option<f64>,
        mut emit: impl FnMut(usize, u32, P, usize),
    ) {
        if self.pepmass_filter.is_enabled() {
            let Some(pepmass_index) = self.pepmass_index.as_ref() else {
                debug_assert!(false, "PEPMASS filter enabled without PEPMASS index");
                return;
            };
            let rows = self.pepmass_row_range(query_precursor_mz);
            pepmass_index.for_each_neutral_loss_in_window(
                rows,
                neutral_loss,
                self.tolerance,
                NeutralLossPostingSlices {
                    value: &self.nl_value,
                    spec_id: &self.nl_spec_id,
                    data: &self.nl_data,
                    to_product: &self.nl_to_product,
                },
                |neutral_loss_index, spec_id, data, product_index| {
                    if self.allows_precursor(query_precursor_mz, spec_id) {
                        emit(neutral_loss_index, spec_id, data, product_index);
                    }
                },
            );
            return;
        }

        let lo = neutral_loss - self.tolerance;
        let hi = neutral_loss + self.tolerance;
        let start = self.nl_value.partition_point(|&v| v.to_f64() < lo);

        for idx in start..self.nl_value.len() {
            let nl_value = self.nl_value[idx].to_f64();
            if nl_value > hi {
                break;
            }
            if nl_value < neutral_loss - self.tolerance || nl_value > neutral_loss + self.tolerance
            {
                continue;
            }
            let spec_id = self.nl_spec_id[idx];
            if self.allows_precursor(query_precursor_mz, spec_id) {
                emit(
                    idx,
                    spec_id,
                    self.nl_data[idx],
                    self.nl_to_product[idx] as usize,
                );
            }
        }
    }

    pub(crate) fn emit_exact_primary_candidates<Q, Emit, TargetRaw, LibraryBound>(
        &self,
        search: DirectThresholdSearch<'_, K, Q>,
        state: &mut SearchState,
        emit: &mut Emit,
        target_raw_score: &mut TargetRaw,
        library_bound: &mut LibraryBound,
    ) where
        Q: SpectrumFloat,
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        let mut candidates_rescored = 0usize;
        let mut results_emitted = 0usize;
        for &spec_id in state.candidate_touched() {
            candidates_rescored = candidates_rescored.saturating_add(1);
            if !self.allows_precursor(search.query_precursor_mz, spec_id) {
                continue;
            }
            if self.emit_exact_threshold_candidate(
                &search,
                spec_id,
                emit,
                target_raw_score,
                library_bound,
            ) {
                results_emitted = results_emitted.saturating_add(1);
            }
        }

        state.add_candidates_rescored(candidates_rescored);
        state.add_results_emitted(results_emitted);
        state.reset_candidates();
    }

    fn emit_exact_threshold_candidate<Q, Emit, TargetRaw, LibraryBound>(
        &self,
        search: &DirectThresholdSearch<'_, K, Q>,
        spec_id: u32,
        emit: &mut Emit,
        target_raw_score: &mut TargetRaw,
        library_bound: &mut LibraryBound,
    ) -> bool
    where
        Q: SpectrumFloat,
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        let lib_meta = &self.spectrum_meta[spec_id as usize];
        if !self.allows_precursor(search.query_precursor_mz, spec_id) {
            return false;
        }
        if library_bound(lib_meta) == 0.0 {
            return false;
        }

        let (raw, count) =
            self.direct_score_for_spectrum(search.query_mz, search.query_data, spec_id);
        if raw < target_raw_score(lib_meta) {
            return false;
        }

        let score = K::finalize(raw, count, search.query_meta, lib_meta);
        if score > 0.0 && score >= search.score_threshold {
            emit(FlashSearchResult {
                spectrum_id: self.public_spectrum_id(spec_id),
                score,
                n_matches: count,
            });
            return true;
        }
        false
    }

    pub(crate) fn for_each_direct_threshold_with_state<Q, Emit, TargetRaw, LibraryBound>(
        &self,
        search: DirectThresholdSearch<'_, K, Q>,
        state: &mut SearchState,
        mut emit: Emit,
        mut target_raw_score: TargetRaw,
        mut library_bound: LibraryBound,
    ) where
        Q: SpectrumFloat,
        Emit: FnMut(FlashSearchResult),
        TargetRaw: FnMut(&K::SpectrumMeta) -> f64,
        LibraryBound: FnMut(&K::SpectrumMeta) -> f64,
    {
        state.reset_diagnostics();
        if self.n_spectra == 0 || search.query_mz.is_empty() {
            return;
        }

        if search.score_threshold < PREFIX_PRUNING_MIN_THRESHOLD {
            self.for_each_direct_with_state(
                search.query_mz,
                search.query_data,
                search.query_meta,
                search.query_precursor_mz,
                state,
                |result| {
                    if result.score >= search.score_threshold {
                        emit(result);
                    }
                },
            );
            return;
        }

        state.prepare_threshold_order(search.query_data);
        let target_query_norm = state.query_suffix_bound[0] * search.score_threshold;
        let prefix_len = state
            .query_suffix_bound
            .iter()
            .position(|&remaining_norm| remaining_norm < target_query_norm)
            .unwrap_or(state.query_order.len());

        if prefix_len * 2 > state.query_order.len() {
            self.for_each_direct_with_state(
                search.query_mz,
                search.query_data,
                search.query_meta,
                search.query_precursor_mz,
                state,
                |result| {
                    if result.score >= search.score_threshold {
                        emit(result);
                    }
                },
            );
            return;
        }

        self.mark_candidates_from_query_order_prefix(
            search.query_mz,
            prefix_len,
            search.query_precursor_mz,
            state,
        );
        self.emit_exact_primary_candidates(
            search,
            state,
            &mut emit,
            &mut target_raw_score,
            &mut library_bound,
        );
    }

    /// Modified (direct + shifted) search with anti-double-counting.
    ///
    /// Phase 1: direct matches (same as direct search), marking matched
    /// product entries in a bit set.
    ///
    /// Phase 2: for each query peak, compute `neutral_loss = query_precursor -
    /// query_mz`, binary-search the neutral-loss index, skip entries whose
    /// product counterpart was already matched in phase 1, accumulate the rest.
    pub(crate) fn search_modified<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: f64,
    ) -> Vec<FlashSearchResult> {
        let mut state = self.new_search_state();
        self.search_modified_with_state(
            query_mz,
            query_data,
            query_meta,
            query_precursor_mz,
            &mut state,
        )
    }

    /// Modified search using a caller-provided [`SearchState`] to avoid
    /// per-query allocation.
    pub(crate) fn search_modified_with_state<Q: SpectrumFloat>(
        &self,
        query_mz: &[Q],
        query_data: &[Q],
        query_meta: &K::SpectrumMeta,
        query_precursor_mz: f64,
        state: &mut SearchState,
    ) -> Vec<FlashSearchResult> {
        state.reset_diagnostics();
        if self.n_spectra == 0 || query_mz.is_empty() {
            return Vec::new();
        }

        state.acc.ensure_capacity(self.n_spectra as usize);
        state.ensure_modified_capacity(self.product_mz.len());

        // Destructure state to avoid borrow conflicts between fields.
        let SearchState {
            acc,
            matched_products,
            direct_scores,
            ..
        } = state;

        // Track which product indices we set so we can reset them efficiently.
        let mut set_indices: Vec<usize> = Vec::new();

        // Phase 1: direct matches.
        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            self.for_each_product_peak_in_window(
                qmz,
                Some(query_precursor_mz),
                |product_index, spec_id, library_data| {
                    let score = K::pair_score(query_data[q_idx].to_f64(), library_data.to_f64());
                    acc.accumulate(spec_id, score);
                    matched_products.set(product_index, true);
                    direct_scores[product_index] = score;
                    set_indices.push(product_index);
                },
            );
        }

        // Phase 2: shifted (neutral loss) matches.
        for (q_idx, &qmz) in query_mz.iter().enumerate() {
            let query_nl = query_precursor_mz - qmz.to_f64();
            self.for_each_neutral_loss_peak_in_window(
                query_nl,
                Some(query_precursor_mz),
                |_, spec_id, library_data, product_idx| {
                    let nl_score = K::pair_score(query_data[q_idx].to_f64(), library_data.to_f64());
                    if matched_products.get(product_idx) {
                        // Library peak already matched directly. Upgrade if NL
                        // gives a better pair score.
                        if nl_score > direct_scores[product_idx] {
                            acc.upgrade(spec_id, direct_scores[product_idx], nl_score);
                        }
                        return;
                    }
                    acc.accumulate(spec_id, nl_score);
                },
            );
        }

        // Reset the matched-product bitset and direct scores for reuse.
        for &idx in &set_indices {
            matched_products.set(idx, false);
            direct_scores[idx] = 0.0;
        }

        let mut results = Vec::with_capacity(acc.touched.len());
        acc.drain(|spec_id, raw, count| {
            let score = K::finalize(
                raw,
                count as usize,
                query_meta,
                &self.spectrum_meta[spec_id as usize],
            );
            if score > 0.0 {
                results.push(FlashSearchResult {
                    spectrum_id: self.public_spectrum_id(spec_id),
                    score,
                    n_matches: count as usize,
                });
            }
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;

    #[derive(Clone, Copy, Default)]
    struct TestKernel;

    impl FlashKernel for TestKernel {
        type SpectrumMeta = ();

        fn spectrum_meta<P: SpectrumFloat>(_peak_data: &[P]) -> Self::SpectrumMeta {}

        fn pair_score(query: f64, library: f64) -> f64 {
            query * library
        }

        fn finalize(
            raw: f64,
            _n_matches: usize,
            _query_meta: &Self::SpectrumMeta,
            _lib_meta: &Self::SpectrumMeta,
        ) -> f64 {
            raw
        }
    }

    fn prepared(precursor_mz: f64, mz: Vec<f64>, data: Vec<f64>) -> PreparedFlashSpectrum<f64> {
        PreparedFlashSpectrum {
            precursor_mz,
            mz,
            data,
        }
    }

    fn build_test_index(
        spectra: PreparedFlashSpectra<f64>,
        tolerance: f64,
    ) -> FlashIndex<TestKernel> {
        FlashIndex::<TestKernel>::build_with_spectrum_id_map(
            tolerance,
            spectra,
            SpectrumIdMap::identity(),
        )
        .expect("test index should build")
    }

    #[test]
    fn dense_accumulator_drains_and_resets_touched_slots() {
        let mut acc = DenseAccumulator::new();
        acc.ensure_capacity(4);
        acc.accumulate(2, 1.5);
        acc.accumulate(2, 0.5);
        acc.accumulate(1, 2.0);

        let mut emitted = Vec::new();
        acc.drain(|id, score, count| emitted.push((id, score, count)));

        emitted.sort_by_key(|&(id, _, _)| id);
        assert_eq!(emitted, vec![(1, 2.0, 1), (2, 2.0, 2)]);
        assert!(acc.touched.is_empty());
        assert_eq!(acc.scores, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(acc.counts, vec![0, 0, 0, 0]);

        let mut second = Vec::new();
        acc.drain(|id, score, count| second.push((id, score, count)));
        assert!(second.is_empty());
    }

    #[test]
    fn dense_accumulator_upgrade_replaces_score_without_changing_count() {
        let mut acc = DenseAccumulator::new();
        acc.ensure_capacity(2);
        acc.accumulate(0, 1.0);
        acc.upgrade(0, 1.0, 3.5);

        let mut emitted = Vec::new();
        acc.drain(|id, score, count| emitted.push((id, score, count)));
        assert_eq!(emitted, vec![(0, 3.5, 1)]);
    }

    #[test]
    fn build_sorts_products_and_lazily_initializes_search_state() {
        let index = build_test_index(
            vec![
                prepared(210.0, vec![110.0, 100.0], vec![4.0, 1.0]),
                prepared(205.0, vec![90.0], vec![2.0]),
            ],
            0.1,
        );

        assert_eq!(index.product_mz, vec![90.0, 100.0, 110.0]);
        assert_eq!(index.product_data, vec![2.0, 1.0, 4.0]);
        assert_eq!(index.n_spectra, 2);
        assert!(index.pepmass_index.is_none());

        let state = index.new_search_state();
        assert!(state.matched_products.is_empty());
        assert!(state.direct_scores.is_empty());
        assert!(state.acc.scores.is_empty());

        let mut direct_state = index.new_search_state();
        let _ = index.search_direct_with_state(&[100.0], &[1.0], &(), None, &mut direct_state);
        assert_eq!(direct_state.acc.scores.len(), 2);
        assert!(direct_state.matched_products.is_empty());
        assert!(direct_state.direct_scores.is_empty());

        let mut modified_state = index.new_search_state();
        let _ = index.search_modified_with_state(&[100.0], &[1.0], &(), 200.0, &mut modified_state);
        assert_eq!(modified_state.acc.scores.len(), 2);
        assert_eq!(modified_state.matched_products.len(), 3);
        assert_eq!(modified_state.direct_scores.len(), 3);
    }

    #[test]
    fn pepmass_2d_index_is_lazy_and_dropped_when_disabled() {
        let mut index = build_test_index(
            vec![
                prepared(200.0, vec![100.0], vec![1.0]),
                prepared(500.0, vec![100.0], vec![1.0]),
            ],
            0.1,
        );

        assert!(index.pepmass_index.is_none());
        let progress = NoopFlashIndexBuildProgress;
        index
            .set_pepmass_filter_with_progress(
                PepmassFilter::within_tolerance(0.5).unwrap(),
                &progress,
            )
            .expect("PEPMASS 2D index should build");
        assert!(index.pepmass_index.is_some());

        let results = index.search_direct(&[100.0], &[1.0], &(), Some(200.0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].spectrum_id, 0);

        index
            .set_pepmass_filter_with_progress(PepmassFilter::disabled(), &progress)
            .expect("PEPMASS filter should be disabled");
        assert!(index.pepmass_index.is_none());
    }

    #[test]
    fn spectrum_block_upper_bound_index_filters_candidate_blocks() {
        let spectra = vec![
            prepared(500.0, vec![100.0], vec![1.0]),
            prepared(500.0, vec![100.0], vec![0.8]),
            prepared(500.0, vec![100.0], vec![0.1]),
        ];
        let block_index = SpectrumBlockUpperBoundIndex::build(&spectra, 0.1, 2, |_, spectrum| {
            Ok(spectrum.data.iter().map(|&value| value.to_f64()).collect())
        })
        .expect("block index should build");
        let block_products = SpectrumBlockProductIndex::build(&spectra, 2)
            .expect("block product index should build");
        let index = build_test_index(spectra, 0.1);
        let mut state = index.new_search_state();

        block_index
            .prepare_allowed_blocks(&[100.0], 0.1, 0.5, &mut state, |_, max_value| max_value);
        let mut scored = Vec::new();
        index.for_each_allowed_block_raw_score(
            &[100.0],
            &[2.0],
            None,
            &block_products,
            &mut state,
            |spec_id, raw, count| scored.push((spec_id, raw, count)),
        );

        scored.sort_by_key(|&(spec_id, _, _)| spec_id);
        assert_eq!(scored, vec![(0, 2.0, 1), (1, 1.6, 1)]);
        assert_eq!(state.diagnostics.spectrum_blocks_evaluated, 2);
        assert_eq!(state.diagnostics.spectrum_blocks_allowed, 1);
        assert_eq!(state.diagnostics.spectrum_blocks_pruned, 1);
        assert_eq!(state.diagnostics.product_postings_visited, 2);
        assert_eq!(state.diagnostics.candidates_marked, 2);
    }

    #[test]
    fn search_direct_filters_zero_scores_and_handles_empty_inputs() {
        let empty_index = build_test_index(vec![], 0.1);
        assert!(
            empty_index
                .search_direct(&[100.0], &[1.0], &(), None)
                .is_empty()
        );

        let zero_score_index = build_test_index(vec![prepared(200.0, vec![100.0], vec![0.0])], 0.1);
        let mut state = zero_score_index.new_search_state();
        let zero_results =
            zero_score_index.search_direct_with_state(&[100.0], &[5.0], &(), None, &mut state);
        assert!(zero_results.is_empty());

        let repeated_empty = zero_score_index.search_direct_with_state(
            &[] as &[f64],
            &[] as &[f64],
            &(),
            None,
            &mut state,
        );
        assert!(repeated_empty.is_empty());
    }

    #[test]
    fn modified_search_upgrades_direct_matches_and_reuses_state_cleanly() {
        let index = build_test_index(vec![prepared(200.0, vec![100.0], vec![1.0])], 0.1);
        let mut state = index.new_search_state();

        let upgraded =
            index.search_modified_with_state(&[100.0, 110.0], &[1.0, 5.0], &(), 210.0, &mut state);
        assert_eq!(
            upgraded,
            vec![FlashSearchResult {
                spectrum_id: 0,
                score: 5.0,
                n_matches: 1,
            }]
        );

        let shifted_only =
            index.search_modified_with_state(&[110.0], &[3.0], &(), 210.0, &mut state);
        assert_eq!(
            shifted_only,
            vec![FlashSearchResult {
                spectrum_id: 0,
                score: 3.0,
                n_matches: 1,
            }]
        );

        let no_match = index.search_modified_with_state(&[150.0], &[2.0], &(), 210.0, &mut state);
        assert!(no_match.is_empty());

        let repeated =
            index.search_modified_with_state(&[100.0, 110.0], &[1.0, 5.0], &(), 210.0, &mut state);
        assert_eq!(repeated, upgraded);
    }
}
