//! A swappable set-sketch abstraction for spectral overlap and LSH retrieval.
//!
//! Under flat intensity weighting, the modified-cosine match count between two
//! spectra is close to the number of buckets they share in m/z space (direct
//! matches) plus the number they share in neutral-loss space `precursor - mz`
//! (shifted matches). [`crate::structs::FlashCosineSketchIndex`] sketches and
//! bands those two spaces separately and treats a collision in either as a
//! candidate, so direct matches are caught by the m/z bands and modified analogs
//! by the neutral-loss bands. A single combined sketch would not retrieve
//! analogs: shifted fragments share only the neutral-loss keys, so the combined
//! Jaccard is capped near 1/3 however high the modified cosine, which a high
//! rows-per-band threshold rejects.
//!
//! [`Sketcher`] is the seam. [`crate::structs::FlashCosineSketchIndex`] is
//! generic over it, so swapping the backend is a type parameter, not a new index
//! type. MinHash is implemented today behind the `minhash` feature; a
//! HyperLogLog backend implements the same trait and drops straight in.

use alloc::vec::Vec;

use crate::traits::{Spectrum, SpectrumFloat};

/// A set sketch over `u64` element keys.
///
/// Implementations choose their own representation and estimators. A sketch must
/// build from keys, estimate the Jaccard index, and emit LSH band hashes used
/// for sublinear candidate retrieval.
pub trait Sketcher {
    /// Build a sketch from set element keys already reduced to `u64`.
    fn from_keys<I: IntoIterator<Item = u64>>(keys: I) -> Self;

    /// Estimate the Jaccard index (intersection over union) in `[0, 1]`.
    fn estimate_jaccard(&self, other: &Self) -> f64;
}

/// A [`Sketcher`] that also supports locality-sensitive hashing by banding.
///
/// Only minwise-bandable sketches (MinHash today, a SetSketch-style sketch
/// later) can implement this. Emitting band hashes whose collision probability
/// rises with the Jaccard index is what makes sublinear candidate retrieval
/// sound. Sketches that estimate similarity another way, for example a plain
/// HyperLogLog via inclusion-exclusion, are [`Sketcher`]s but not
/// `LshSketcher`s.
pub trait LshSketcher: Sketcher {
    /// Append one LSH band hash per band to `out`, splitting the sketch into at
    /// most `num_bands` bands. Two sketches sharing any band hash are LSH
    /// candidates. The build and query sides must request the same `num_bands`,
    /// and the number of hashes appended is deterministic for a given sketch
    /// type and `num_bands`.
    fn band_hashes_into(&self, num_bands: usize, out: &mut Vec<u64>);
}

/// Build a sketch of a spectrum's bucket set with the chosen backend `K`.
pub fn sketch_spectrum<S, K>(spectrum: &S, mz_tolerance: f64) -> K
where
    S: Spectrum,
    K: Sketcher,
{
    K::from_keys(spectrum_sketch_keys(spectrum, mz_tolerance))
}

/// The bucket keys of `spectrum`: one key per peak in m/z space (direct matches)
/// and one per peak in neutral-loss space `precursor - mz` (shifted matches),
/// tagged into disjoint key ranges so a shared key always means a shared bucket.
///
/// Buckets are `2 * mz_tolerance` wide. Two peaks within tolerance usually share
/// a bucket, though a pair straddling a bucket boundary can miss, which only
/// lowers the proxy's recall. `mz_tolerance` must be positive.
pub fn spectrum_sketch_keys<S>(spectrum: &S, mz_tolerance: f64) -> impl Iterator<Item = u64> + '_
where
    S: Spectrum,
{
    bucket_keys_from_mz(
        spectrum.mz().map(SpectrumFloat::to_f64),
        spectrum.precursor_mz().to_f64(),
        mz_tolerance,
    )
}

/// The bucket keys for a spectrum given its m/z values and precursor directly,
/// shared by [`spectrum_sketch_keys`] and the index build path that already has
/// the packed per-spectrum peaks.
pub(crate) fn bucket_keys_from_mz<I>(
    mz: I,
    precursor_mz: f64,
    mz_tolerance: f64,
) -> impl Iterator<Item = u64>
where
    I: IntoIterator<Item = f64>,
{
    let width = 2.0 * mz_tolerance;
    debug_assert!(width > 0.0, "mz_tolerance must be positive");
    mz.into_iter().flat_map(move |mz| {
        [
            bucket_key(mz, width, false),
            bucket_key(precursor_mz - mz, width, true),
        ]
    })
}

/// The m/z-space bucket keys: one direct-match key per peak. Banding a sketch of
/// these alone retrieves spectra that share fragments at the same m/z.
pub(crate) fn mz_bucket_keys_from_mz<I>(mz: I, mz_tolerance: f64) -> impl Iterator<Item = u64>
where
    I: IntoIterator<Item = f64>,
{
    let width = 2.0 * mz_tolerance;
    debug_assert!(width > 0.0, "mz_tolerance must be positive");
    mz.into_iter().map(move |mz| bucket_key(mz, width, false))
}

/// The neutral-loss-space bucket keys: one shifted-match key per peak, keyed by
/// `precursor - mz`. An analog's shifted fragments share these keys, so banding a
/// sketch of these alone retrieves modified analogs that share no direct m/z.
pub(crate) fn neutral_loss_bucket_keys_from_mz<I>(
    mz: I,
    precursor_mz: f64,
    mz_tolerance: f64,
) -> impl Iterator<Item = u64>
where
    I: IntoIterator<Item = f64>,
{
    let width = 2.0 * mz_tolerance;
    debug_assert!(width > 0.0, "mz_tolerance must be positive");
    mz.into_iter()
        .map(move |mz| bucket_key(precursor_mz - mz, width, true))
}

/// Map a value to a `u64` bucket key, tagging the neutral-loss space into a
/// disjoint range so direct and shifted buckets never collide.
#[inline]
fn bucket_key(value: f64, width: f64, neutral_loss: bool) -> u64 {
    // Offset keeps negative neutral-loss buckets non-negative, and the space bit
    // sits far above any realistic bucket id, so the two spaces are disjoint.
    const KEY_OFFSET: i64 = 1 << 32;
    const NEUTRAL_LOSS_SPACE: u64 = 1 << 62;
    let bucket = (value / width).floor() as i64;
    let key = bucket.wrapping_add(KEY_OFFSET) as u64;
    if neutral_loss {
        key | NEUTRAL_LOSS_SPACE
    } else {
        key
    }
}

#[cfg(feature = "minhash")]
mod minhash_backend {
    use alloc::vec::Vec;

    use minhash_rs::prelude::{Maximal, Min, MinHash, Primitive, XorShift, band_hash};

    use super::{LshSketcher, Sketcher};

    impl<Word, const PERMUTATIONS: usize> Sketcher for MinHash<Word, PERMUTATIONS>
    where
        Word: Min + Ord + Clone + Copy + Eq + Maximal + XorShift + Into<u64>,
        u64: Primitive<Word>,
    {
        fn from_keys<I: IntoIterator<Item = u64>>(keys: I) -> Self {
            keys.into_iter().collect()
        }

        fn estimate_jaccard(&self, other: &Self) -> f64 {
            self.estimate_jaccard_index(other)
        }
    }

    impl<Word, const PERMUTATIONS: usize> LshSketcher for MinHash<Word, PERMUTATIONS>
    where
        Word: Min + Ord + Clone + Copy + Eq + Maximal + XorShift + Into<u64> + core::hash::Hash,
        u64: Primitive<Word>,
    {
        fn band_hashes_into(&self, num_bands: usize, out: &mut Vec<u64>) {
            let rows = (PERMUTATIONS / num_bands.clamp(1, PERMUTATIONS)).max(1);
            let bands = PERMUTATIONS / rows;
            let words = self.as_ref();
            for band in 0..bands {
                let start = band * rows;
                out.push(band_hash(&words[start..start + rows]));
            }
        }
    }
}

#[cfg(all(test, feature = "minhash"))]
mod tests {
    use alloc::vec::Vec;

    use minhash_rs::prelude::MinHash;

    use super::{LshSketcher, Sketcher, sketch_spectrum, spectrum_sketch_keys};
    use crate::structs::GenericSpectrum;
    use crate::traits::SpectrumMut;

    /// Exact Jaccard of two key multisets, deduplicating into sets first.
    fn true_jaccard(left: &[u64], right: &[u64]) -> f64 {
        let mut left = left.to_vec();
        left.sort_unstable();
        left.dedup();
        let mut right = right.to_vec();
        right.sort_unstable();
        right.dedup();

        let (mut i, mut j, mut intersection) = (0usize, 0usize, 0usize);
        while i < left.len() && j < right.len() {
            match left[i].cmp(&right[j]) {
                core::cmp::Ordering::Less => i += 1,
                core::cmp::Ordering::Greater => j += 1,
                core::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        let union = left.len() + right.len() - intersection;
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    #[test]
    fn minhash_estimates_random_set_jaccard() {
        let left: Vec<u64> = (0..1000).collect();
        let right: Vec<u64> = (500..1500).collect();
        let left_sketch = <MinHash<u64, 256> as Sketcher>::from_keys(left.iter().copied());
        let right_sketch = <MinHash<u64, 256> as Sketcher>::from_keys(right.iter().copied());

        let estimate = left_sketch.estimate_jaccard(&right_sketch);
        let truth = true_jaccard(&left, &right);
        assert!(
            (estimate - truth).abs() < 0.05,
            "estimate {estimate} should be near truth {truth}"
        );
    }

    #[test]
    fn minhash_band_hashes_agree_for_identical_sketches() {
        let keys: Vec<u64> = (0..200).collect();
        let sketch = <MinHash<u64, 128> as Sketcher>::from_keys(keys.iter().copied());

        let mut a = Vec::new();
        let mut b = Vec::new();
        sketch.band_hashes_into(16, &mut a);
        sketch.band_hashes_into(16, &mut b);

        assert_eq!(a.len(), 16, "16 bands of 8 rows over 128 permutations");
        assert_eq!(a, b, "band hashes are deterministic for a given sketch");
    }

    #[test]
    fn spectrum_sketch_tracks_bucket_jaccard() {
        let tolerance = 0.02;
        let mut a: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 5).unwrap();
        a.add_peaks([
            (100.0, 1.0),
            (150.0, 1.0),
            (200.0, 1.0),
            (250.0, 1.0),
            (300.0, 1.0),
        ])
        .unwrap();
        let mut b: GenericSpectrum = GenericSpectrum::try_with_capacity(500.0, 5).unwrap();
        b.add_peaks([
            (100.005, 1.0),
            (150.005, 1.0),
            (200.005, 1.0),
            (400.0, 1.0),
            (450.0, 1.0),
        ])
        .unwrap();

        let a_keys: Vec<u64> = spectrum_sketch_keys(&a, tolerance).collect();
        let b_keys: Vec<u64> = spectrum_sketch_keys(&b, tolerance).collect();
        let truth = true_jaccard(&a_keys, &b_keys);

        let a_sketch: MinHash<u64, 256> = sketch_spectrum(&a, tolerance);
        let b_sketch: MinHash<u64, 256> = sketch_spectrum(&b, tolerance);
        let estimate = a_sketch.estimate_jaccard(&b_sketch);

        assert!(
            (estimate - truth).abs() < 0.1,
            "sketch estimate {estimate} should track true bucket Jaccard {truth}"
        );
    }
}
