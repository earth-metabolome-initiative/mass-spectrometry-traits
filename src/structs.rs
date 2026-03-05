//! Structures for mass spectrometry data.

mod cosine_common;
mod entropy_common;
mod flash_common;
pub mod flash_cosine_index;
pub mod flash_entropy_index;
pub mod generic_spectrum;
pub mod greedy_cosine;
pub mod hungarian_cosine;
pub mod iterators;
pub mod linear_cosine;
pub mod linear_entropy;
pub mod merge_close_peaks;
pub mod modified_greedy_cosine;
pub mod modified_hungarian_cosine;
pub mod modified_linear_cosine;
pub mod modified_linear_entropy;
pub mod similarity_errors;

pub use flash_common::{FlashSearchResult, SearchState};
pub use flash_cosine_index::{FlashCosineIndex, FlashCosineIndexError};
pub use flash_entropy_index::{FlashEntropyIndex, FlashEntropyIndexError};
pub use generic_spectrum::{GenericSpectrum, GenericSpectrumMutationError};
pub use greedy_cosine::GreedyCosine;
pub use hungarian_cosine::HungarianCosine;
pub use iterators::GreedySharedPeaks;
pub use linear_cosine::LinearCosine;
pub use linear_entropy::LinearEntropy;
pub use merge_close_peaks::MergeClosePeaks;
pub use modified_greedy_cosine::ModifiedGreedyCosine;
pub use modified_hungarian_cosine::ModifiedHungarianCosine;
pub use modified_linear_cosine::ModifiedLinearCosine;
pub use modified_linear_entropy::ModifiedLinearEntropy;
pub use similarity_errors::{SimilarityComputationError, SimilarityConfigError};
