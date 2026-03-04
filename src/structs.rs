//! Structures for mass spectrometry data.

mod cosine_common;
pub mod entropy_similarity;
pub mod generic_spectrum;
pub mod greedy_cosine;
pub mod hungarian_cosine;
pub mod iterators;
pub mod linear_cosine;
pub mod merge_close_peaks;
pub mod modified_greedy_cosine;
pub mod modified_hungarian_cosine;
pub mod modified_linear_cosine;
pub mod similarity_errors;

pub use entropy_similarity::EntropySimilarity;
pub use generic_spectrum::{GenericSpectrum, GenericSpectrumMutationError};
pub use greedy_cosine::GreedyCosine;
pub use hungarian_cosine::HungarianCosine;
pub use iterators::GreedySharedPeaks;
pub use linear_cosine::LinearCosine;
pub use merge_close_peaks::MergeClosePeaks;
pub use modified_greedy_cosine::ModifiedGreedyCosine;
pub use modified_hungarian_cosine::ModifiedHungarianCosine;
pub use modified_linear_cosine::ModifiedLinearCosine;
pub use similarity_errors::{SimilarityComputationError, SimilarityConfigError};
