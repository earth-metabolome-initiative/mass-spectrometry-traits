//! Structures for mass spectrometry data.

mod cosine_common;
pub mod entropy_similarity;
pub mod generic_spectrum;
pub mod hungarian_cosine;
pub mod iterators;
pub mod modified_hungarian_cosine;
pub mod similarity_errors;

pub use entropy_similarity::EntropySimilarity;
pub use generic_spectrum::GenericSpectrum;
pub use hungarian_cosine::HungarianCosine;
pub use iterators::GreedySharedPeaks;
pub use modified_hungarian_cosine::ModifiedHungarianCosine;
pub use similarity_errors::{SimilarityComputationError, SimilarityConfigError};
