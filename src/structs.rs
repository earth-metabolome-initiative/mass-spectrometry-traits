//! Structures for mass spectrometry data.

mod cosine_common;
pub mod entropy_similarity;
pub mod exact_cosine;
pub mod generic_spectrum;
pub mod iterators;
pub mod modified_cosine;
pub mod similarity_errors;

pub use entropy_similarity::EntropySimilarity;
pub use exact_cosine::ExactCosine;
pub use generic_spectrum::GenericSpectrum;
pub use iterators::GreedySharedPeaks;
pub use modified_cosine::ModifiedCosine;
pub use similarity_errors::{SimilarityComputationError, SimilarityConfigError};
