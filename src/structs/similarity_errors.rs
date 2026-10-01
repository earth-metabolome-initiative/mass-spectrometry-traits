//! Error types for spectral similarity configuration and execution.

/// Error returned when building similarity scorers with invalid parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SimilarityConfigError {
    /// A numeric parameter was NaN or infinite.
    #[error("parameter `{0}` must be finite")]
    NonFiniteParameter(&'static str),
    /// A parameter combination is invalid.
    #[error("invalid parameter `{0}`")]
    InvalidParameter(&'static str),
    /// Tolerance must be zero or positive.
    #[error("parameter `mz_tolerance` must be >= 0")]
    NegativeTolerance,
}

/// Error returned when computing similarity scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SimilarityComputationError {
    /// A numeric value required by the algorithm was not finite.
    #[error("value `{0}` must be finite")]
    NonFiniteValue(&'static str),
    /// A computation parameter combination is invalid.
    #[error("invalid value `{0}`")]
    InvalidParameter(&'static str),
    /// Tolerance must be zero or positive.
    #[error("value `mz_tolerance` must be >= 0")]
    NegativeTolerance,
    /// An index did not fit the data-structure index type.
    #[error("index overflow while building similarity data structures")]
    IndexOverflow,
    /// Sparse index construction failed while inserting an entry.
    #[error("failed while building sparse index")]
    IndexConstructionFailed,
    /// Graph construction failed while inserting a match edge.
    #[error("failed while building peak matching graph")]
    GraphConstructionFailed,
    /// Assignment solver failed unexpectedly.
    #[error("assignment solver failed")]
    AssignmentFailed,
    /// Linear cosine precondition violated: peaks must be strictly separated.
    #[error(
        "`{0}` violates strict peak spacing precondition: consecutive peaks must be > 2 * mz_tolerance apart"
    )]
    InvalidPeakSpacing(&'static str),
}
