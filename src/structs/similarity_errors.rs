//! Error types for spectral similarity configuration and execution.

/// Error returned when building similarity scorers with invalid parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SimilarityConfigError {
    /// A numeric parameter was NaN or infinite.
    #[error("parameter `{0}` must be finite")]
    NonFiniteParameter(&'static str),
    /// A numeric parameter could not be represented as an `f64`.
    #[error("parameter `{0}` must be representable as f64")]
    NonRepresentableParameter(&'static str),
    /// Tolerance must be zero or positive.
    #[error("parameter `mz_tolerance` must be >= 0")]
    NegativeTolerance,
}

/// Error returned when computing similarity scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SimilarityComputationError {
    /// A value required by the algorithm could not be represented as an `f64`.
    #[error("value `{0}` must be representable as f64")]
    ValueNotRepresentable(&'static str),
    /// A numeric value required by the algorithm was not finite.
    #[error("value `{0}` must be finite")]
    NonFiniteValue(&'static str),
    /// Tolerance must be zero or positive.
    #[error("value `mz_tolerance` must be >= 0")]
    NegativeTolerance,
    /// Peak index did not fit expected matrix index type.
    #[error("peak index overflow while building match graph")]
    IndexOverflow,
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
