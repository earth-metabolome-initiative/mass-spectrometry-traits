//! Error types for spectral similarity configuration and execution.

/// Error returned when building similarity scorers with invalid parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityConfigError {
    /// A numeric parameter was NaN or infinite.
    NonFiniteParameter(&'static str),
    /// A numeric parameter could not be represented as an `f64`.
    NonRepresentableParameter(&'static str),
    /// Tolerance must be zero or positive.
    NegativeTolerance,
}

impl core::fmt::Display for SimilarityConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteParameter(name) => write!(f, "parameter `{name}` must be finite"),
            Self::NonRepresentableParameter(name) => {
                write!(f, "parameter `{name}` must be representable as f64")
            }
            Self::NegativeTolerance => write!(f, "parameter `mz_tolerance` must be >= 0"),
        }
    }
}

impl core::error::Error for SimilarityConfigError {}

/// Error returned when computing similarity scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityComputationError {
    /// A value required by the algorithm could not be represented as an `f64`.
    ValueNotRepresentable(&'static str),
    /// Peak index did not fit expected matrix index type.
    IndexOverflow,
    /// Assignment solver failed unexpectedly.
    AssignmentFailed,
}

impl core::fmt::Display for SimilarityComputationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ValueNotRepresentable(name) => {
                write!(f, "value `{name}` must be representable as f64")
            }
            Self::IndexOverflow => write!(f, "peak index overflow while building match graph"),
            Self::AssignmentFailed => write!(f, "assignment solver failed"),
        }
    }
}

impl core::error::Error for SimilarityComputationError {}
