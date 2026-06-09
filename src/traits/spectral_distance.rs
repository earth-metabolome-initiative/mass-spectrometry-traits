//! Metric distance derived from a spectral similarity.

use crate::structs::{LinearCosine, LinearEntropy, ModifiedLinearCosine, ModifiedLinearEntropy};

/// The metric distance of a spectral similarity in `[0, 1]`: cosine uses the
/// geodesic `arccos(sim)`, entropy the Jensen-Shannon `sqrt(1 - sim)`.
pub trait SpectralDistanceMetric {
    /// Maps a similarity in `[0, 1]` to a non-negative metric distance.
    fn distance(&self, similarity: f64) -> f64;
}

impl SpectralDistanceMetric for LinearCosine {
    fn distance(&self, similarity: f64) -> f64 {
        similarity.clamp(-1.0, 1.0).acos()
    }
}

impl SpectralDistanceMetric for ModifiedLinearCosine {
    fn distance(&self, similarity: f64) -> f64 {
        similarity.clamp(-1.0, 1.0).acos()
    }
}

impl SpectralDistanceMetric for LinearEntropy {
    fn distance(&self, similarity: f64) -> f64 {
        (1.0 - similarity).max(0.0).sqrt()
    }
}

impl SpectralDistanceMetric for ModifiedLinearEntropy {
    fn distance(&self, similarity: f64) -> f64 {
        (1.0 - similarity).max(0.0).sqrt()
    }
}
