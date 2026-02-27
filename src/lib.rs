#![doc = include_str!("../README.md")]

pub mod structs;
pub mod traits;

/// Prelude module for the mass_spectrometry crate.
pub mod prelude {
    pub use geometric_traits::prelude::ScalarSimilarity;

    pub use crate::{structs::*, traits::*};
}
