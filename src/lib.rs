#![no_std]
#![doc = include_str!("../README.md")]

extern crate alloc;

mod numeric_validation;

pub mod fuzzing;
pub mod structs;
pub mod traits;

/// Prelude module for the mass_spectrometry crate.
pub mod prelude {
    pub use geometric_traits::prelude::ScalarSimilarity;

    pub use crate::{structs::*, traits::*};
}
