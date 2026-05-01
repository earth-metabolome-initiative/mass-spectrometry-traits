#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

extern crate alloc;

#[cfg(any(feature = "mem_size", feature = "mem_dbg"))]
extern crate mem_dbg_crate as mem_dbg;

mod numeric_validation;

pub mod fuzzing;
pub mod structs;
pub mod traits;

/// Prelude module for the mass_spectrometry crate.
pub mod prelude {
    pub use geometric_traits::prelude::ScalarSimilarity;

    pub use crate::numeric_validation::{ELECTRON_MASS, MAX_MZ};
    pub use crate::{structs::*, traits::*};
}
