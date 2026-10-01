#![no_main]

use libfuzzer_sys::fuzz_target;
use mass_spectrometry::fuzzing::run_hungarian_cosine_case;

fuzz_target!(|data: &[u8]| {
    let _ = run_hungarian_cosine_case(data);
});
