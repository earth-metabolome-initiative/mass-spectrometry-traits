#![no_main]

use libfuzzer_sys::fuzz_target;
use mass_spectrometry::fuzzing::run_modified_linear_entropy_case;

fuzz_target!(|data: &[u8]| {
    let _ = run_modified_linear_entropy_case(data);
});
