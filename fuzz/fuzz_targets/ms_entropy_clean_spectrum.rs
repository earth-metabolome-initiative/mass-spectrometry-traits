#![no_main]

use libfuzzer_sys::fuzz_target;
use mass_spectrometry::fuzzing::run_ms_entropy_clean_spectrum_case;

fuzz_target!(|data: &[u8]| {
    let _ = run_ms_entropy_clean_spectrum_case(data);
});
