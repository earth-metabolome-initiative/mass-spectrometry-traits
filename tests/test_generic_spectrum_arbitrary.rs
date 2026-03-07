#![cfg(feature = "proptest")]

use mass_spectrometry::prelude::{ELECTRON_MASS, GenericSpectrum, Spectrum};
use proptest::prelude::*;

proptest! {
    #[test]
    fn arbitrary_f32_spectra_respect_invariants(spectrum in any::<GenericSpectrum<f32, f32>>()) {
        prop_assert!(spectrum.precursor_mz().is_finite());
        prop_assert!(spectrum.precursor_mz() >= ELECTRON_MASS as f32);

        let mut last_mz: Option<f32> = None;
        for (mz, intensity) in spectrum.peaks() {
            prop_assert!(mz.is_finite());
            prop_assert!(mz >= ELECTRON_MASS as f32);
            prop_assert!(intensity.is_finite());
            prop_assert!(intensity > 0.0);

            if let Some(previous_mz) = last_mz {
                prop_assert!(mz > previous_mz);
            }
            last_mz = Some(mz);
        }
    }

    #[test]
    fn arbitrary_u32_spectra_have_strictly_increasing_mz(spectrum in any::<GenericSpectrum<u32, u32>>()) {
        let mut last_mz: Option<u32> = None;
        for mz in spectrum.mz() {
            if let Some(previous_mz) = last_mz {
                prop_assert!(mz > previous_mz);
            }
            last_mz = Some(mz);
        }
    }
}
