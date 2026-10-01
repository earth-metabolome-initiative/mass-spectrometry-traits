#![cfg(feature = "proptest")]

use mass_spectrometry::prelude::{ELECTRON_MASS, GenericSpectrum, Spectrum};
use proptest::prelude::*;

proptest! {
    #[test]
    fn arbitrary_spectra_respect_invariants(spectrum in any::<GenericSpectrum>()) {
        prop_assert!(spectrum.precursor_mz().is_finite());
        prop_assert!(spectrum.precursor_mz() >= ELECTRON_MASS);

        let mut last_mz: Option<f64> = None;
        for (mz, intensity) in spectrum.peaks() {
            prop_assert!(mz.is_finite());
            prop_assert!(mz >= ELECTRON_MASS);
            prop_assert!(intensity.is_finite());
            prop_assert!(intensity > 0.0);

            if let Some(previous_mz) = last_mz {
                prop_assert!(mz > previous_mz);
            }
            last_mz = Some(mz);
        }
    }
}
