//! Submodule providing data for adenosine 5 diphosphate.

use crate::traits::SpectrumAlloc;

/// Trait for a spectrum of adenosine 5 diphosphate.
pub trait Adenosine5DiphosphateSpectrum: SpectrumAlloc {
    /// Create a new spectrum of adenosine 5 diphosphate.
    fn adenosine_5_diphosphate() -> Self;
}

/// The precursor mass over charge value for adenosine 5 diphosphate.
pub const ADENOSINE_5_DIPHOSPHATE_PRECURSOR_MZ: f32 = 428.037;

/// The mass over charge values for adenosine 5 diphosphate.
pub const ADENOSINE_5_DIPHOSPHATE_MZ: [f32; 35] = [
    53.014656, 55.030468, 57.07093, 67.030762, 71.108841, 74.240028, 74.975319, 97.031715,
    105.282509, 117.537148, 119.039421, 121.104729, 135.059341, 136.067856, 137.069412, 139.117798,
    176.06456, 178.082748, 178.959366, 185.127487, 203.139771, 204.096573, 214.086105, 216.043716,
    232.1035, 250.114273, 268.123352, 312.079376, 328.2388, 330.097229, 348.107483, 349.097778,
    365.352722, 399.667542, 428.087128,
];
/// The intensities for adenosine 5 diphosphate.
pub const ADENOSINE_5_DIPHOSPHATE_INTENSITIES: [f32; 35] = [
    7974.241699,
    20735.828125,
    11555.847656,
    17080.486328,
    53804.425781,
    7378.633789,
    11020.600586,
    1210244.625,
    8619.276367,
    9960.108398,
    38282.386719,
    11386.817383,
    9347.266602,
    43313096.0,
    59694.179688,
    33892.960938,
    11745.665039,
    168555.5625,
    11135.495117,
    44391.125,
    161494.390625,
    26130.632812,
    74223.507812,
    61770.90625,
    422652.28125,
    231343.6875,
    38153.898438,
    59822.953125,
    9034.679688,
    313545.125,
    4266114.5,
    9137.298828,
    8237.135742,
    8412.500977,
    2474639.75,
];

impl<S: SpectrumAlloc> Adenosine5DiphosphateSpectrum for S
where
    S::Mz: From<f32>,
    S::Intensity: From<f32>,
{
    fn adenosine_5_diphosphate() -> Self {
        let mut spectrum = Self::with_capacity(
            ADENOSINE_5_DIPHOSPHATE_PRECURSOR_MZ.into(),
            ADENOSINE_5_DIPHOSPHATE_MZ.len(),
        );
        for (&mz, &intensity) in ADENOSINE_5_DIPHOSPHATE_MZ
            .iter()
            .zip(ADENOSINE_5_DIPHOSPHATE_INTENSITIES.iter())
        {
            spectrum
                .add_peak(mz.into(), intensity.into())
                .expect("Failed to add adenosine 5 diphosphate peak to spectrum");
        }
        spectrum
    }
}
