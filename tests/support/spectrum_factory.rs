use std::iter;

use mass_spectrometry::prelude::*;

type SpectrumBuilder =
    fn() -> Result<GenericSpectrum, <GenericSpectrum as SpectrumMut>::MutationError>;

struct SpectrumEntry {
    canonical: &'static str,
    builder: SpectrumBuilder,
    aliases: &'static [&'static str],
}

macro_rules! spectrum_registry {
    ($(($name:literal, $ctor:ident, [$($alias:literal),* $(,)?])),+ $(,)?) => {
        const SPECTRUM_REGISTRY: &[SpectrumEntry] = &[
            $(SpectrumEntry {
                canonical: $name,
                builder: GenericSpectrum::$ctor,
                aliases: &[$($alias),*],
            },)+
        ];
    };
}

spectrum_registry!(
    ("acephate", acephate, []),
    ("acetyl_coenzyme_a", acetyl_coenzyme_a, []),
    ("adenine", adenine, []),
    ("adenosine", adenosine, []),
    ("adenosine_5_diphosphate", adenosine_5_diphosphate, []),
    ("adenosine_5_monophosphate", adenosine_5_monophosphate, []),
    ("alanine", alanine, []),
    ("arachidic_acid", arachidic_acid, []),
    ("arachidonic_acid", arachidonic_acid, []),
    ("arginine", arginine, []),
    ("ascorbic_acid", ascorbic_acid, []),
    ("aspartic_acid", aspartic_acid, []),
    ("aspirin", aspirin, []),
    ("avermectin", avermectin, []),
    ("biotin", biotin, []),
    ("boscalid", boscalid, []),
    ("chlorantraniliprole", chlorantraniliprole, []),
    ("chlorfluazuron", chlorfluazuron, []),
    ("chlorotoluron", chlorotoluron, []),
    ("citric_acid", citric_acid, []),
    ("clothianidin", clothianidin, []),
    ("cocaine", cocaine, []),
    ("cyazofamid", cyazofamid, []),
    ("cymoxanil", cymoxanil, []),
    ("cysteine", cysteine, []),
    ("cytidine", cytidine, []),
    ("cytidine_5_diphosphate", cytidine_5_diphosphate, []),
    ("cytidine_5_triphosphate", cytidine_5_triphosphate, []),
    ("desmosterol", desmosterol, []),
    ("diflubenzuron", diflubenzuron, []),
    ("dihydrosphingosine", dihydrosphingosine, []),
    ("diniconazole", diniconazole, []),
    ("dinotefuran", dinotefuran, []),
    ("diuron", diuron, []),
    ("doramectin", doramectin, []),
    ("elaidic_acid", elaidic_acid, []),
    ("epimeloscine", epimeloscine, []),
    ("eprinomectin", eprinomectin, []),
    ("ethiprole", ethiprole, []),
    ("ethirimol", ethirimol, []),
    ("fipronil", fipronil, []),
    ("flonicamid", flonicamid, []),
    ("fluazinam", fluazinam, []),
    ("fludioxinil", fludioxinil, []),
    ("flufenoxuron", flufenoxuron, []),
    ("fluometuron", fluometuron, []),
    ("flutolanil", flutolanil, []),
    ("folic_acid", folic_acid, []),
    ("forchlorfenuron", forchlorfenuron, []),
    ("fuberidazole", fuberidazole, []),
    ("glucose", glucose, []),
    ("halofenozide", halofenozide, []),
    ("hexaflumuron", hexaflumuron, []),
    ("hydramethylnon", hydramethylnon, []),
    (
        "hydroxy_cholesterol",
        hydroxy_cholesterol,
        ["hydroxycholesterol"]
    ),
    ("ivermectin", ivermectin, []),
    ("lufenuron", lufenuron, []),
    ("metaflumizone", metaflumizone, []),
    ("neburon", neburon, []),
    ("nitenpyram", nitenpyram, []),
    ("novaluron", novaluron, []),
    ("phenylalanine", phenylalanine, []),
    ("prothioconazole", prothioconazole, []),
    ("pymetrozine", pymetrozine, []),
    ("pyrimethanil", pyrimethanil, []),
    ("salicin", salicin, []),
    ("stypoltrione", stypoltrione, []),
    ("sulfentrazone", sulfentrazone, []),
    ("tebufenozide", tebufenozide, []),
    ("teflubenzuron", teflubenzuron, []),
    ("thidiazuron", thidiazuron, []),
    ("thiophanate", thiophanate, []),
    ("triadimefon", triadimefon, []),
    ("triflumuron", triflumuron, [])
);

#[allow(dead_code)]
pub fn canonical_spectrum_names() -> Vec<&'static str> {
    SPECTRUM_REGISTRY
        .iter()
        .map(|entry| entry.canonical)
        .collect()
}

#[allow(dead_code)]
pub fn known_spectrum_names() -> Vec<&'static str> {
    SPECTRUM_REGISTRY
        .iter()
        .flat_map(|entry| iter::once(entry.canonical).chain(entry.aliases.iter().copied()))
        .collect()
}

/// Build a `GenericSpectrum` by compound name.
pub fn build_spectrum(name: &str) -> Option<GenericSpectrum> {
    for entry in SPECTRUM_REGISTRY {
        if name == entry.canonical || entry.aliases.contains(&name) {
            return (entry.builder)().ok();
        }
    }
    None
}
