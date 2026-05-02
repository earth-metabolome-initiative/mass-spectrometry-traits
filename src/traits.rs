//! Submodule defining traits for Mass Spectrometry

pub mod reference_spectra;
pub mod spectra;
pub mod spectra_index;
pub mod spectral_filter;
pub mod spectral_pipeline;
pub mod spectral_pipeline_builder;
pub mod spectral_processor;
pub mod spectral_similarity;
pub mod spectrum;
pub mod spectrum_annotation;
pub mod spectrum_mut;

pub use reference_spectra::{
    AcephateSpectrum, AcetylCoenzymeASpectrum, AdenineSpectrum, Adenosine5DiphosphateSpectrum,
    Adenosine5MonophosphateSpectrum, AdenosineSpectrum, AlanineSpectrum, ArachidicAcidSpectrum,
    ArachidonicAcidSpectrum, ArginineSpectrum, AscorbicAcidSpectrum, AsparticAcidSpectrum,
    AspirinSpectrum, AvermectinSpectrum, BiotinSpectrum, BoscalidSpectrum,
    ChlorantraniliproleSpectrum, ChlorfluazuronSpectrum, ChlorotoluronSpectrum, CitricAcidSpectrum,
    ClothianidinSpectrum, CocaineSpectrum, CyazofamidSpectrum, CymoxanilSpectrum, CysteineSpectrum,
    Cytidine5DiphosphateSpectrum, Cytidine5TriphosphateSpectrum, CytidineSpectrum,
    DesmosterolSpectrum, DiflubenzuronSpectrum, DihydrosphingosineSpectrum, DiniconazoleSpectrum,
    DinotefuranSpectrum, DiuronSpectrum, DoramectinSpectrum, ElaidicAcidSpectrum,
    EpimeloscineSpectrum, EprinomectinSpectrum, EthiproleSpectrum, EthirimolSpectrum,
    FipronilSpectrum, FlonicamidSpectrum, FluazinamSpectrum, FludioxinilSpectrum,
    FlufenoxuronSpectrum, FluometuronSpectrum, FlutolanilSpectrum, FolicAcidSpectrum,
    ForchlorfenuronSpectrum, FuberidazoleSpectrum, GlucoseSpectrum, HalofenozideSpectrum,
    HexaflumuronSpectrum, HydramethylnonSpectrum, HydroxyCholesterolSpectrum, IvermectinSpectrum,
    LufenuronSpectrum, MetaflumizoneSpectrum, NeburonSpectrum, NitenpyramSpectrum,
    NovaluronSpectrum, PhenylalanineSpectrum, ProthioconazoleSpectrum, PymetrozineSpectrum,
    PyrimethanilSpectrum, SalicinSpectrum, StypolTrioneSpectrum, SulfentrazoneSpectrum,
    TebufenozideSpectrum, TeflubenzuronSpectrum, ThidiazuronSpectrum, ThiophanateSpectrum,
    TriadimefonSpectrum, TriflumuronSpectrum,
};
pub use spectra::Spectra;
pub use spectra_index::SpectraIndex;
pub use spectral_filter::SpectralFilter;
pub use spectral_pipeline::SpectralPipeline;
pub use spectral_pipeline_builder::SpectralPipelineBuilder;
pub use spectral_processor::SpectralProcessor;
pub use spectral_similarity::ScalarSpectralSimilarity;
pub use spectrum::{Spectrum, SpectrumFloat};
pub use spectrum_annotation::Annotation;
pub use spectrum_mut::{
    RandomSpectrumConfig, RandomSpectrumGenerationError, SpectrumAlloc, SpectrumMut,
};
