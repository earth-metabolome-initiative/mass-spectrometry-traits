//! Submodule providing reference spectra for common molecules.
#![allow(clippy::excessive_precision)]

macro_rules! impl_reference_spectrum {
    ($trait_name:ident, $method_name:ident, $precursor:ident, $mz:ident, $intensities:ident) => {
        impl<S: crate::traits::SpectrumAlloc> $trait_name for S
        where
            S::Mz: From<f32>,
            S::Intensity: From<f32>,
        {
            fn $method_name() -> Result<Self, <Self as crate::traits::SpectrumMut>::MutationError> {
                let mut spectrum = Self::with_capacity($precursor.into(), $mz.len());
                for (&mz, &intensity) in $mz.iter().zip($intensities.iter()) {
                    spectrum.add_peak(mz.into(), intensity.into())?;
                }
                Ok(spectrum)
            }
        }
    };
}

pub(crate) use impl_reference_spectrum;

pub mod acephate;
pub mod acetyl_coenzyme_a;
pub mod adenine;
pub mod adenosine;
pub mod adenosine_5_diphosphate;
pub mod adenosine_5_monophosphate;
pub mod alanine;
pub mod arachidic_acid;
pub mod arachidonic_acid;
pub mod arginine;
pub mod ascorbic_acid;
pub mod aspartic_acid;
pub mod aspirin;
pub mod avermectin;
pub mod biotin;
pub mod boscalid;
pub mod chlorantraniliprole;
pub mod chlorfluazuron;
pub mod chlorotoluron;
pub mod citric_acid;
pub mod clothianidin;
pub mod cocaine;
pub mod cyazofamid;
pub mod cymoxanil;
pub mod cysteine;
pub mod cytidine;
pub mod cytidine_5_diphosphate;
pub mod cytidine_5_triphosphate;
pub mod desmosterol;
pub mod diflubenzuron;
pub mod dihydrosphingosine;
pub mod diniconazole;
pub mod dinotefuran;
pub mod diuron;
pub mod doramectin;
pub mod elaidic_acid;
pub mod epimeloscine;
pub mod eprinomectin;
pub mod ethiprole;
pub mod ethirimol;
pub mod fipronil;
pub mod flonicamid;
pub mod fluazinam;
pub mod fludioxinil;
pub mod flufenoxuron;
pub mod fluometuron;
pub mod flutolanil;
pub mod folic_acid;
pub mod forchlorfenuron;
pub mod fuberidazole;
pub mod glucose;
pub mod halofenozide;
pub mod hexaflumuron;
pub mod hydramethylnon;
pub mod hydroxycholesterol;
pub mod ivermectin;
pub mod lufenuron;
pub mod metaflumizone;
pub mod n1_2_dierucoyl_sn_glycero_3_phosphocholine;
pub mod n1_2_dioleoyl_rac_glycerol;
pub mod n1_oleoyl_sn_glycero_3_phosphocholine;
pub mod n1_palmitoyl_sn_glycero_3_phosphocholine;
pub mod n2_5_dihydroxybenzoic_acid;
pub mod n4_aminobenzoic_acid;
pub mod n4_cholesten_3_one;
pub mod neburon;
pub mod nitenpyram;
pub mod novaluron;
pub mod phenylalanine;
pub mod prothioconazole;
pub mod pymetrozine;
pub mod pyrimethanil;
pub mod salicin;
pub mod sulfentrazone;
pub mod tebufenozide;
pub mod teflubenzuron;
pub mod thidiazuron;
pub mod thiophanate;
pub mod triadimefon;
pub mod triflumuron;

pub use acephate::AcephateSpectrum;
pub use acetyl_coenzyme_a::AcetylCoenzymeASpectrum;
pub use adenine::AdenineSpectrum;
pub use adenosine::AdenosineSpectrum;
pub use adenosine_5_diphosphate::Adenosine5DiphosphateSpectrum;
pub use adenosine_5_monophosphate::Adenosine5MonophosphateSpectrum;
pub use alanine::AlanineSpectrum;
pub use arachidic_acid::ArachidicAcidSpectrum;
pub use arachidonic_acid::ArachidonicAcidSpectrum;
pub use arginine::ArginineSpectrum;
pub use ascorbic_acid::AscorbicAcidSpectrum;
pub use aspartic_acid::AsparticAcidSpectrum;
pub use aspirin::AspirinSpectrum;
pub use avermectin::AvermectinSpectrum;
pub use biotin::BiotinSpectrum;
pub use boscalid::BoscalidSpectrum;
pub use chlorantraniliprole::ChlorantraniliproleSpectrum;
pub use chlorfluazuron::ChlorfluazuronSpectrum;
pub use chlorotoluron::ChlorotoluronSpectrum;
pub use citric_acid::CitricAcidSpectrum;
pub use clothianidin::ClothianidinSpectrum;
pub use cocaine::CocaineSpectrum;
pub use cyazofamid::CyazofamidSpectrum;
pub use cymoxanil::CymoxanilSpectrum;
pub use cysteine::CysteineSpectrum;
pub use cytidine::CytidineSpectrum;
pub use cytidine_5_diphosphate::Cytidine5DiphosphateSpectrum;
pub use cytidine_5_triphosphate::Cytidine5TriphosphateSpectrum;
pub use desmosterol::DesmosterolSpectrum;
pub use diflubenzuron::DiflubenzuronSpectrum;
pub use dihydrosphingosine::DihydrosphingosineSpectrum;
pub use diniconazole::DiniconazoleSpectrum;
pub use dinotefuran::DinotefuranSpectrum;
pub use diuron::DiuronSpectrum;
pub use doramectin::DoramectinSpectrum;
pub use elaidic_acid::ElaidicAcidSpectrum;
pub use epimeloscine::EpimeloscineSpectrum;
pub use eprinomectin::EprinomectinSpectrum;
pub use ethiprole::EthiproleSpectrum;
pub use ethirimol::EthirimolSpectrum;
pub use fipronil::FipronilSpectrum;
pub use flonicamid::FlonicamidSpectrum;
pub use fluazinam::FluazinamSpectrum;
pub use fludioxinil::FludioxinilSpectrum;
pub use flufenoxuron::FlufenoxuronSpectrum;
pub use fluometuron::FluometuronSpectrum;
pub use flutolanil::FlutolanilSpectrum;
pub use folic_acid::FolicAcidSpectrum;
pub use forchlorfenuron::ForchlorfenuronSpectrum;
pub use fuberidazole::FuberidazoleSpectrum;
pub use glucose::GlucoseSpectrum;
pub use halofenozide::HalofenozideSpectrum;
pub use hexaflumuron::HexaflumuronSpectrum;
pub use hydramethylnon::HydramethylnonSpectrum;
pub use hydroxycholesterol::HydroxyCholesterolSpectrum;
pub use ivermectin::IvermectinSpectrum;
pub use lufenuron::LufenuronSpectrum;
pub use metaflumizone::MetaflumizoneSpectrum;
pub use n1_2_dierucoyl_sn_glycero_3_phosphocholine::N12DierucoylSnGlycero3PhosphocholineSpectrum;
pub use n1_2_dioleoyl_rac_glycerol::N12DioleoylRacGlycerolSpectrum;
pub use n1_oleoyl_sn_glycero_3_phosphocholine::N1OleoylSnGlycero3PhosphocholineSpectrum;
pub use n1_palmitoyl_sn_glycero_3_phosphocholine::N1PalmitoylSnGlycero3PhosphocholineSpectrum;
pub use n2_5_dihydroxybenzoic_acid::N25DihydroxybenzoicAcidSpectrum;
pub use n4_aminobenzoic_acid::N4AminobenzoicAcidSpectrum;
pub use n4_cholesten_3_one::N4Cholesten3OneSpectrum;
pub use neburon::NeburonSpectrum;
pub use nitenpyram::NitenpyramSpectrum;
pub use novaluron::NovaluronSpectrum;
pub use phenylalanine::PhenylalanineSpectrum;
pub use prothioconazole::ProthioconazoleSpectrum;
pub use pymetrozine::PymetrozineSpectrum;
pub use pyrimethanil::PyrimethanilSpectrum;
pub use salicin::SalicinSpectrum;
pub use sulfentrazone::SulfentrazoneSpectrum;
pub use tebufenozide::TebufenozideSpectrum;
pub use teflubenzuron::TeflubenzuronSpectrum;
pub use thidiazuron::ThidiazuronSpectrum;
pub use thiophanate::ThiophanateSpectrum;
pub use triadimefon::TriadimefonSpectrum;
pub use triflumuron::TriflumuronSpectrum;
