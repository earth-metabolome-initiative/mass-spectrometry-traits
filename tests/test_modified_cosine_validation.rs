//! Validation of ModifiedCosine against matchms ModifiedCosine (greedy).
//!
//! Since matchms uses greedy assignment and we use optimal (Crouse LAPJV),
//! our scores should be >= matchms scores. We also verify symmetry.

use mass_spectrometry::prelude::*;

/// Build a GenericSpectrum<f32, f32> by compound name.
fn build_spectrum(name: &str) -> Option<GenericSpectrum<f32, f32>> {
    Some(match name {
        "acephate" => GenericSpectrum::acephate(),
        "acetyl_coenzyme_a" => GenericSpectrum::acetyl_coenzyme_a(),
        "adenine" => GenericSpectrum::adenine(),
        "adenosine" => GenericSpectrum::adenosine(),
        "adenosine_5_diphosphate" => GenericSpectrum::adenosine_5_diphosphate(),
        "adenosine_5_monophosphate" => GenericSpectrum::adenosine_5_monophosphate(),
        "alanine" => GenericSpectrum::alanine(),
        "arachidic_acid" => GenericSpectrum::arachidic_acid(),
        "arachidonic_acid" => GenericSpectrum::arachidonic_acid(),
        "arginine" => GenericSpectrum::arginine(),
        "ascorbic_acid" => GenericSpectrum::ascorbic_acid(),
        "aspartic_acid" => GenericSpectrum::aspartic_acid(),
        "aspirin" => GenericSpectrum::aspirin(),
        "avermectin" => GenericSpectrum::avermectin(),
        "biotin" => GenericSpectrum::biotin(),
        "boscalid" => GenericSpectrum::boscalid(),
        "chlorantraniliprole" => GenericSpectrum::chlorantraniliprole(),
        "chlorfluazuron" => GenericSpectrum::chlorfluazuron(),
        "chlorotoluron" => GenericSpectrum::chlorotoluron(),
        "citric_acid" => GenericSpectrum::citric_acid(),
        "clothianidin" => GenericSpectrum::clothianidin(),
        "cocaine" => GenericSpectrum::cocaine(),
        "cyazofamid" => GenericSpectrum::cyazofamid(),
        "cymoxanil" => GenericSpectrum::cymoxanil(),
        "cysteine" => GenericSpectrum::cysteine(),
        "cytidine" => GenericSpectrum::cytidine(),
        "cytidine_5_diphosphate" => GenericSpectrum::cytidine_5_diphosphate(),
        "cytidine_5_triphosphate" => GenericSpectrum::cytidine_5_triphosphate(),
        "desmosterol" => GenericSpectrum::desmosterol(),
        "diflubenzuron" => GenericSpectrum::diflubenzuron(),
        "dihydrosphingosine" => GenericSpectrum::dihydrosphingosine(),
        "diniconazole" => GenericSpectrum::diniconazole(),
        "dinotefuran" => GenericSpectrum::dinotefuran(),
        "diuron" => GenericSpectrum::diuron(),
        "doramectin" => GenericSpectrum::doramectin(),
        "elaidic_acid" => GenericSpectrum::elaidic_acid(),
        "epimeloscine" => GenericSpectrum::epimeloscine(),
        "eprinomectin" => GenericSpectrum::eprinomectin(),
        "ethiprole" => GenericSpectrum::ethiprole(),
        "ethirimol" => GenericSpectrum::ethirimol(),
        "fipronil" => GenericSpectrum::fipronil(),
        "flonicamid" => GenericSpectrum::flonicamid(),
        "fluazinam" => GenericSpectrum::fluazinam(),
        "fludioxinil" => GenericSpectrum::fludioxinil(),
        "flufenoxuron" => GenericSpectrum::flufenoxuron(),
        "fluometuron" => GenericSpectrum::fluometuron(),
        "flutolanil" => GenericSpectrum::flutolanil(),
        "folic_acid" => GenericSpectrum::folic_acid(),
        "forchlorfenuron" => GenericSpectrum::forchlorfenuron(),
        "fuberidazole" => GenericSpectrum::fuberidazole(),
        "glucose" => GenericSpectrum::glucose(),
        "halofenozide" => GenericSpectrum::halofenozide(),
        "hexaflumuron" => GenericSpectrum::hexaflumuron(),
        "hydramethylnon" => GenericSpectrum::hydramethylnon(),
        "hydroxy_cholesterol" | "hydroxycholesterol" => GenericSpectrum::hydroxy_cholesterol(),
        "ivermectin" => GenericSpectrum::ivermectin(),
        "lufenuron" => GenericSpectrum::lufenuron(),
        "metaflumizone" => GenericSpectrum::metaflumizone(),
        "n1_2_dierucoyl_sn_glycero_3_phosphocholine" => {
            GenericSpectrum::n1_2_dierucoyl_sn_glycero_3_phosphocholine()
        }
        "n1_2_dioleoyl_rac_glycerol" => GenericSpectrum::n1_2_dioleoyl_rac_glycerol(),
        "n1_oleoyl_sn_glycero_3_phosphocholine" => {
            GenericSpectrum::n1_oleoyl_sn_glycero_3_phosphocholine()
        }
        "n1_palmitoyl_sn_glycero_3_phosphocholine" => {
            GenericSpectrum::n1_palmitoyl_sn_glycero_3_phosphocholine()
        }
        "n2_5_dihydroxybenzoic_acid" => GenericSpectrum::n2_5_dihydroxybenzoic_acid(),
        "n4_aminobenzoic_acid" => GenericSpectrum::n4_aminobenzoic_acid(),
        "n4_cholesten_3_one" => GenericSpectrum::n4_cholesten_3_one(),
        "neburon" => GenericSpectrum::neburon(),
        "nitenpyram" => GenericSpectrum::nitenpyram(),
        "novaluron" => GenericSpectrum::novaluron(),
        "phenylalanine" => GenericSpectrum::phenylalanine(),
        "prothioconazole" => GenericSpectrum::prothioconazole(),
        "pymetrozine" => GenericSpectrum::pymetrozine(),
        "pyrimethanil" => GenericSpectrum::pyrimethanil(),
        "salicin" => GenericSpectrum::salicin(),
        "sulfentrazone" => GenericSpectrum::sulfentrazone(),
        "tebufenozide" => GenericSpectrum::tebufenozide(),
        "teflubenzuron" => GenericSpectrum::teflubenzuron(),
        "thidiazuron" => GenericSpectrum::thidiazuron(),
        "thiophanate" => GenericSpectrum::thiophanate(),
        "triadimefon" => GenericSpectrum::triadimefon(),
        "triflumuron" => GenericSpectrum::triflumuron(),
        _ => return None,
    })
}

#[test]
fn validate_modified_cosine_against_matchms() {
    let csv_path = "tests/fixtures/expected_modified_cosine_similarities.csv";
    let mut reader = csv::Reader::from_path(csv_path)
        .unwrap_or_else(|e| panic!("Failed to open {csv_path}: {e}"));

    let mut tested = 0u32;
    let mut failures: Vec<String> = Vec::new();

    for result in reader.records() {
        let record = result.expect("Failed to read CSV record");

        let left_name = &record[0];
        let right_name = &record[1];
        let tolerance: f32 = record[2].parse().expect("bad tolerance");
        let mz_power: f32 = record[3].parse().expect("bad mz_power");
        let intensity_power: f32 = record[4].parse().expect("bad intensity_power");
        let matchms_score: f64 = record[5].parse().expect("bad score");

        let Some(left) = build_spectrum(left_name) else {
            continue;
        };
        let Some(right) = build_spectrum(right_name) else {
            continue;
        };

        let modified = ModifiedCosine::new(mz_power, intensity_power, tolerance);
        let (score, matches) = modified.similarity(&left, &right);
        let our_score = score as f64;

        // Our optimal assignment should produce scores >= matchms greedy
        // (minus small tolerance for floating point differences).
        if our_score < matchms_score - 1e-4 {
            failures.push(format!(
                "{left_name} vs {right_name} (tol={tolerance}, mz_pow={mz_power}, int_pow={intensity_power}): \
                 our_score={our_score:.12} < matchms_score={matchms_score:.12}"
            ));
        }

        // Also verify symmetry.
        let (score_ba, matches_ba) = modified.similarity(&right, &left);
        if (score as f64 - score_ba as f64).abs() >= 1e-6 {
            failures.push(format!(
                "{left_name} vs {right_name} (tol={tolerance}, mz_pow={mz_power}, int_pow={intensity_power}): \
                 symmetry violation: sim(A,B)={score:.12} != sim(B,A)={score_ba:.12}, \
                 matches(A,B)={matches} != matches(B,A)={matches_ba}"
            ));
        }

        tested += 1;
    }

    println!("Tested: {tested}, Failures: {}", failures.len());

    if !failures.is_empty() {
        let sample: Vec<&str> = failures.iter().take(100).map(|s| s.as_str()).collect();
        panic!(
            "{} / {tested} tests failed (showing first {}):\n{}",
            failures.len(),
            sample.len(),
            sample.join("\n")
        );
    }

    assert!(tested > 0, "No tests were run");
}
