use mass_spectrometry::prelude::{
    FlashSearchResult, GenericSpectrumMutationError, RandomSpectrumGenerationError,
    SimilarityComputationError, SimilarityConfigError,
};

#[test]
fn similarity_config_error_messages_are_stable() {
    assert_eq!(
        SimilarityConfigError::NonFiniteParameter("mz_power").to_string(),
        "parameter `mz_power` must be finite"
    );
    assert_eq!(
        SimilarityConfigError::InvalidParameter("max_peak_num").to_string(),
        "invalid parameter `max_peak_num`"
    );
    assert_eq!(
        SimilarityConfigError::NegativeTolerance.to_string(),
        "parameter `mz_tolerance` must be >= 0"
    );
}

#[test]
fn similarity_computation_error_messages_are_stable() {
    assert_eq!(
        SimilarityComputationError::NonFiniteValue("query_precursor_mz").to_string(),
        "value `query_precursor_mz` must be finite"
    );
    assert_eq!(
        SimilarityComputationError::GraphConstructionFailed.to_string(),
        "failed while building peak matching graph"
    );
    assert_eq!(
        SimilarityComputationError::InvalidPeakSpacing("left spectrum").to_string(),
        "`left spectrum` violates strict peak spacing precondition: consecutive peaks must be > 2 * mz_tolerance apart"
    );
}

#[test]
fn random_spectrum_generation_error_messages_are_stable() {
    assert_eq!(
        RandomSpectrumGenerationError::<GenericSpectrumMutationError>::InvalidConfig(
            "mz_max must be >= mz_min"
        )
        .to_string(),
        "invalid random spectrum config: mz_max must be >= mz_min"
    );
    assert_eq!(
        RandomSpectrumGenerationError::<GenericSpectrumMutationError>::NonFiniteValue("mz_min")
            .to_string(),
        "value must be finite: mz_min"
    );
    assert_eq!(
        RandomSpectrumGenerationError::Mutation(GenericSpectrumMutationError::UnsortedMz)
            .to_string(),
        "mz values must be added in sorted order"
    );
}

#[test]
fn flash_search_results_sort_and_compare_by_all_fields() {
    let mut results = vec![
        FlashSearchResult {
            spectrum_id: 2,
            score: 0.25,
            n_matches: 1,
        },
        FlashSearchResult {
            spectrum_id: 0,
            score: 0.95,
            n_matches: 4,
        },
        FlashSearchResult {
            spectrum_id: 1,
            score: 0.5,
            n_matches: 2,
        },
    ];

    results.sort_by_key(|result| result.spectrum_id);

    assert_eq!(
        results,
        vec![
            FlashSearchResult {
                spectrum_id: 0,
                score: 0.95,
                n_matches: 4,
            },
            FlashSearchResult {
                spectrum_id: 1,
                score: 0.5,
                n_matches: 2,
            },
            FlashSearchResult {
                spectrum_id: 2,
                score: 0.25,
                n_matches: 1,
            },
        ]
    );
    assert_ne!(
        results[0],
        FlashSearchResult {
            spectrum_id: 0,
            score: 0.95,
            n_matches: 3,
        }
    );
}
