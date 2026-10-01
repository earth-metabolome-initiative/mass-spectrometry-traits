use std::{cell::Cell, rc::Rc, vec, vec::Vec};

use mass_spectrometry::prelude::{
    GenericSpectrum, SpectralFilter, SpectralPipeline, SpectralPipelineBuilder, SpectralProcessor,
    Spectrum, SpectrumAlloc, SpectrumMut,
};

type FilterBox = Box<dyn SpectralFilter<Spectrum = GenericSpectrum>>;
type ProcessorBox = Box<dyn SpectralProcessor<Spectrum = GenericSpectrum>>;

fn filter_ref(filter: &FilterBox) -> &dyn SpectralFilter<Spectrum = GenericSpectrum> {
    filter.as_ref()
}

fn processor_ref(processor: &ProcessorBox) -> &dyn SpectralProcessor<Spectrum = GenericSpectrum> {
    processor.as_ref()
}

fn make_spectrum(precursor_mz: f64, peaks: &[(f64, f64)]) -> GenericSpectrum {
    let mut spectrum = GenericSpectrum::with_capacity(precursor_mz, peaks.len())
        .expect("test spectrum allocation should succeed");
    for &(mz, intensity) in peaks {
        spectrum
            .add_peak(mz, intensity)
            .expect("test peaks must be valid and sorted");
    }
    spectrum
}

fn peaks(spectrum: &GenericSpectrum) -> Vec<(f64, f64)> {
    spectrum.peaks().collect()
}

#[derive(Default)]
struct TestPipelineBuilderImpl {
    filters: Vec<FilterBox>,
    processors: Vec<ProcessorBox>,
}

struct TestPipeline {
    filters: Vec<FilterBox>,
    processors: Vec<ProcessorBox>,
}

impl SpectralPipelineBuilder for TestPipelineBuilderImpl {
    type Pipeline = TestPipeline;
    type Spectrum = GenericSpectrum;

    fn filter(mut self, filter: FilterBox) -> Self {
        self.filters.push(filter);
        self
    }

    fn processor(mut self, processor: ProcessorBox) -> Self {
        self.processors.push(processor);
        self
    }

    fn build(self) -> Self::Pipeline {
        TestPipeline {
            filters: self.filters,
            processors: self.processors,
        }
    }
}

impl SpectralPipeline for TestPipeline {
    type Spectrum = GenericSpectrum;
    type Filters<'a>
        = core::iter::Map<
        std::slice::Iter<'a, FilterBox>,
        fn(&'a FilterBox) -> &'a dyn SpectralFilter<Spectrum = GenericSpectrum>,
    >
    where
        Self: 'a;
    type Processors<'a>
        = core::iter::Map<
        std::slice::Iter<'a, ProcessorBox>,
        fn(&'a ProcessorBox) -> &'a dyn SpectralProcessor<Spectrum = GenericSpectrum>,
    >
    where
        Self: 'a;

    fn filters(&self) -> Self::Filters<'_> {
        self.filters.iter().map(filter_ref)
    }

    fn processors(&self) -> Self::Processors<'_> {
        self.processors.iter().map(processor_ref)
    }
}

struct MinPeakCountFilter {
    min_peaks: usize,
}

impl SpectralFilter for MinPeakCountFilter {
    type Spectrum = GenericSpectrum;

    fn filter(&self, spectrum: &Self::Spectrum) -> bool {
        spectrum.len() >= self.min_peaks
    }
}

struct RejectLowPrecursorFilter {
    min_precursor_exclusive: f64,
    calls: Rc<Cell<usize>>,
}

impl SpectralFilter for RejectLowPrecursorFilter {
    type Spectrum = GenericSpectrum;

    fn filter(&self, spectrum: &Self::Spectrum) -> bool {
        self.calls.set(self.calls.get() + 1);
        spectrum.precursor_mz() > self.min_precursor_exclusive
    }
}

struct CountingPassFilter {
    calls: Rc<Cell<usize>>,
}

impl SpectralFilter for CountingPassFilter {
    type Spectrum = GenericSpectrum;

    fn filter(&self, _spectrum: &Self::Spectrum) -> bool {
        self.calls.set(self.calls.get() + 1);
        true
    }
}

struct AddIntensityProcessor {
    amount: f64,
}

impl SpectralProcessor for AddIntensityProcessor {
    type Spectrum = GenericSpectrum;

    fn process(&self, spectrum: &Self::Spectrum) -> Self::Spectrum {
        let mut output = GenericSpectrum::with_capacity(spectrum.precursor_mz(), spectrum.len())
            .expect("processor should preserve valid precursor");
        for (mz, intensity) in spectrum.peaks() {
            output
                .add_peak(mz, intensity + self.amount)
                .expect("processor should preserve sorted valid peaks");
        }
        output
    }
}

struct ScaleIntensityProcessor {
    factor: f64,
}

impl SpectralProcessor for ScaleIntensityProcessor {
    type Spectrum = GenericSpectrum;

    fn process(&self, spectrum: &Self::Spectrum) -> Self::Spectrum {
        let mut output = GenericSpectrum::with_capacity(spectrum.precursor_mz(), spectrum.len())
            .expect("processor should preserve valid precursor");
        for (mz, intensity) in spectrum.peaks() {
            output
                .add_peak(mz, intensity * self.factor)
                .expect("processor should preserve sorted valid peaks");
        }
        output
    }
}

#[test]
fn empty_pipeline_is_passthrough() {
    let pipeline = TestPipelineBuilderImpl::default().build();
    let spectrum = make_spectrum(350.0, &[(50.0, 1.0), (100.0, 2.0)]);
    let expected_precursor = spectrum.precursor_mz();
    let expected_peaks = peaks(&spectrum);

    let processed = pipeline.process([spectrum]).collect::<Vec<_>>();

    assert_eq!(pipeline.filters().count(), 0);
    assert_eq!(pipeline.processors().count(), 0);
    assert_eq!(processed.len(), 1);
    assert_eq!(processed[0].precursor_mz(), expected_precursor);
    assert_eq!(peaks(&processed[0]), expected_peaks);
}

#[test]
fn filters_short_circuit_when_a_stage_rejects_spectrum() {
    let reject_calls = Rc::new(Cell::new(0));
    let pass_calls = Rc::new(Cell::new(0));

    let pipeline = TestPipelineBuilderImpl::default()
        .filter(Box::new(RejectLowPrecursorFilter {
            min_precursor_exclusive: 150.0,
            calls: Rc::clone(&reject_calls),
        }))
        .filter(Box::new(CountingPassFilter {
            calls: Rc::clone(&pass_calls),
        }))
        .build();

    let rejected = make_spectrum(120.0, &[(50.0, 1.0)]);
    let processed = pipeline.process([rejected]).collect::<Vec<_>>();

    assert!(processed.is_empty());
    assert_eq!(reject_calls.get(), 1);
    assert_eq!(pass_calls.get(), 0);
}

#[test]
fn processors_run_in_declared_order() {
    let pipeline = TestPipelineBuilderImpl::default()
        .processor(Box::new(AddIntensityProcessor { amount: 1.0 }))
        .processor(Box::new(ScaleIntensityProcessor { factor: 2.0 }))
        .build();

    let spectrum = make_spectrum(200.0, &[(100.0, 2.0)]);
    let processed = pipeline.process([spectrum]).collect::<Vec<_>>();

    assert_eq!(pipeline.processors().count(), 2);
    assert_eq!(processed.len(), 1);
    assert_eq!(processed[0].peak_nth(0), (100.0, 6.0));
}

#[test]
fn process_handles_multiple_spectra_and_excludes_filtered_out_items() {
    let pipeline = TestPipelineBuilderImpl::default()
        .filter(Box::new(MinPeakCountFilter { min_peaks: 2 }))
        .processor(Box::new(ScaleIntensityProcessor { factor: 0.5 }))
        .build();

    let rejected = make_spectrum(100.0, &[(10.0, 4.0)]);
    let accepted = make_spectrum(200.0, &[(20.0, 6.0), (30.0, 8.0)]);
    let processed = pipeline
        .process(vec![rejected, accepted])
        .collect::<Vec<_>>();

    assert_eq!(pipeline.filters().count(), 1);
    assert_eq!(processed.len(), 1);
    assert_eq!(processed[0].precursor_mz(), 200.0);
    assert_eq!(peaks(&processed[0]), vec![(20.0, 3.0), (30.0, 4.0)]);
}
