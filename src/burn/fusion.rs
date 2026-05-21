//! `Fusion<B>` integration for the spectral similarity kernels.
//!
//! Registers each (metric x shape) as a distinct custom op in Burn's fusion
//! IR, the op name embeds `M::NAME` so the fusion runtime can cache and
//! merge streams metric-by-metric. The `Operation` impl simply calls back
//! into `B`'s `SpectralKernelBackend<M>` implementation, so the actual GPU
//! work is the same. Fusion just lets the surrounding tensor operations
//! batch and reorder around our kernels.

use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Element as _, Shape};
use burn_fusion::stream::{Operation, OperationStreams};
use burn_fusion::{Fusion, FusionBackend};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use core::marker::PhantomData;

use crate::burn::api::{
    CrossConfig, PairedConfig, PairwisePrimitive, RankingConfig, SpectralKernelBackend,
    SpectrumPrimitive,
};
use crate::burn::metrics::KernelMetric;

impl<B, M> SpectralKernelBackend<M> for Fusion<B>
where
    B: FusionBackend + SpectralKernelBackend<M> + Send + Sync,
    M: KernelMetric,
{
    fn paired_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        params: PairwisePrimitive<Self>,
        config: PairedConfig<M>,
    ) -> FloatTensor<Self> {
        let [batch_size, _left_peaks] = left.mz.shape.dims();
        let [right_rows, _right_peaks] = right.mz.shape.dims();
        assert_eq!(
            batch_size, right_rows,
            "paired kernel requires the same number of left and right rows"
        );

        let output_shape = Shape::new([batch_size]);
        let streams = OperationStreams::with_inputs([
            &left.mz,
            &left.intensity,
            &left.precursor,
            &right.mz,
            &right.intensity,
            &right.precursor,
            &params.mz_power,
            &params.intensity_power,
            &params.mz_tolerance,
        ]);
        let client = left.mz.client.clone();
        let dtype = left.mz.dtype;
        let output = TensorIr::uninit(client.create_empty_handle(), output_shape, dtype);
        let desc = CustomOpIr::new(
            M::PAIRED_FUSION_NAME,
            &[
                left.mz.into_ir(),
                left.intensity.into_ir(),
                left.precursor.into_ir(),
                right.mz.into_ir(),
                right.intensity.into_ir(),
                right.precursor.into_ir(),
                params.mz_power.into_ir(),
                params.intensity_power.into_ir(),
                params.mz_tolerance.into_ir(),
            ],
            &[output],
        );

        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                PairedFusionForward::<B, M> {
                    desc,
                    config,
                    backend: PhantomData,
                    metric: PhantomData,
                },
            )
            .output()
    }

    fn cross_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        config: CrossConfig<M>,
    ) -> FloatTensor<Self> {
        let [m_rows, _left_peaks] = left.mz.shape.dims();
        let [n_rows, _right_peaks] = right.mz.shape.dims();
        let output_shape = Shape::new([m_rows, n_rows]);
        let streams = OperationStreams::with_inputs([
            &left.mz,
            &left.intensity,
            &left.precursor,
            &right.mz,
            &right.intensity,
            &right.precursor,
        ]);
        let client = left.mz.client.clone();
        let dtype = left.mz.dtype;
        let output = TensorIr::uninit(client.create_empty_handle(), output_shape, dtype);
        let desc = CustomOpIr::new(
            M::CROSS_FUSION_NAME,
            &[
                left.mz.into_ir(),
                left.intensity.into_ir(),
                left.precursor.into_ir(),
                right.mz.into_ir(),
                right.intensity.into_ir(),
                right.precursor.into_ir(),
            ],
            &[output],
        );

        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                CrossFusionForward::<B, M> {
                    desc,
                    config,
                    backend: PhantomData,
                    metric: PhantomData,
                },
            )
            .output()
    }

    fn ranking_score(
        teacher: SpectrumPrimitive<Self>,
        config: RankingConfig<M>,
    ) -> (
        IntTensor<Self>,
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let [teacher_rows, teacher_peaks] = teacher.mz.shape.dims();
        let [precursor_rows] = teacher.precursor.shape.dims();
        let max_peaks = config.max_peaks();
        let batch_start = config.batch_start();
        let batch_items = config.batch_items();
        assert_eq!(
            teacher_rows, precursor_rows,
            "ranking kernel: precursor cache must have one value per teacher row"
        );
        assert!(
            teacher_peaks <= max_peaks,
            "ranking kernel: teacher peak_width {teacher_peaks} exceeds config.max_peaks {max_peaks}",
        );
        assert!(
            batch_items >= 3,
            "ranking kernel: batch_items must be >= 3, got {batch_items}",
        );
        assert!(
            batch_start + batch_items <= teacher_rows,
            "ranking kernel: batch slice [{batch_start}, {}) is outside the teacher cache [0, {teacher_rows})",
            batch_start + batch_items,
        );

        let streams =
            OperationStreams::with_inputs([&teacher.mz, &teacher.intensity, &teacher.precursor]);
        let client = teacher.mz.client.clone();
        let candidate_count = config.effective_candidates_per_anchor();
        let candidate_shape = Shape::new([batch_items, candidate_count]);
        let position_shape = Shape::new([batch_items]);
        let gap_shape = Shape::new([batch_items]);
        let scores_shape = Shape::new([batch_items, candidate_count]);
        let dtype = teacher.mz.dtype;
        let candidate_index = TensorIr::uninit(
            client.create_empty_handle(),
            candidate_shape,
            B::IntElem::dtype(),
        );
        let best_candidate_position = TensorIr::uninit(
            client.create_empty_handle(),
            position_shape,
            B::IntElem::dtype(),
        );
        let top2_gap = TensorIr::uninit(client.create_empty_handle(), gap_shape, dtype);
        let candidate_scores = TensorIr::uninit(client.create_empty_handle(), scores_shape, dtype);
        let desc = CustomOpIr::new(
            M::RANKING_FUSION_NAME,
            &[
                teacher.mz.into_ir(),
                teacher.intensity.into_ir(),
                teacher.precursor.into_ir(),
            ],
            &[
                candidate_index,
                best_candidate_position,
                top2_gap,
                candidate_scores,
            ],
        );

        let mut outputs = client.register(
            streams,
            OperationIr::Custom(desc.clone()),
            RankingFusionForward::<B, M> {
                desc,
                config,
                backend: PhantomData,
                metric: PhantomData,
            },
        );
        let candidate_scores = outputs
            .pop()
            .expect("ranking custom op has candidate-scores output");
        let top2_gap = outputs
            .pop()
            .expect("ranking custom op has top-2 gap output");
        let best_candidate_position = outputs
            .pop()
            .expect("ranking custom op has best candidate position output");
        let candidate_index = outputs
            .pop()
            .expect("ranking custom op has candidate-index output");

        (
            candidate_index,
            best_candidate_position,
            top2_gap,
            candidate_scores,
        )
    }
}

#[derive(Debug)]
struct PairedFusionForward<B: FusionBackend, M: KernelMetric> {
    desc: CustomOpIr,
    config: PairedConfig<M>,
    backend: PhantomData<B>,
    metric: PhantomData<M>,
}

impl<B, M> Operation<B::FusionRuntime> for PairedFusionForward<B, M>
where
    B: FusionBackend + SpectralKernelBackend<M> + Send + Sync,
    M: KernelMetric,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let (inputs, outputs) = self.desc.as_fixed::<9, 1>();
        let output = B::paired_score(
            SpectrumPrimitive {
                mz: handles.get_float_tensor::<B>(&inputs[0]),
                intensity: handles.get_float_tensor::<B>(&inputs[1]),
                precursor: handles.get_float_tensor::<B>(&inputs[2]),
            },
            SpectrumPrimitive {
                mz: handles.get_float_tensor::<B>(&inputs[3]),
                intensity: handles.get_float_tensor::<B>(&inputs[4]),
                precursor: handles.get_float_tensor::<B>(&inputs[5]),
            },
            PairwisePrimitive {
                mz_power: handles.get_float_tensor::<B>(&inputs[6]),
                intensity_power: handles.get_float_tensor::<B>(&inputs[7]),
                mz_tolerance: handles.get_float_tensor::<B>(&inputs[8]),
            },
            self.config,
        );
        handles.register_float_tensor::<B>(&outputs[0].id, output);
    }
}

#[derive(Debug)]
struct CrossFusionForward<B: FusionBackend, M: KernelMetric> {
    desc: CustomOpIr,
    config: CrossConfig<M>,
    backend: PhantomData<B>,
    metric: PhantomData<M>,
}

impl<B, M> Operation<B::FusionRuntime> for CrossFusionForward<B, M>
where
    B: FusionBackend + SpectralKernelBackend<M> + Send + Sync,
    M: KernelMetric,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let (inputs, outputs) = self.desc.as_fixed::<6, 1>();
        let output = B::cross_score(
            SpectrumPrimitive {
                mz: handles.get_float_tensor::<B>(&inputs[0]),
                intensity: handles.get_float_tensor::<B>(&inputs[1]),
                precursor: handles.get_float_tensor::<B>(&inputs[2]),
            },
            SpectrumPrimitive {
                mz: handles.get_float_tensor::<B>(&inputs[3]),
                intensity: handles.get_float_tensor::<B>(&inputs[4]),
                precursor: handles.get_float_tensor::<B>(&inputs[5]),
            },
            self.config,
        );
        handles.register_float_tensor::<B>(&outputs[0].id, output);
    }
}

#[derive(Debug)]
struct RankingFusionForward<B: FusionBackend, M: KernelMetric> {
    desc: CustomOpIr,
    config: RankingConfig<M>,
    backend: PhantomData<B>,
    metric: PhantomData<M>,
}

impl<B, M> Operation<B::FusionRuntime> for RankingFusionForward<B, M>
where
    B: FusionBackend + SpectralKernelBackend<M> + Send + Sync,
    M: KernelMetric,
{
    fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
        let (inputs, outputs) = self.desc.as_fixed::<3, 4>();
        let (candidate_index, best_candidate_position, top2_gap, candidate_scores) =
            B::ranking_score(
                SpectrumPrimitive {
                    mz: handles.get_float_tensor::<B>(&inputs[0]),
                    intensity: handles.get_float_tensor::<B>(&inputs[1]),
                    precursor: handles.get_float_tensor::<B>(&inputs[2]),
                },
                self.config,
            );
        handles.register_int_tensor::<B>(&outputs[0].id, candidate_index);
        handles.register_int_tensor::<B>(&outputs[1].id, best_candidate_position);
        handles.register_float_tensor::<B>(&outputs[2].id, top2_gap);
        handles.register_float_tensor::<B>(&outputs[3].id, candidate_scores);
    }
}
