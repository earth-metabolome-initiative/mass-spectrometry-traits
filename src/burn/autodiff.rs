//! `Autodiff<B, C>` integration for the spectral similarity kernels.
//!
//! These kernels are non-differentiable (m/z matching is a discrete tolerance
//! lookup. The modified variants use DP on a discrete conflict graph).
//! Following the spectral-autoencoder pattern, the autodiff wrapper installs
//! a `NoGradientBackward` stub for each shape so the kernels can compose with
//! `Autodiff<B>` contexts during training. Gradients are simply not propagated
//! through them. Treat their outputs as teacher labels or ranking signals, not
//! as inputs to a loss differentiable in the scored spectra.
//!
//! One generic `impl SpectralKernelBackend<M> for Autodiff<B, C>` block covers
//! every metric marker. `M` flows through via the inner-backend bound.

use core::marker::PhantomData;

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::ops::{FloatTensor, IntTensor};

use crate::burn::api::{
    CrossConfig, PairedConfig, PairwisePrimitive, RankingConfig, SpectralKernelBackend,
    SpectrumPrimitive,
};
use crate::burn::metrics::KernelMetric;

#[derive(Debug)]
struct PairedNoGradient<M>(PhantomData<M>);

#[derive(Debug)]
struct CrossNoGradient<M>(PhantomData<M>);

#[derive(Debug)]
struct RankingNoGradient<M>(PhantomData<M>);

impl<B, M> Backward<B, 9> for PairedNoGradient<M>
where
    B: burn::tensor::backend::Backend,
    M: KernelMetric,
{
    type State = ();

    fn backward(
        self,
        _ops: Ops<Self::State, 9>,
        _grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
    }
}

impl<B, M> Backward<B, 6> for CrossNoGradient<M>
where
    B: burn::tensor::backend::Backend,
    M: KernelMetric,
{
    type State = ();

    fn backward(
        self,
        _ops: Ops<Self::State, 6>,
        _grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
    }
}

impl<B, M> Backward<B, 3> for RankingNoGradient<M>
where
    B: burn::tensor::backend::Backend,
    M: KernelMetric,
{
    type State = ();

    fn backward(
        self,
        _ops: Ops<Self::State, 3>,
        _grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
    }
}

impl<B, C, M> SpectralKernelBackend<M> for Autodiff<B, C>
where
    B: SpectralKernelBackend<M>,
    C: CheckpointStrategy,
    M: KernelMetric,
{
    fn paired_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        params: PairwisePrimitive<Self>,
        config: PairedConfig<M>,
    ) -> FloatTensor<Self> {
        match PairedNoGradient::<M>(PhantomData)
            .prepare::<C>([
                left.mz.node.clone(),
                left.intensity.node.clone(),
                left.precursor.node.clone(),
                right.mz.node.clone(),
                right.intensity.node.clone(),
                right.precursor.node.clone(),
                params.mz_power.node.clone(),
                params.intensity_power.node.clone(),
                params.mz_tolerance.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::paired_score(
                    SpectrumPrimitive {
                        mz: left.mz.primitive.clone(),
                        intensity: left.intensity.primitive.clone(),
                        precursor: left.precursor.primitive.clone(),
                    },
                    SpectrumPrimitive {
                        mz: right.mz.primitive.clone(),
                        intensity: right.intensity.primitive.clone(),
                        precursor: right.precursor.primitive.clone(),
                    },
                    PairwisePrimitive {
                        mz_power: params.mz_power.primitive.clone(),
                        intensity_power: params.intensity_power.primitive.clone(),
                        mz_tolerance: params.mz_tolerance.primitive.clone(),
                    },
                    config,
                );
                prep.finish((), output)
            }
            OpsKind::UnTracked(prep) => {
                let output = B::paired_score(
                    SpectrumPrimitive {
                        mz: left.mz.primitive,
                        intensity: left.intensity.primitive,
                        precursor: left.precursor.primitive,
                    },
                    SpectrumPrimitive {
                        mz: right.mz.primitive,
                        intensity: right.intensity.primitive,
                        precursor: right.precursor.primitive,
                    },
                    PairwisePrimitive {
                        mz_power: params.mz_power.primitive,
                        intensity_power: params.intensity_power.primitive,
                        mz_tolerance: params.mz_tolerance.primitive,
                    },
                    config,
                );
                prep.finish(output)
            }
        }
    }

    fn cross_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        config: CrossConfig<M>,
    ) -> FloatTensor<Self> {
        match CrossNoGradient::<M>(PhantomData)
            .prepare::<C>([
                left.mz.node.clone(),
                left.intensity.node.clone(),
                left.precursor.node.clone(),
                right.mz.node.clone(),
                right.intensity.node.clone(),
                right.precursor.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::cross_score(
                    SpectrumPrimitive {
                        mz: left.mz.primitive.clone(),
                        intensity: left.intensity.primitive.clone(),
                        precursor: left.precursor.primitive.clone(),
                    },
                    SpectrumPrimitive {
                        mz: right.mz.primitive.clone(),
                        intensity: right.intensity.primitive.clone(),
                        precursor: right.precursor.primitive.clone(),
                    },
                    config,
                );
                prep.finish((), output)
            }
            OpsKind::UnTracked(prep) => {
                let output = B::cross_score(
                    SpectrumPrimitive {
                        mz: left.mz.primitive,
                        intensity: left.intensity.primitive,
                        precursor: left.precursor.primitive,
                    },
                    SpectrumPrimitive {
                        mz: right.mz.primitive,
                        intensity: right.intensity.primitive,
                        precursor: right.precursor.primitive,
                    },
                    config,
                );
                prep.finish(output)
            }
        }
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
        // Both float outputs are non-differentiable in the spectra. We wrap
        // each through its own no-grad backward node so the autodiff graph
        // sees a defined identity for both tensors while propagating no
        // gradient through them.
        let nodes = [
            teacher.mz.node.clone(),
            teacher.intensity.node.clone(),
            teacher.precursor.node.clone(),
        ];

        let (candidate_index, best_position, top2_gap, candidate_scores) = B::ranking_score(
            SpectrumPrimitive {
                mz: teacher.mz.primitive,
                intensity: teacher.intensity.primitive,
                precursor: teacher.precursor.primitive,
            },
            config,
        );

        let wrapped_top2_gap = match RankingNoGradient::<M>(PhantomData)
            .prepare::<C>(nodes.clone())
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((), top2_gap),
            OpsKind::UnTracked(prep) => prep.finish(top2_gap),
        };
        let wrapped_candidate_scores = match RankingNoGradient::<M>(PhantomData)
            .prepare::<C>(nodes)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((), candidate_scores),
            OpsKind::UnTracked(prep) => prep.finish(candidate_scores),
        };

        (
            candidate_index,
            best_position,
            wrapped_top2_gap,
            wrapped_candidate_scores,
        )
    }
}
