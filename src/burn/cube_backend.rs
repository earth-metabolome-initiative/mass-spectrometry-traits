//! `CubeBackend` implementations of [`crate::burn::api::SpectralKernelBackend`].
//!
//! One blanket impl covers every metric. The body delegates to the three
//! free `*_score_impl` functions, which already encapsulate the shared
//! scaffolding (device asserts, shape checks, output allocation, launch
//! dispatch) and are generic in the metric marker.

use burn::tensor::Shape;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::cubecl::{CubeCount, CubeDim, calculate_cube_count_elemwise};
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use crate::burn::api::{
    CrossConfig, PairedConfig, PairwisePrimitive, RankingConfig, SpectralKernelBackend,
    SpectrumPrimitive,
};
use crate::burn::launchers::{cross_forward, paired_forward, ranking_forward};
use crate::burn::metrics::KernelMetric;
use crate::burn::scorer_trait::SpectralPairScorer;

pub(crate) fn paired_score_impl<R, F, I, BT, M>(
    left: SpectrumPrimitive<CubeBackend<R, F, I, BT>>,
    right: SpectrumPrimitive<CubeBackend<R, F, I, BT>>,
    params: PairwisePrimitive<CubeBackend<R, F, I, BT>>,
    config: PairedConfig<M>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
    M: SpectralPairScorer + KernelMetric,
{
    let SpectrumPrimitive {
        mz: left_mz,
        intensity: left_intensity,
        precursor: left_precursor,
    } = left;
    let SpectrumPrimitive {
        mz: right_mz,
        intensity: right_intensity,
        precursor: right_precursor,
    } = right;
    let PairwisePrimitive {
        mz_power,
        intensity_power,
        mz_tolerance,
    } = params;

    left_mz.assert_is_on_same_device(&left_intensity);
    left_mz.assert_is_on_same_device(&left_precursor);
    left_mz.assert_is_on_same_device(&right_mz);
    left_mz.assert_is_on_same_device(&right_intensity);
    left_mz.assert_is_on_same_device(&right_precursor);
    left_mz.assert_is_on_same_device(&mz_power);
    left_mz.assert_is_on_same_device(&intensity_power);
    left_mz.assert_is_on_same_device(&mz_tolerance);

    let [batch_size, left_peaks] = left_mz.meta.shape().dims();
    let [right_rows, right_peaks] = right_mz.meta.shape().dims();
    let max_peaks = config.max_peaks();
    assert_eq!(
        batch_size, right_rows,
        "paired kernel requires the same number of left and right rows"
    );
    assert!(
        left_peaks <= max_peaks,
        "paired kernel: left peak_width {left_peaks} exceeds config.max_peaks {max_peaks}",
    );
    assert!(
        right_peaks <= max_peaks,
        "paired kernel: right peak_width {right_peaks} exceeds config.max_peaks {max_peaks}",
    );

    let output_shape = Shape::new([batch_size]);
    let output = empty_device_dtype(
        left_mz.client.clone(),
        left_mz.device.clone(),
        output_shape,
        left_mz.dtype,
    );
    let cube_dim = CubeDim::new(&left_mz.client, batch_size);
    let cube_count = calculate_cube_count_elemwise(&left_mz.client, batch_size, cube_dim);

    let client = left_mz.client.clone();
    paired_forward::launch::<F, M, R>(
        &client,
        cube_count,
        cube_dim,
        left_mz.into_tensor_arg(),
        left_intensity.into_tensor_arg(),
        left_precursor.into_tensor_arg(),
        right_mz.into_tensor_arg(),
        right_intensity.into_tensor_arg(),
        right_precursor.into_tensor_arg(),
        mz_power.into_tensor_arg(),
        intensity_power.into_tensor_arg(),
        mz_tolerance.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        config.epsilon(),
        max_peaks as u32,
        config.weighted(),
    );

    output
}

const CROSS_TILE_X: u32 = 16;
const CROSS_TILE_Y: u32 = 16;

pub(crate) fn cross_score_impl<R, F, I, BT, M>(
    left: SpectrumPrimitive<CubeBackend<R, F, I, BT>>,
    right: SpectrumPrimitive<CubeBackend<R, F, I, BT>>,
    config: CrossConfig<M>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
    M: SpectralPairScorer + KernelMetric,
{
    let SpectrumPrimitive {
        mz: left_mz,
        intensity: left_intensity,
        precursor: left_precursor,
    } = left;
    let SpectrumPrimitive {
        mz: right_mz,
        intensity: right_intensity,
        precursor: right_precursor,
    } = right;

    left_mz.assert_is_on_same_device(&left_intensity);
    left_mz.assert_is_on_same_device(&left_precursor);
    left_mz.assert_is_on_same_device(&right_mz);
    left_mz.assert_is_on_same_device(&right_intensity);
    left_mz.assert_is_on_same_device(&right_precursor);

    let [m_rows, left_peaks] = left_mz.meta.shape().dims();
    let [n_rows, right_peaks] = right_mz.meta.shape().dims();
    let max_peaks = config.max_peaks();
    assert!(
        left_peaks <= max_peaks,
        "cross kernel: left peak_width {left_peaks} exceeds config.max_peaks {max_peaks}",
    );
    assert!(
        right_peaks <= max_peaks,
        "cross kernel: right peak_width {right_peaks} exceeds config.max_peaks {max_peaks}",
    );

    let output_shape = Shape::new([m_rows, n_rows]);
    let output = empty_device_dtype(
        left_mz.client.clone(),
        left_mz.device.clone(),
        output_shape,
        left_mz.dtype,
    );
    let cube_dim = CubeDim::new_2d(CROSS_TILE_X, CROSS_TILE_Y);
    let cubes_x = (m_rows as u32).div_ceil(CROSS_TILE_X).max(1);
    let cubes_y = (n_rows as u32).div_ceil(CROSS_TILE_Y).max(1);
    let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);

    let client = left_mz.client.clone();
    cross_forward::launch::<F, M, R>(
        &client,
        cube_count,
        cube_dim,
        left_mz.into_tensor_arg(),
        left_intensity.into_tensor_arg(),
        left_precursor.into_tensor_arg(),
        right_mz.into_tensor_arg(),
        right_intensity.into_tensor_arg(),
        right_precursor.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        config.mz_power(),
        config.intensity_power(),
        config.mz_tolerance(),
        config.epsilon(),
        max_peaks as u32,
        config.weighted(),
    );

    output
}

#[allow(clippy::type_complexity)]
pub(crate) fn ranking_score_impl<R, F, I, BT, M>(
    teacher: SpectrumPrimitive<CubeBackend<R, F, I, BT>>,
    config: RankingConfig<M>,
) -> (
    IntTensor<CubeBackend<R, F, I, BT>>,
    IntTensor<CubeBackend<R, F, I, BT>>,
    FloatTensor<CubeBackend<R, F, I, BT>>,
    FloatTensor<CubeBackend<R, F, I, BT>>,
)
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
    M: SpectralPairScorer + KernelMetric,
{
    let SpectrumPrimitive {
        mz: teacher_mz,
        intensity: teacher_intensity,
        precursor: teacher_precursor,
    } = teacher;

    teacher_mz.assert_is_on_same_device(&teacher_intensity);
    teacher_mz.assert_is_on_same_device(&teacher_precursor);

    let [teacher_rows, teacher_peaks] = teacher_mz.meta.shape().dims();
    let [precursor_rows] = teacher_precursor.meta.shape().dims();
    let max_peaks = config.max_peaks();
    assert_eq!(
        teacher_rows, precursor_rows,
        "ranking kernel: precursor cache must have one value per teacher row"
    );
    assert!(
        teacher_peaks <= max_peaks,
        "ranking kernel: teacher peak_width {teacher_peaks} exceeds config.max_peaks {max_peaks}",
    );
    let batch_start = config.batch_start();
    let batch_items = config.batch_items();
    assert!(
        batch_items >= 3,
        "ranking kernel: batch_items must be >= 3 (one anchor + at least two partners), got {batch_items}",
    );
    assert!(
        batch_start + batch_items <= teacher_rows,
        "ranking kernel: batch slice [{batch_start}, {}) is outside the teacher cache [0, {teacher_rows})",
        batch_start + batch_items,
    );

    let candidate_count = config.effective_candidates_per_anchor();
    let candidate_shape = Shape::new([batch_items, candidate_count]);
    let position_shape = Shape::new([batch_items]);
    let gap_shape = Shape::new([batch_items]);
    let scores_shape = Shape::new([batch_items, candidate_count]);
    let candidate_index = empty_device_dtype(
        teacher_mz.client.clone(),
        teacher_mz.device.clone(),
        candidate_shape,
        I::dtype(),
    );
    let best_candidate_position = empty_device_dtype(
        teacher_mz.client.clone(),
        teacher_mz.device.clone(),
        position_shape,
        I::dtype(),
    );
    let top2_gap = empty_device_dtype(
        teacher_mz.client.clone(),
        teacher_mz.device.clone(),
        gap_shape,
        teacher_mz.dtype,
    );
    let candidate_scores = empty_device_dtype(
        teacher_mz.client.clone(),
        teacher_mz.device.clone(),
        scores_shape,
        teacher_mz.dtype,
    );

    let cube_dim = CubeDim::new(&teacher_mz.client, batch_items);
    let cube_count = calculate_cube_count_elemwise(&teacher_mz.client, batch_items, cube_dim);

    let client = teacher_mz.client.clone();
    ranking_forward::launch::<F, I, M, R>(
        &client,
        cube_count,
        cube_dim,
        teacher_mz.into_tensor_arg(),
        teacher_intensity.into_tensor_arg(),
        teacher_precursor.into_tensor_arg(),
        candidate_index.clone().into_tensor_arg(),
        best_candidate_position.clone().into_tensor_arg(),
        top2_gap.clone().into_tensor_arg(),
        candidate_scores.clone().into_tensor_arg(),
        batch_start as u32,
        batch_items as u32,
        config.candidates_per_anchor() as u32,
        config.mz_power(),
        config.intensity_power(),
        config.mz_tolerance(),
        config.seed() as u32,
        config.epsilon(),
        max_peaks as u32,
        config.weighted(),
    );

    (
        candidate_index,
        best_candidate_position,
        top2_gap,
        candidate_scores,
    )
}

impl<R, F, I, BT, M> SpectralKernelBackend<M> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
    M: SpectralPairScorer + KernelMetric,
{
    fn paired_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        params: PairwisePrimitive<Self>,
        config: PairedConfig<M>,
    ) -> FloatTensor<Self> {
        paired_score_impl::<R, F, I, BT, M>(left, right, params, config)
    }

    fn cross_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        config: CrossConfig<M>,
    ) -> FloatTensor<Self> {
        cross_score_impl::<R, F, I, BT, M>(left, right, config)
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
        ranking_score_impl::<R, F, I, BT, M>(teacher, config)
    }
}
