use burn_cubecl::cubecl::prelude::*;

/// Compute the normalized product of a single peak's m/z and intensity, using
/// the per-row maxima as denominators (numerically stable two-phase normalization).
///
/// Returns `0` for padding peaks (zero intensity), matching the CPU
/// `prepare_peak_products` contract in `src/structs/cosine_common.rs`.
#[cube]
pub fn peak_product<F: Float>(
    mz_tensor: &Tensor<F>,
    intensity_tensor: &Tensor<F>,
    row: usize,
    peak: usize,
    mz_power: F,
    intensity_power: F,
    mz_max: F,
    intensity_max: F,
    product_max: F,
    epsilon: F,
) -> F {
    let zero = F::new(0.0_f32);
    let intensity =
        intensity_tensor[row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
    let mut product = zero;
    if intensity > zero {
        let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
        let intensity_component = intensity.max(epsilon).powf(intensity_power) / intensity_max;
        let mz_component = mz.max(epsilon).powf(mz_power) / mz_max;
        product = intensity_component * mz_component / product_max;
    }
    product
}
