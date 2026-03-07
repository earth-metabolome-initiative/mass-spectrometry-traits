use num_traits::ToPrimitive;

/// Rest mass of an electron in Daltons — minimum physically meaningful m/z.
pub const ELECTRON_MASS: f64 = 5.485_799_09e-4;

/// Maximum allowed m/z in Daltons (generous upper bound).
pub const MAX_MZ: f64 = 2_000_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NumericValidationError {
    NonRepresentable(&'static str),
    NonFinite(&'static str),
}

#[inline]
pub(crate) fn checked_to_f64<T: ToPrimitive>(
    value: T,
    name: &'static str,
) -> Result<f64, NumericValidationError> {
    let value = value
        .to_f64()
        .ok_or(NumericValidationError::NonRepresentable(name))?;
    ensure_finite_f64(value, name)?;
    Ok(value)
}

#[inline]
pub(crate) fn ensure_finite_f64(
    value: f64,
    name: &'static str,
) -> Result<(), NumericValidationError> {
    if !value.is_finite() {
        return Err(NumericValidationError::NonFinite(name));
    }
    Ok(())
}
