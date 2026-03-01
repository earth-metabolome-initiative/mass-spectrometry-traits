use num_traits::ToPrimitive;

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
