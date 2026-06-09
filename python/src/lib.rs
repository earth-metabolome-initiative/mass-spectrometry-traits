use ::mass_spectrometry::structs::splash::splash_from_raw_peaks_with_type;
use ::mass_spectrometry::structs::{SplashError, SplashSpectrumType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

#[pyfunction(name = "splash", signature = (peaks, spectrum_type = "mass"))]
fn py_splash(peaks: &Bound<'_, PyAny>, spectrum_type: &str) -> PyResult<String> {
    let peaks = peaks.extract::<Vec<(f64, f64)>>().map_err(|error| {
        PyValueError::new_err(format!(
            "peaks must be a sequence of (mz, intensity) pairs: {error}"
        ))
    })?;
    let spectrum_type = parse_spectrum_type(spectrum_type)?;

    splash_from_raw_peaks_with_type(peaks, spectrum_type).map_err(splash_error_to_value_error)
}

fn parse_spectrum_type(value: &str) -> PyResult<SplashSpectrumType> {
    match value {
        "mass" => Ok(SplashSpectrumType::MassSpectrum),
        "nmr" => Ok(SplashSpectrumType::Nmr),
        "uv" => Ok(SplashSpectrumType::Uv),
        "ir" => Ok(SplashSpectrumType::Ir),
        "raman" => Ok(SplashSpectrumType::Raman),
        _ => Err(PyValueError::new_err(format!(
            "unknown spectrum_type {value:?}; expected one of: mass, nmr, uv, ir, raman"
        ))),
    }
}

fn splash_error_to_value_error(error: SplashError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn mass_spectrometry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_splash, m)?)?;
    Ok(())
}
