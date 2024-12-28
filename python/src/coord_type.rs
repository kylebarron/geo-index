use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

/// We only support Float32 and Float64 for now
pub enum CoordType {
    Float32,
    Float64,
}

// TODO: also check for numpy dtype here

impl<'a> FromPyObject<'a> for CoordType {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s = ob.extract::<PyBackedStr>()?;
        match s.to_lowercase().as_str() {
            "float32" | "f32" => Ok(Self::Float32),
            "float64" | "f64" => Ok(Self::Float64),
            _ => Err(PyValueError::new_err(
                "Unexpected coordinate type. Should be one of 'float32' or 'float64'.",
            )),
        }
    }
}

impl From<CoordType> for geo_index::CoordType {
    fn from(value: CoordType) -> Self {
        match value {
            CoordType::Float32 => geo_index::CoordType::Float32,
            CoordType::Float64 => geo_index::CoordType::Float64,
        }
    }
}
