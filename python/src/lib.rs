pub mod kdtree;
pub mod rtree;

use pyo3::exceptions::PyRuntimeWarning;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyfunction]
fn ___version() -> &'static str {
    VERSION
}

/// Raise RuntimeWarning for debug builds
#[pyfunction]
fn check_debug_build(py: Python) -> PyResult<()> {
    #[cfg(debug_assertions)]
    {
        let warnings_mod = py.import_bound(intern!(py, "warnings"))?;
        let warning = PyRuntimeWarning::new_err(
            "geoindex-rs has not been compiled in release mode. Performance will be degraded.",
        );
        let args = PyTuple::new_bound(py, vec![warning.into_py(py)]);
        warnings_mod.call_method1(intern!(py, "warn"), args)?;
    }

    Ok(())
}

#[pymodule]
fn _rust(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    check_debug_build(py)?;

    m.add_wrapped(wrap_pyfunction!(___version))?;

    m.add_class::<rtree::RTree>()?;
    m.add_class::<kdtree::KDTree>()?;

    Ok(())
}
