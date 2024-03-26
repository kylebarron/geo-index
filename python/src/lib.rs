pub mod kdtree;
pub mod rtree;

use pyo3::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyfunction]
fn ___version() -> &'static str {
    VERSION
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(___version))?;

    m.add_class::<rtree::RTree>()?;
    m.add_class::<kdtree::KDTree>()?;

    Ok(())
}
