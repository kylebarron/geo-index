mod builder;
pub(crate) mod intersection;
mod metadata;
pub(crate) mod search;

use pyo3::intern;
use pyo3::prelude::*;

// https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
// https://github.com/PyO3/pyo3/issues/759#issuecomment-977835119
pub fn register_rtree_module(
    py: Python<'_>,
    parent_module: &Bound<'_, PyModule>,
    parent_module_str: &str,
) -> PyResult<()> {
    let full_module_string = format!("{}.rtree", parent_module_str);

    let child_module = PyModule::new(parent_module.py(), "rtree")?;

    child_module.add_class::<builder::RTree>()?;
    child_module.add_class::<builder::RTreeBuilder>()?;
    child_module.add_class::<metadata::RTreeMetadata>()?;
    child_module.add_wrapped(wrap_pyfunction!(search::search))?;
    child_module.add_wrapped(wrap_pyfunction!(intersection::intersection_candidates))?;

    parent_module.add_submodule(&child_module)?;

    py.import(intern!(py, "sys"))?
        .getattr(intern!(py, "modules"))?
        .set_item(full_module_string.as_str(), &child_module)?;

    // needs to be set *after* `add_submodule()`
    child_module.setattr("__name__", full_module_string)?;

    Ok(())
}
