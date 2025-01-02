use arrow_array::types::{Float32Type, Float64Type};
use geo_index::rtree::RTreeIndex;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::rtree::input::PyRTreeRef;
use crate::util::boxes_to_arrow;

#[pyfunction]
pub fn boxes_at_level(py: Python, index: PyRTreeRef, level: usize) -> PyResult<PyObject> {
    let array = match index {
        PyRTreeRef::Float32(tree) => {
            let boxes = tree
                .boxes_at_level(level)
                .map_err(|err| PyIndexError::new_err(err.to_string()))?;
            boxes_to_arrow::<Float32Type>(boxes, tree.buffer().clone())
        }
        PyRTreeRef::Float64(tree) => {
            let boxes = tree
                .boxes_at_level(level)
                .map_err(|err| PyIndexError::new_err(err.to_string()))?;
            boxes_to_arrow::<Float64Type>(boxes, tree.buffer().clone())
        }
    };
    PyArray::from_array_ref(array).to_arro3(py)
}
