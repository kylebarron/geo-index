use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::ScalarBuffer;
use geo_index::rtree::RTreeIndex;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::rtree::input::PyRTreeRef;

#[pyfunction]
#[pyo3(signature = (index, x, y, *, max_results = None, max_distance = None))]
pub fn neighbors(
    py: Python,
    index: PyRTreeRef,
    x: f64,
    y: f64,
    max_results: Option<usize>,
    max_distance: Option<f64>,
) -> PyResult<PyObject> {
    let results = match index {
        PyRTreeRef::Float32(tree) => tree.neighbors(
            x as f32,
            y as f32,
            max_results,
            max_distance.map(|x| x as f32),
        ),
        PyRTreeRef::Float64(tree) => tree.neighbors(x, y, max_results, max_distance),
    };
    let results = UInt32Array::new(ScalarBuffer::from(results), None);
    Ok(PyArray::from_array_ref(Arc::new(results))
        .to_arro3(py)?
        .unbind())
}
