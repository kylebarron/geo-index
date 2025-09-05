use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::ScalarBuffer;
use geo_index::rtree::RTreeIndex;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::rtree::input::PyRTreeRef;

/// Search an RTree given the provided bounding box.
///
/// Results are the indexes of the inserted objects in insertion order.
///
/// Args:
///     min_x: min x coordinate of bounding box
///     min_y: min y coordinate of bounding box
///     max_x: max x coordinate of bounding box
///     max_y: max y coordinate of bounding box
#[pyfunction]
pub fn search(
    py: Python,
    index: PyRTreeRef,
    min_x: Bound<PyAny>,
    min_y: Bound<PyAny>,
    max_x: Bound<PyAny>,
    max_y: Bound<PyAny>,
) -> PyResult<Py<PyAny>> {
    let results = match index {
        PyRTreeRef::Float32(tree) => tree.search(
            min_x.extract()?,
            min_y.extract()?,
            max_x.extract()?,
            max_y.extract()?,
        ),
        PyRTreeRef::Float64(tree) => tree.search(
            min_x.extract()?,
            min_y.extract()?,
            max_x.extract()?,
            max_y.extract()?,
        ),
    };
    let results = UInt32Array::new(ScalarBuffer::from(results), None);
    Ok(PyArray::from_array_ref(Arc::new(results))
        .to_arro3(py)?
        .unbind())
}
