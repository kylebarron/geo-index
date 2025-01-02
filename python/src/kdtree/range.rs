use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::ScalarBuffer;
use geo_index::kdtree::KDTreeIndex;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::kdtree::input::PyKDTreeRef;

#[pyfunction]
pub fn range(
    py: Python,
    index: PyKDTreeRef,
    min_x: Bound<PyAny>,
    min_y: Bound<PyAny>,
    max_x: Bound<PyAny>,
    max_y: Bound<PyAny>,
) -> PyResult<PyObject> {
    let results = match index {
        PyKDTreeRef::Float32(tree) => tree.range(
            min_x.extract()?,
            min_y.extract()?,
            max_x.extract()?,
            max_y.extract()?,
        ),
        PyKDTreeRef::Float64(tree) => tree.range(
            min_x.extract()?,
            min_y.extract()?,
            max_x.extract()?,
            max_y.extract()?,
        ),
    };
    let results = UInt32Array::new(ScalarBuffer::from(results), None);
    PyArray::from_array_ref(Arc::new(results)).to_arro3(py)
}
