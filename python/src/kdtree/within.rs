use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::ScalarBuffer;
use geo_index::kdtree::KDTreeIndex;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::kdtree::input::PyKDTreeRef;

#[pyfunction]
pub fn within(
    py: Python,
    index: PyKDTreeRef,
    qx: Bound<PyAny>,
    qy: Bound<PyAny>,
    r: Bound<PyAny>,
) -> PyResult<Py<PyAny>> {
    let results = match index {
        PyKDTreeRef::Float32(tree) => tree.within(qx.extract()?, qy.extract()?, r.extract()?),
        PyKDTreeRef::Float64(tree) => tree.within(qx.extract()?, qy.extract()?, r.extract()?),
    };
    let results = UInt32Array::new(ScalarBuffer::from(results), None);
    Ok(PyArray::from_array_ref(Arc::new(results))
        .to_arro3(py)?
        .unbind())
}
