use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::ScalarBuffer;
use geo_index::rtree::{RTreeIndex, RTreeRef};
use geo_index::CoordType;
use pyo3::prelude::*;
use pyo3_arrow::buffer::PyArrowBuffer;
use pyo3_arrow::PyArray;

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
    index: PyArrowBuffer,
    min_x: Bound<PyAny>,
    min_y: Bound<PyAny>,
    max_x: Bound<PyAny>,
    max_y: Bound<PyAny>,
) -> PyResult<PyObject> {
    let buffer = index.into_inner();
    let slice = buffer.as_slice();
    let coord_type = CoordType::from_buffer(&slice).unwrap();
    let results = match coord_type {
        CoordType::Float32 => {
            let tree = RTreeRef::<f32>::try_new(&slice).unwrap();
            tree.search(
                min_x.extract()?,
                min_y.extract()?,
                max_x.extract()?,
                max_y.extract()?,
            )
        }
        CoordType::Float64 => {
            let tree = RTreeRef::<f64>::try_new(&slice).unwrap();
            tree.search(
                min_x.extract()?,
                min_y.extract()?,
                max_x.extract()?,
                max_y.extract()?,
            )
        }
        _ => todo!("Only f32 and f64 implemented so far"),
    };
    let results = UInt32Array::new(ScalarBuffer::from(results), None);
    PyArray::from_array_ref(Arc::new(results)).to_arro3(py)
}
