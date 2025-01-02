use std::sync::Arc;

use arrow_array::{RecordBatch, UInt32Array};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field, Schema};
use geo_index::rtree::RTreeIndex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;

use crate::rtree::input::PyRTreeRef;

#[pyfunction]
pub fn tree_join(py: Python, left: PyRTreeRef, right: PyRTreeRef) -> PyResult<PyObject> {
    let (left_candidates, right_candidates): (Vec<u32>, Vec<u32>) = match (left, right) {
        (PyRTreeRef::Float32(left_tree), PyRTreeRef::Float32(right_tree)) => left_tree
            .intersection_candidates_with_other_tree(&right_tree)
            .unzip(),
        (PyRTreeRef::Float64(left_tree), PyRTreeRef::Float64(right_tree)) => left_tree
            .intersection_candidates_with_other_tree(&right_tree)
            .unzip(),
        _ => {
            return Err(PyValueError::new_err(
                "Both indexes must have the same coordinate type".to_string(),
            ))
        }
    };

    let left_results = Arc::new(UInt32Array::new(ScalarBuffer::from(left_candidates), None));
    let right_results = Arc::new(UInt32Array::new(ScalarBuffer::from(right_candidates), None));
    let fields = vec![
        Field::new("left", DataType::UInt32, false),
        Field::new("right", DataType::UInt32, false),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, vec![left_results, right_results]).unwrap();
    PyRecordBatch::new(batch).to_arro3(py)
}
