use std::sync::Arc;

use arrow_array::{StructArray, UInt32Array};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use geo_index::rtree::{RTreeIndex, RTreeRef};
use geo_index::CoordType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::buffer::PyArrowBuffer;
use pyo3_arrow::PyArray;

#[pyfunction]
pub fn intersection_candidates(
    py: Python,
    left: PyArrowBuffer,
    right: PyArrowBuffer,
) -> PyResult<PyObject> {
    let left = left.into_inner();
    let left = left.as_slice();
    let right = right.into_inner();
    let right = right.as_slice();

    let left_coord_type = CoordType::from_buffer(&left).unwrap();
    let right_coord_type = CoordType::from_buffer(&right).unwrap();

    if left_coord_type != right_coord_type {
        return Err(PyValueError::new_err(
            "Both indexes must have the same coordinate type".to_string(),
        ));
    }

    let (left_candidates, right_candidates): (Vec<u32>, Vec<u32>) = match left_coord_type {
        CoordType::Float32 => {
            let left_tree = RTreeRef::<f32>::try_new(&left).unwrap();
            let right_tree = RTreeRef::<f32>::try_new(&right).unwrap();
            left_tree
                .intersection_candidates_with_other_tree(&right_tree)
                .unzip()
        }
        CoordType::Float64 => {
            let left_tree = RTreeRef::<f64>::try_new(&left).unwrap();
            let right_tree = RTreeRef::<f64>::try_new(&right).unwrap();
            left_tree
                .intersection_candidates_with_other_tree(&right_tree)
                .unzip()
        }
        _ => todo!("Only f32 and f64 implemented so far"),
    };

    let left_results = Arc::new(UInt32Array::new(ScalarBuffer::from(left_candidates), None));
    let right_results = Arc::new(UInt32Array::new(ScalarBuffer::from(right_candidates), None));
    let fields = vec![
        Field::new("left", DataType::UInt32, false),
        Field::new("right", DataType::UInt32, false),
    ];
    let out = StructArray::new(fields.into(), vec![left_results, right_results], None);
    PyArray::from_array_ref(Arc::new(out)).to_arro3(py)
}
