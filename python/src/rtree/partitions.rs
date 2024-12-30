use std::sync::Arc;

use arrow_array::{ArrayRef, UInt32Array};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use geo_index::indices::Indices;
use geo_index::rtree::{RTreeIndex, RTreeRef};
use geo_index::CoordType;
use pyo3::prelude::*;
use pyo3_arrow::buffer::PyArrowBuffer;
use pyo3_arrow::PyChunkedArray;

#[pyfunction]
pub fn partitions(py: Python, index: PyArrowBuffer) -> PyResult<PyObject> {
    let buffer = index.into_inner();
    let slice = buffer.as_slice();
    let coord_type = CoordType::from_buffer(&slice).unwrap();
    let result = match coord_type {
        CoordType::Float32 => {
            let tree = RTreeRef::<f32>::try_new(&slice).unwrap();
            let node_size = tree.node_size();
            match tree.indices() {
                Indices::U16(indices) => indices_to_chunked_array(indices, node_size),
                Indices::U32(indices) => indices_to_chunked_array_u32(indices, node_size),
            }
        }
        CoordType::Float64 => {
            let tree = RTreeRef::<f64>::try_new(&slice).unwrap();
            let node_size = tree.node_size();
            match tree.indices() {
                Indices::U16(indices) => indices_to_chunked_array(indices, node_size),
                Indices::U32(indices) => indices_to_chunked_array_u32(indices, node_size),
            }
        }
        _ => todo!("Only f32 and f64 implemented so far"),
    };
    result.to_arro3(py)
}

fn indices_to_chunked_array(indices: &[u16], node_size: u16) -> PyChunkedArray {
    let array_chunks = indices
        .chunks(node_size as usize)
        .map(|chunk| {
            Arc::new(UInt32Array::new(
                ScalarBuffer::from(Vec::from_iter(chunk.iter().map(|x| *x as u32))),
                None,
            )) as ArrayRef
        })
        .collect::<Vec<_>>();
    PyChunkedArray::try_new(
        array_chunks,
        Arc::new(Field::new("indices", DataType::UInt32, false)),
    )
    .unwrap()
}

fn indices_to_chunked_array_u32(indices: &[u32], node_size: u16) -> PyChunkedArray {
    let array_chunks = indices
        .chunks(node_size as usize)
        .map(|chunk| {
            Arc::new(UInt32Array::new(ScalarBuffer::from(chunk.to_vec()), None)) as ArrayRef
        })
        .collect::<Vec<_>>();
    PyChunkedArray::try_new(
        array_chunks,
        Arc::new(Field::new("indices", DataType::UInt32, false)),
    )
    .unwrap()
}
