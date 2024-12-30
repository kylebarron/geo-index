use std::sync::Arc;

use arrow_array::builder::{UInt16Builder, UInt32Builder};
use arrow_array::types::{UInt16Type, UInt32Type};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_buffer::alloc::Allocation;
use arrow_schema::{Field, Schema};
use geo_index::indices::Indices;
use geo_index::rtree::{RTreeIndex, RTreeRef};
use geo_index::CoordType;
use pyo3::prelude::*;
use pyo3_arrow::buffer::PyArrowBuffer;
use pyo3_arrow::PyRecordBatch;

use crate::util::slice_to_arrow;

#[pyfunction]
pub fn partitions(py: Python, index: PyArrowBuffer) -> PyResult<PyObject> {
    let buffer = index.into_inner();
    let owner = Arc::new(buffer.clone());
    let slice = buffer.as_slice();
    let coord_type = CoordType::from_buffer(&slice).unwrap();
    let (indices, partition_ids) = match coord_type {
        CoordType::Float32 => {
            let tree = RTreeRef::<f32>::try_new(&slice).unwrap();
            let indices = indices_to_arrow(tree.indices(), tree.num_items(), owner);
            let partition_ids = partition_id_array(tree.num_items(), tree.node_size());
            (indices, partition_ids)
        }
        CoordType::Float64 => {
            let tree = RTreeRef::<f64>::try_new(&slice).unwrap();
            let indices = indices_to_arrow(tree.indices(), tree.num_items(), owner);
            let partition_ids = partition_id_array(tree.num_items(), tree.node_size());
            (indices, partition_ids)
        }
        _ => todo!("Only f32 and f64 implemented so far"),
    };

    let fields = vec![
        Field::new("indices", indices.data_type().clone(), false),
        Field::new("partition_id", partition_ids.data_type().clone(), false),
    ];
    let schema = Schema::new(fields);
    PyRecordBatch::new(RecordBatch::try_new(schema.into(), vec![indices, partition_ids]).unwrap())
        .to_arro3(py)
}

fn indices_to_arrow(indices: Indices, num_items: u32, owner: Arc<dyn Allocation>) -> ArrayRef {
    match indices {
        Indices::U16(slice) => slice_to_arrow::<UInt16Type>(&slice[0..num_items as usize], owner),
        Indices::U32(slice) => slice_to_arrow::<UInt32Type>(&slice[0..num_items as usize], owner),
    }
}

fn partition_id_array(num_items: u32, node_size: u16) -> ArrayRef {
    let num_full_nodes = num_items / node_size as u32;
    let remainder = num_items % node_size as u32;

    // Check if the partition ids fit inside a u16
    // We add 1 to cover the remainder
    if num_full_nodes + 1 < u16::MAX as _ {
        let mut output_array = UInt16Builder::with_capacity(num_items as _);

        let mut partition_id = 0;
        for _ in 0..num_full_nodes {
            output_array.append_value_n(partition_id, node_size as usize);
            partition_id += 1;
        }

        // The loop omits the last node
        output_array.append_value_n(partition_id, remainder as usize);

        Arc::new(output_array.finish())
    } else {
        let mut output_array = UInt32Builder::with_capacity(num_items as _);

        let mut partition_id = 0;
        for _ in 0..num_full_nodes {
            output_array.append_value_n(partition_id, node_size as usize);
            partition_id += 1;
        }

        // The loop omits the last node
        output_array.append_value_n(partition_id, remainder as usize);

        Arc::new(output_array.finish())
    }
}
