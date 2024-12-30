use geo_index::rtree::{RTreeIndex, RTreeMetadata, RTreeRef, DEFAULT_RTREE_NODE_SIZE};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_arrow::buffer::PyArrowBuffer;

use crate::coord_type::CoordType;

enum PyRTreeMetadataInner {
    Float32(RTreeMetadata<f32>),
    Float64(RTreeMetadata<f64>),
}

impl PyRTreeMetadataInner {
    fn node_size(&self) -> u16 {
        match self {
            Self::Float32(meta) => meta.node_size(),
            Self::Float64(meta) => meta.node_size(),
        }
    }

    fn num_items(&self) -> u32 {
        match self {
            Self::Float32(meta) => meta.num_items(),
            Self::Float64(meta) => meta.num_items(),
        }
    }

    fn num_nodes(&self) -> usize {
        match self {
            Self::Float32(meta) => meta.num_nodes(),
            Self::Float64(meta) => meta.num_nodes(),
        }
    }

    fn level_bounds(&self) -> Vec<usize> {
        match self {
            Self::Float32(meta) => meta.level_bounds().to_vec(),
            Self::Float64(meta) => meta.level_bounds().to_vec(),
        }
    }

    fn data_buffer_length(&self) -> usize {
        match self {
            Self::Float32(meta) => meta.data_buffer_length(),
            Self::Float64(meta) => meta.data_buffer_length(),
        }
    }
}

#[pyclass(name = "RTreeMetadata")]
pub struct PyRTreeMetadata(PyRTreeMetadataInner);

#[pymethods]
impl PyRTreeMetadata {
    #[new]
    #[pyo3(signature = (num_items, node_size = DEFAULT_RTREE_NODE_SIZE, coord_type = None))]
    fn new(num_items: u32, node_size: u16, coord_type: Option<CoordType>) -> Self {
        let coord_type = coord_type.unwrap_or(CoordType::Float64);
        match coord_type {
            CoordType::Float32 => Self(PyRTreeMetadataInner::Float32(
                geo_index::rtree::RTreeMetadata::<f32>::new(num_items, node_size),
            )),
            CoordType::Float64 => Self(PyRTreeMetadataInner::Float64(
                geo_index::rtree::RTreeMetadata::<f64>::new(num_items, node_size),
            )),
        }
    }

    #[classmethod]
    fn from_index(_cls: &Bound<PyType>, index: PyArrowBuffer) -> PyResult<Self> {
        let buffer = index.into_inner();
        let slice = buffer.as_slice();
        let coord_type = geo_index::CoordType::from_buffer(&slice).unwrap();
        match coord_type {
            geo_index::CoordType::Float32 => {
                let tree = RTreeRef::<f32>::try_new(&slice).unwrap();
                Ok(Self(PyRTreeMetadataInner::Float32(tree.metadata().clone())))
            }
            geo_index::CoordType::Float64 => {
                let tree = RTreeRef::<f64>::try_new(&slice).unwrap();
                Ok(Self(PyRTreeMetadataInner::Float64(tree.metadata().clone())))
            }
            _ => todo!("Only f32 and f64 implemented so far"),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RTreeMetadata(num_items={}, node_size={})",
            self.0.num_items(),
            self.0.node_size()
        )
    }

    #[getter]
    fn node_size(&self) -> u16 {
        self.0.node_size()
    }

    #[getter]
    fn num_items(&self) -> u32 {
        self.0.num_items()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.0.num_nodes()
    }

    #[getter]
    fn level_bounds(&self) -> Vec<usize> {
        self.0.level_bounds()
    }

    #[getter]
    fn data_buffer_length(&self) -> usize {
        self.0.data_buffer_length()
    }
}
