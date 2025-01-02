use geo_index::rtree::{RTreeIndex, RTreeMetadata, DEFAULT_RTREE_NODE_SIZE};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::coord_type::CoordType;
use crate::rtree::input::PyRTreeRef;

pub(crate) enum PyRTreeMetadataInner {
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
    fn from_index(_cls: &Bound<PyType>, index: PyRTreeRef) -> PyResult<Self> {
        match index {
            PyRTreeRef::Float32(tree) => {
                Ok(Self(PyRTreeMetadataInner::Float32(tree.metadata().clone())))
            }
            PyRTreeRef::Float64(tree) => {
                Ok(Self(PyRTreeMetadataInner::Float64(tree.metadata().clone())))
            }
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
    fn num_levels(&self) -> usize {
        self.0.level_bounds().len()
    }

    #[getter]
    fn data_buffer_length(&self) -> usize {
        self.0.data_buffer_length()
    }
}

impl From<RTreeMetadata<f32>> for PyRTreeMetadata {
    fn from(value: RTreeMetadata<f32>) -> Self {
        Self(PyRTreeMetadataInner::Float32(value))
    }
}

impl From<RTreeMetadata<f64>> for PyRTreeMetadata {
    fn from(value: RTreeMetadata<f64>) -> Self {
        Self(PyRTreeMetadataInner::Float64(value))
    }
}
