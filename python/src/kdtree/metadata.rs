use geo_index::kdtree::{KDTreeIndex, KDTreeMetadata, DEFAULT_KDTREE_NODE_SIZE};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::coord_type::CoordType;
use crate::kdtree::input::PyKDTreeRef;

pub(crate) enum PyKDTreeMetadataInner {
    Float32(KDTreeMetadata<f32>),
    Float64(KDTreeMetadata<f64>),
}

impl PyKDTreeMetadataInner {
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

    fn num_bytes(&self) -> usize {
        match self {
            Self::Float32(meta) => meta.data_buffer_length(),
            Self::Float64(meta) => meta.data_buffer_length(),
        }
    }
}

#[pyclass(name = "KDTreeMetadata", frozen)]
pub struct PyKDTreeMetadata(PyKDTreeMetadataInner);

#[pymethods]
impl PyKDTreeMetadata {
    #[new]
    #[pyo3(signature = (num_items, node_size = DEFAULT_KDTREE_NODE_SIZE, coord_type = None))]
    fn new(num_items: u32, node_size: u16, coord_type: Option<CoordType>) -> Self {
        let coord_type = coord_type.unwrap_or(CoordType::Float64);
        match coord_type {
            CoordType::Float32 => Self(PyKDTreeMetadataInner::Float32(KDTreeMetadata::<f32>::new(
                num_items, node_size,
            ))),
            CoordType::Float64 => Self(PyKDTreeMetadataInner::Float64(KDTreeMetadata::<f64>::new(
                num_items, node_size,
            ))),
        }
    }

    #[classmethod]
    fn from_index(_cls: &Bound<PyType>, index: PyKDTreeRef) -> PyResult<Self> {
        match index {
            PyKDTreeRef::Float32(tree) => {
                Ok(Self(PyKDTreeMetadataInner::Float32(*tree.metadata())))
            }
            PyKDTreeRef::Float64(tree) => {
                Ok(Self(PyKDTreeMetadataInner::Float64(*tree.metadata())))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "KDTreeMetadata(num_items={}, node_size={})",
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
    fn num_bytes(&self) -> usize {
        self.0.num_bytes()
    }
}

impl From<KDTreeMetadata<f32>> for PyKDTreeMetadata {
    fn from(value: KDTreeMetadata<f32>) -> Self {
        Self(PyKDTreeMetadataInner::Float32(value))
    }
}

impl From<KDTreeMetadata<f64>> for PyKDTreeMetadata {
    fn from(value: KDTreeMetadata<f64>) -> Self {
        Self(PyKDTreeMetadataInner::Float64(value))
    }
}
