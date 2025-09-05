use std::sync::Arc;

use arrow_buffer::Buffer;
use geo_index::indices::Indices;
use geo_index::kdtree::{KDTreeIndex, KDTreeMetadata};
use geo_index::{CoordType, IndexableNum};
use pyo3::prelude::*;
use pyo3_arrow::buffer::PyArrowBuffer;

/// A KDTree on external memory
#[derive(Debug, Clone)]
pub(crate) struct ExternalKDTree<N: IndexableNum> {
    buffer: Arc<Buffer>,
    metadata: KDTreeMetadata<N>,
}

impl<N: IndexableNum> KDTreeIndex<N> for ExternalKDTree<N> {
    fn coords(&self) -> &[N] {
        self.metadata.coords_slice(&self.buffer)
    }

    fn indices(&self) -> Indices<'_> {
        self.metadata.indices_slice(&self.buffer)
    }

    fn metadata(&self) -> &KDTreeMetadata<N> {
        &self.metadata
    }
}

impl<N: IndexableNum> ExternalKDTree<N> {
    #[allow(dead_code)]
    pub(crate) fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }
}

pub(crate) enum PyKDTreeRef {
    Float32(ExternalKDTree<f32>),
    Float64(ExternalKDTree<f64>),
}

impl<'py> FromPyObject<'py> for PyKDTreeRef {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let buffer = ob.extract::<PyArrowBuffer>()?.into_inner();
        let owner = Arc::new(buffer.clone());
        let slice = buffer.as_slice();
        let coord_type = CoordType::from_buffer(&slice).unwrap();
        match coord_type {
            CoordType::Float32 => Ok(Self::Float32(ExternalKDTree {
                buffer: owner,
                metadata: KDTreeMetadata::from_slice(slice).unwrap(),
            })),
            CoordType::Float64 => Ok(Self::Float64(ExternalKDTree {
                buffer: owner,
                metadata: KDTreeMetadata::from_slice(slice).unwrap(),
            })),
            _ => todo!("Only f32 and f64 implemented so far"),
        }
    }
}
