use std::sync::Arc;

use arrow_buffer::Buffer;
use geo_index::indices::Indices;
use geo_index::rtree::{RTreeIndex, RTreeMetadata};
use geo_index::{CoordType, IndexableNum};
use pyo3::prelude::*;
use pyo3_arrow::buffer::PyArrowBuffer;

/// An RTree on external memory
#[derive(Debug, Clone)]
pub(crate) struct ExternalRTree<N: IndexableNum> {
    buffer: Arc<Buffer>,
    metadata: RTreeMetadata<N>,
}

impl<N: IndexableNum> RTreeIndex<N> for ExternalRTree<N> {
    fn boxes(&self) -> &[N] {
        self.metadata.boxes_slice(&self.buffer)
    }

    fn indices(&self) -> Indices<'_> {
        self.metadata.indices_slice(&self.buffer)
    }

    fn metadata(&self) -> &RTreeMetadata<N> {
        &self.metadata
    }
}

impl<N: IndexableNum> ExternalRTree<N> {
    pub(crate) fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }
}

pub(crate) enum PyRTreeRef {
    Float32(ExternalRTree<f32>),
    Float64(ExternalRTree<f64>),
}

impl<'py> FromPyObject<'py> for PyRTreeRef {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let buffer = ob.extract::<PyArrowBuffer>()?.into_inner();
        let owner = Arc::new(buffer.clone());
        let slice = buffer.as_slice();
        let coord_type = CoordType::from_buffer(&slice).unwrap();
        match coord_type {
            CoordType::Float32 => Ok(Self::Float32(ExternalRTree {
                buffer: owner,
                metadata: RTreeMetadata::from_slice(slice).unwrap(),
            })),
            CoordType::Float64 => Ok(Self::Float64(ExternalRTree {
                buffer: owner,
                metadata: RTreeMetadata::from_slice(slice).unwrap(),
            })),
            _ => todo!("Only f32 and f64 implemented so far"),
        }
    }
}
