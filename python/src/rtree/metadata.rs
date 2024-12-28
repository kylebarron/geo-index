use geo_index::rtree::DEFAULT_RTREE_NODE_SIZE;
use pyo3::prelude::*;

use crate::coord_type::CoordType;

enum RTreeMetadataInner {
    Float32(geo_index::rtree::RTreeMetadata<f32>),
    Float64(geo_index::rtree::RTreeMetadata<f64>),
}

#[pyclass]
pub struct RTreeMetadata(RTreeMetadataInner);

#[pymethods]
impl RTreeMetadata {
    #[new]
    #[pyo3(signature = (num_items, node_size = DEFAULT_RTREE_NODE_SIZE, coord_type = None))]
    fn new(num_items: u32, node_size: u16, coord_type: Option<CoordType>) -> Self {
        let coord_type = coord_type.unwrap_or(CoordType::Float64);
        match coord_type {
            CoordType::Float32 => Self(RTreeMetadataInner::Float32(
                geo_index::rtree::RTreeMetadata::<f32>::new(num_items, node_size),
            )),
            CoordType::Float64 => Self(RTreeMetadataInner::Float64(
                geo_index::rtree::RTreeMetadata::<f64>::new(num_items, node_size),
            )),
        }
    }

    #[getter]
    fn node_size(&self) -> u16 {
        match &self.0 {
            RTreeMetadataInner::Float32(meta) => meta.node_size(),
            RTreeMetadataInner::Float64(meta) => meta.node_size(),
        }
    }

    #[getter]
    fn num_items(&self) -> u32 {
        match &self.0 {
            RTreeMetadataInner::Float32(meta) => meta.num_items(),
            RTreeMetadataInner::Float64(meta) => meta.num_items(),
        }
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.0 {
            RTreeMetadataInner::Float32(meta) => meta.num_nodes(),
            RTreeMetadataInner::Float64(meta) => meta.num_nodes(),
        }
    }

    #[getter]
    fn level_bounds(&self) -> Vec<usize> {
        match &self.0 {
            RTreeMetadataInner::Float32(meta) => meta.level_bounds().to_vec(),
            RTreeMetadataInner::Float64(meta) => meta.level_bounds().to_vec(),
        }
    }

    #[getter]
    fn data_buffer_length(&self) -> usize {
        match &self.0 {
            RTreeMetadataInner::Float32(meta) => meta.data_buffer_length(),
            RTreeMetadataInner::Float64(meta) => meta.data_buffer_length(),
        }
    }
}
