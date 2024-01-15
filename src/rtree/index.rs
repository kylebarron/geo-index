use std::marker::PhantomData;

use bytemuck::cast_slice;

use crate::error::GeoIndexError;
use crate::indices::Indices;
use crate::r#type::IndexableNum;
use crate::rtree::constants::VERSION;
use crate::rtree::r#trait::RTreeIndex;
use crate::rtree::util::compute_num_nodes;

/// An owned RTree buffer.
///
/// Usually this will be created from scratch via [`RTreeBuilder`][crate::rtree::RTreeBuilder].
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedRTree<N: IndexableNum> {
    pub(crate) buffer: Vec<u8>,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_bounds: Vec<usize>,
    pub(crate) phantom: PhantomData<N>,
}

impl<N: IndexableNum> OwnedRTree<N> {
    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }

    pub fn as_ref(&self) -> RTreeRef<N> {
        RTreeRef {
            boxes: self.boxes(),
            indices: self.indices().into_owned(),
            node_size: self.node_size,
            num_items: self.num_items,
            num_nodes: self.num_nodes,
            level_bounds: self.level_bounds.clone(),
        }
    }
}

impl<N: IndexableNum> AsRef<[u8]> for OwnedRTree<N> {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

/// A reference on an external RTree buffer.
///
/// Usually this will be created from an [`OwnedRTree`] via its [`as_ref`][OwnedRTree::as_ref]
/// method, but it can also be created from any existing data buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct RTreeRef<'a, N: IndexableNum> {
    pub(crate) boxes: &'a [N],
    pub(crate) indices: Indices<'a>,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_bounds: Vec<usize>,
}

impl<'a, N: IndexableNum> RTreeRef<'a, N> {
    pub fn try_new<T: AsRef<[u8]>>(data: &'a T) -> Result<Self, GeoIndexError> {
        let data = data.as_ref();
        // TODO: validate length of slice?

        let magic = data[0];
        if magic != 0xfb {
            return Err(GeoIndexError::General(
                "Data not in Flatbush format.".to_string(),
            ));
        }

        let version_and_type = data[1];
        let version = version_and_type >> 4;
        if version != VERSION {
            return Err(GeoIndexError::General(
                format!("Got v{} data when expected v{}.", version, VERSION).to_string(),
            ));
        }

        let type_ = version_and_type & 0x0f;
        if type_ != N::TYPE_INDEX {
            return Err(GeoIndexError::General(
                format!(
                    "Got type {} data when expected type {}.",
                    type_,
                    N::TYPE_INDEX
                )
                .to_string(),
            ));
        }

        let node_size: u16 = cast_slice(&data[2..4])[0];
        let num_items: u32 = cast_slice(&data[4..8])[0];
        let node_size = node_size as usize;
        let num_items = num_items as usize;

        let (num_nodes, level_bounds) = compute_num_nodes(num_items, node_size);

        let indices_bytes_per_element = if num_nodes < 16384 { 2 } else { 4 };
        let nodes_byte_length = num_nodes * 4 * N::BYTES_PER_ELEMENT;
        let indices_byte_length = num_nodes * indices_bytes_per_element;

        // TODO: assert length of `data` matches expected
        let boxes = cast_slice(&data[8..8 + nodes_byte_length]);
        let indices_buf = &data[8 + nodes_byte_length..8 + nodes_byte_length + indices_byte_length];
        let indices = Indices::new(indices_buf, num_nodes);

        Ok(Self {
            boxes,
            indices,
            node_size,
            num_items,
            num_nodes,
            level_bounds,
        })
    }
}
