use std::borrow::Cow;
use std::marker::PhantomData;

use bytemuck::cast_slice;

use crate::error::{GeoIndexError, Result};
use crate::indices::Indices;
use crate::r#type::IndexableNum;
use crate::rtree::constants::VERSION;
use crate::rtree::r#trait::RTreeIndex;
use crate::rtree::util::compute_num_nodes;

/// Common metadata to describe a tree
#[derive(Debug, Clone, PartialEq)]
pub struct TreeMetadata<N: IndexableNum> {
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) num_nodes: usize,
    pub(crate) level_bounds: Vec<usize>,
    nodes_byte_length: usize,
    indices_byte_length: usize,
    phantom: PhantomData<N>,
}

impl<N: IndexableNum> TreeMetadata<N> {
    pub fn try_new(data: &[u8]) -> Result<Self> {
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

        let total_byte_length = 8 + nodes_byte_length + indices_byte_length;
        if data.len() != total_byte_length {
            return Err(GeoIndexError::General(format!(
                "Incorrect buffer length. Expected {} got {}.",
                total_byte_length,
                data.len()
            )));
        }

        Ok(Self {
            node_size,
            num_items,
            num_nodes,
            level_bounds,
            nodes_byte_length,
            indices_byte_length,
            phantom: PhantomData,
        })
    }

    pub(crate) unsafe fn new_unchecked(
        node_size: usize,
        num_items: usize,
        num_nodes: usize,
        level_bounds: Vec<usize>,
    ) -> Self {
        let indices_bytes_per_element = if num_nodes < 16384 { 2 } else { 4 };
        let nodes_byte_length = num_nodes * 4 * N::BYTES_PER_ELEMENT;
        let indices_byte_length = num_nodes * indices_bytes_per_element;

        Self {
            node_size,
            num_items,
            num_nodes,
            level_bounds,
            nodes_byte_length,
            indices_byte_length,
            phantom: PhantomData,
        }
    }

    pub fn boxes_slice<'a>(&self, data: &'a [u8]) -> &'a [N] {
        cast_slice(&data[8..8 + self.nodes_byte_length])
    }

    pub fn indices_slice<'a>(&self, data: &'a [u8]) -> Indices<'a> {
        let indices_buf = &data
            [8 + self.nodes_byte_length..8 + self.nodes_byte_length + self.indices_byte_length];
        Indices::new(indices_buf, self.num_nodes)
    }

    pub fn node_size(&self) -> usize {
        self.node_size
    }
    pub fn num_items(&self) -> usize {
        self.num_items
    }
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }
    pub fn level_bounds(&self) -> &[usize] {
        self.level_bounds.as_slice()
    }
}

/// An owned RTree buffer.
///
/// Usually this will be created from scratch via [`RTreeBuilder`][crate::rtree::RTreeBuilder].
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedRTree<N: IndexableNum> {
    pub(crate) buffer: Vec<u8>,
    pub(crate) metadata: TreeMetadata<N>,
}

impl<N: IndexableNum> OwnedRTree<N> {
    pub fn try_new(buffer: Vec<u8>) -> Result<Self> {
        let metadata = TreeMetadata::try_new(&buffer)?;
        Ok(Self { buffer, metadata })
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }

    pub fn as_rtree_ref(&self) -> RTreeRef<N> {
        RTreeRef {
            boxes: self.boxes(),
            indices: self.indices(),
            metadata: Cow::Borrowed(&self.metadata),
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
    pub(crate) metadata: Cow<'a, TreeMetadata<N>>,
}

impl<'a, N: IndexableNum> RTreeRef<'a, N> {
    pub fn try_new<T: AsRef<[u8]>>(data: &'a T) -> Result<Self> {
        let data = data.as_ref();
        let metadata = TreeMetadata::try_new(data)?;
        let boxes = metadata.boxes_slice(data);
        let indices = metadata.indices_slice(data);

        Ok(Self {
            boxes,
            indices,
            metadata: Cow::Owned(metadata),
        })
    }
}
