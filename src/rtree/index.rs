use std::marker::PhantomData;

use bytemuck::cast_slice;

use crate::error::{GeoIndexError, Result};
use crate::indices::Indices;
use crate::r#type::IndexableNum;
use crate::rtree::constants::VERSION;
use crate::rtree::util::compute_num_nodes;

/// Common metadata to describe an RTree
///
/// You can use the metadata to infer the total byte size of a tree given the provided criteria.
/// See [`data_buffer_length`][Self::data_buffer_length].
#[derive(Debug, Clone, PartialEq)]
pub struct RTreeMetadata<N: IndexableNum> {
    node_size: u16,
    num_items: u32,
    num_nodes: usize,
    level_bounds: Vec<usize>,
    pub(crate) nodes_byte_length: usize,
    pub(crate) indices_byte_length: usize,
    phantom: PhantomData<N>,
}

impl<N: IndexableNum> RTreeMetadata<N> {
    /// Construct a new [`RTreeMetadata`] from a number of items and node size.
    pub fn new(num_items: u32, node_size: u16) -> Self {
        assert!((2..=65535).contains(&node_size));

        let (num_nodes, level_bounds) = compute_num_nodes(num_items, node_size);

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

    /// Construct a new [`RTreeMetadata`] from an existing byte slice conforming to the "flatbush
    /// ABI", such as what [`RTreeBuilder`] generates.
    pub fn from_slice(data: &[u8]) -> Result<Self> {
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
                format!("Got v{version} data when expected v{VERSION}.").to_string(),
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

        let slf = Self::new(num_items, node_size);
        if slf.data_buffer_length() != data.len() {
            return Err(GeoIndexError::General(format!(
                "Expected {} bytes but received byte slice with {} bytes",
                slf.data_buffer_length(),
                data.len()
            )));
        }

        Ok(slf)
    }

    /// The maximum number of items per node.
    pub fn node_size(&self) -> u16 {
        self.node_size
    }

    /// The number of items indexed in the tree.
    pub fn num_items(&self) -> u32 {
        self.num_items
    }

    /// The total number of nodes at all levels in the tree.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// The offsets into [`RTreeIndex::boxes`][crate::rtree::RTreeIndex::boxes] where each level's
    /// boxes starts and ends. The tree is laid out bottom-up, and there's an implicit initial 0.
    /// So the boxes of the lowest level of the tree are located from
    /// `boxes[0..self.level_bounds()[0]]`.
    pub fn level_bounds(&self) -> &[usize] {
        &self.level_bounds
    }

    /// The number of bytes that an RTree with this metadata would have.
    ///
    /// ```
    /// use geo_index::rtree::RTreeMetadata;
    ///
    /// let metadata = RTreeMetadata::<f64>::new(25000, 16);
    /// assert_eq!(metadata.data_buffer_length(), 960_092);
    /// ```
    pub fn data_buffer_length(&self) -> usize {
        8 + self.nodes_byte_length + self.indices_byte_length
    }

    /// Access the slice of boxes from the data buffer this metadata represents.
    pub fn boxes_slice<'a>(&self, data: &'a [u8]) -> &'a [N] {
        cast_slice(&data[8..8 + self.nodes_byte_length])
    }

    /// Access the slice of indices from the data buffer this metadata represents.
    pub fn indices_slice<'a>(&self, data: &'a [u8]) -> Indices<'a> {
        let indices_buf = &data
            [8 + self.nodes_byte_length..8 + self.nodes_byte_length + self.indices_byte_length];
        Indices::new(indices_buf, self.num_nodes)
    }
}

/// An owned RTree buffer.
///
/// Usually this will be created from scratch via [`RTreeBuilder`][crate::rtree::RTreeBuilder].
#[derive(Debug, Clone, PartialEq)]
pub struct RTree<N: IndexableNum> {
    pub(crate) buffer: Vec<u8>,
    pub(crate) metadata: RTreeMetadata<N>,
}

impl<N: IndexableNum> RTree<N> {
    /// Access the underlying buffer of this RTree.
    ///
    /// This buffer can then be persisted and passed to `RTreeRef::try_new`.
    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }
}

impl<N: IndexableNum> AsRef<[u8]> for RTree<N> {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

/// A reference on an external RTree buffer.
///
/// Usually this will be created from an [`RTree`] via its [`as_ref`][RTree::as_ref]
/// method, but it can also be created from any existing data buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct RTreeRef<'a, N: IndexableNum> {
    pub(crate) boxes: &'a [N],
    pub(crate) indices: Indices<'a>,
    pub(crate) metadata: RTreeMetadata<N>,
}

impl<'a, N: IndexableNum> RTreeRef<'a, N> {
    /// Construct a new RTree from an external byte slice.
    ///
    /// This byte slice must conform to the "flatbush ABI", that is, the ABI originally implemented
    /// by the JavaScript [`flatbush` library](https://github.com/mourner/flatbush). You can
    /// extract such a buffer either via [`RTree::into_inner`] or from the `.data` attribute
    /// of the JavaScript `Flatbush` object.
    pub fn try_new<T: AsRef<[u8]>>(data: &'a T) -> Result<Self> {
        let data = data.as_ref();
        let metadata = RTreeMetadata::from_slice(data)?;
        let boxes = metadata.boxes_slice(data);
        let indices = metadata.indices_slice(data);

        Ok(Self {
            boxes,
            indices,
            metadata,
        })
    }

    /// Construct a new RTreeRef without doing any validation
    ///
    /// # Safety
    ///
    /// `metadata` must be valid for this data buffer.
    pub unsafe fn new_unchecked<T: AsRef<[u8]>>(
        data: &'a T,
        metadata: RTreeMetadata<N>,
    ) -> Result<Self> {
        let data = data.as_ref();
        let boxes = metadata.boxes_slice(data);
        let indices = metadata.indices_slice(data);

        Ok(Self {
            boxes,
            indices,
            metadata,
        })
    }
}
