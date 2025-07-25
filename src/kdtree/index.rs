use std::marker::PhantomData;

use bytemuck::cast_slice;

use crate::error::{GeoIndexError, Result};
use crate::indices::Indices;
use crate::kdtree::constants::{KDBUSH_HEADER_SIZE, KDBUSH_MAGIC, KDBUSH_VERSION};
use crate::r#type::IndexableNum;

/// Common metadata to describe a KDTree
///
/// You can use the metadata to infer the total byte size of a tree given the provided criteria.
/// See [`data_buffer_length`][Self::data_buffer_length].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KDTreeMetadata<N: IndexableNum> {
    node_size: u16,
    num_items: u32,
    phantom: PhantomData<N>,
    pub(crate) indices_byte_size: usize,
    pub(crate) pad_coords_byte_size: usize,
    pub(crate) coords_byte_size: usize,
}

impl<N: IndexableNum> KDTreeMetadata<N> {
    /// Construct a new [`KDTreeMetadata`] from a number of items and node size.
    pub fn new(num_items: u32, node_size: u16) -> Self {
        assert!((2..=65535).contains(&node_size));

        let coords_byte_size = (num_items as usize) * 2 * N::BYTES_PER_ELEMENT;
        let indices_bytes_per_element = if num_items < 65536 { 2 } else { 4 };
        let indices_byte_size = (num_items as usize) * indices_bytes_per_element;
        let pad_coords_byte_size = (8 - (indices_byte_size % 8)) % 8;

        Self {
            node_size,
            num_items,
            phantom: PhantomData,
            indices_byte_size,
            pad_coords_byte_size,
            coords_byte_size,
        }
    }

    /// Construct a new [`KDTreeMetadata`] from an existing byte slice conforming to the "kdbush
    /// ABI", such as what [`KDTreeBuilder`] generates.
    pub fn from_slice(data: &[u8]) -> Result<Self> {
        if data[0] != KDBUSH_MAGIC {
            return Err(GeoIndexError::General(
                "Data not in Kdbush format.".to_string(),
            ));
        }

        let version_and_type = data[1];
        let version = version_and_type >> 4;
        if version != KDBUSH_VERSION {
            return Err(GeoIndexError::General(
                format!("Got v{version} data when expected v{KDBUSH_VERSION}.").to_string(),
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

    /// The number of bytes that a KDTree with this metadata would have.
    ///
    /// ```
    /// use geo_index::kdtree::KDTreeMetadata;
    ///
    /// let metadata = KDTreeMetadata::<f64>::new(25000, 16);
    /// assert_eq!(metadata.data_buffer_length(), 450_008);
    /// ```
    pub fn data_buffer_length(&self) -> usize {
        KDBUSH_HEADER_SIZE
            + self.coords_byte_size
            + self.indices_byte_size
            + self.pad_coords_byte_size
    }

    /// Access the slice of coordinates from the data buffer this metadata represents.
    pub fn coords_slice<'a>(&self, data: &'a [u8]) -> &'a [N] {
        let coords_byte_start =
            KDBUSH_HEADER_SIZE + self.indices_byte_size + self.pad_coords_byte_size;
        let coords_byte_end = KDBUSH_HEADER_SIZE
            + self.indices_byte_size
            + self.pad_coords_byte_size
            + self.coords_byte_size;
        cast_slice(&data[coords_byte_start..coords_byte_end])
    }

    /// Access the slice of indices from the data buffer this metadata represents.
    pub fn indices_slice<'a>(&self, data: &'a [u8]) -> Indices<'a> {
        let indices_buf = &data[KDBUSH_HEADER_SIZE..KDBUSH_HEADER_SIZE + self.indices_byte_size];

        if self.num_items < 65536 {
            Indices::U16(cast_slice(indices_buf))
        } else {
            Indices::U32(cast_slice(indices_buf))
        }
    }
}

/// An owned KDTree buffer.
///
/// Usually this will be created from scratch via [`KDTreeBuilder`][crate::kdtree::KDTreeBuilder].
#[derive(Debug, Clone, PartialEq)]
pub struct KDTree<N: IndexableNum> {
    pub(crate) buffer: Vec<u8>,
    pub(crate) metadata: KDTreeMetadata<N>,
}

impl<N: IndexableNum> KDTree<N> {
    /// Consume this KDTree, returning the underlying buffer.
    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }
}

impl<N: IndexableNum> AsRef<[u8]> for KDTree<N> {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

/// A reference on an external KDTree buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct KDTreeRef<'a, N: IndexableNum> {
    pub(crate) coords: &'a [N],
    pub(crate) indices: Indices<'a>,
    pub(crate) metadata: KDTreeMetadata<N>,
}

impl<'a, N: IndexableNum> KDTreeRef<'a, N> {
    /// Construct a new KDTreeRef from an external byte slice.
    ///
    /// This byte slice must conform to the "kdbush ABI", that is, the ABI originally implemented
    /// by the JavaScript [`kdbush` library](https://github.com/mourner/kdbush). You can extract
    /// such a buffer either via [`KDTree::into_inner`] or from the `.data` attribute of the
    /// JavaScript `KDBush` object.
    pub fn try_new<T: AsRef<[u8]>>(data: &'a T) -> Result<Self> {
        let data = data.as_ref();
        let metadata = KDTreeMetadata::from_slice(data)?;
        let coords = metadata.coords_slice(data);
        let indices = metadata.indices_slice(data);

        Ok(Self {
            coords,
            indices,
            metadata,
        })
    }

    /// Construct a new KDTreeRef without doing any validation
    ///
    /// # Safety
    ///
    /// `metadata` must be valid for this data buffer.
    pub unsafe fn new_unchecked<T: AsRef<[u8]>>(
        data: &'a T,
        metadata: KDTreeMetadata<N>,
    ) -> Result<Self> {
        let data = data.as_ref();
        let coords = metadata.coords_slice(data);
        let indices = metadata.indices_slice(data);

        Ok(Self {
            coords,
            indices,
            metadata,
        })
    }
}
