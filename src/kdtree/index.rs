use std::marker::PhantomData;

use bytemuck::cast_slice;

use crate::error::GeoIndexError;
use crate::indices::Indices;
use crate::kdtree::constants::{KDBUSH_HEADER_SIZE, KDBUSH_MAGIC, KDBUSH_VERSION};
use crate::r#type::IndexableNum;

/// An owned KDTree buffer.
///
/// Usually this will be created from scratch via [`KDTreeBuilder`][crate::kdtree::KDTreeBuilder].
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedKDTree<N: IndexableNum> {
    pub(crate) buffer: Vec<u8>,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
    pub(crate) phantom: PhantomData<N>,
}

impl<N: IndexableNum> OwnedKDTree<N> {
    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }

    pub fn as_ref(&self) -> KDTreeRef<N> {
        KDTreeRef::try_new(self).unwrap()
    }
}

impl<N: IndexableNum> AsRef<[u8]> for OwnedKDTree<N> {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

/// A reference on an external KDTree buffer.
///
/// Usually this will be created from an [`OwnedKDTree`] via its [`as_ref`][OwnedKDTree::as_ref]
/// method, but it can also be created from any existing data buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KDTreeRef<'a, N: IndexableNum> {
    pub(crate) coords: &'a [N],
    pub(crate) ids: Indices<'a>,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
}

impl<'a, N: IndexableNum> KDTreeRef<'a, N> {
    pub fn try_new<T: AsRef<[u8]>>(data: &'a T) -> Result<Self, GeoIndexError> {
        let data = data.as_ref();
        // TODO: validate length of slice?

        if data[0] != KDBUSH_MAGIC {
            return Err(GeoIndexError::General(
                "Data not in Kdbush format.".to_string(),
            ));
        }

        let version_and_type = data[1];
        let version = version_and_type >> 4;
        if version != KDBUSH_VERSION {
            return Err(GeoIndexError::General(
                format!("Got v{} data when expected v{}.", version, KDBUSH_VERSION).to_string(),
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

        let coords_byte_size = num_items * 2 * N::BYTES_PER_ELEMENT;
        let indices_bytes_per_element = if num_items < 65536 { 2 } else { 4 };
        let ids_byte_size = num_items * indices_bytes_per_element;
        let pad_coords_byte_size = (8 - (ids_byte_size % 8)) % 8;

        let data_buffer_length =
            KDBUSH_HEADER_SIZE + coords_byte_size + ids_byte_size + pad_coords_byte_size;
        assert_eq!(data.len(), data_buffer_length);

        let indices_buf = &data[KDBUSH_HEADER_SIZE..KDBUSH_HEADER_SIZE + ids_byte_size];
        let ids = if num_items < 65536 {
            Indices::U16(cast_slice(indices_buf))
        } else {
            Indices::U32(cast_slice(indices_buf))
        };
        let coords_byte_start = KDBUSH_HEADER_SIZE + ids_byte_size + pad_coords_byte_size;
        let coords_byte_end =
            KDBUSH_HEADER_SIZE + ids_byte_size + pad_coords_byte_size + coords_byte_size;
        let coords = cast_slice(&data[coords_byte_start..coords_byte_end]);

        Ok(Self {
            coords,
            ids,
            node_size,
            num_items,
        })
    }
}
