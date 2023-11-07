use bytemuck::cast_slice;

use crate::indices::Indices;
use crate::kdbush::constants::{KDBUSH_HEADER_SIZE, KDBUSH_MAGIC, KDBUSH_VERSION};
use crate::kdbush::error::KdbushError;

#[derive(Debug, Clone, PartialEq)]
pub struct OwnedKdbush {
    pub(crate) buffer: Vec<u8>,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
}

impl OwnedKdbush {
    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }

    pub fn as_flatbush(&self) -> KdbushRef {
        KdbushRef::try_new(self).unwrap()
    }
}

impl AsRef<[u8]> for OwnedKdbush {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KdbushRef<'a> {
    pub(crate) coords: &'a [f64],
    pub(crate) ids: Indices<'a>,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
}

impl<'a> KdbushRef<'a> {
    pub fn try_new<T: AsRef<[u8]>>(data: &'a T) -> Result<Self, KdbushError> {
        let data = data.as_ref();

        if data[0] != KDBUSH_MAGIC {
            return Err(KdbushError::General(
                "Data does not appear to be in a Kdbush format.".to_string(),
            ));
        }

        let version_and_type = data[1];
        let version = version_and_type >> 4;
        if version != KDBUSH_VERSION {
            return Err(KdbushError::General(
                format!("Got v{} data when expected v{}.", version, KDBUSH_VERSION).to_string(),
            ));
        }

        let node_size: u16 = cast_slice(&data[2..4])[0];
        let num_items: u32 = cast_slice(&data[4..8])[0];
        let node_size = node_size as usize;
        let num_items = num_items as usize;

        let f64_bytes_per_element = 8;
        let coords_byte_size = num_items * 2 * f64_bytes_per_element;
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
