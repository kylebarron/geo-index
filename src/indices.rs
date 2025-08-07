//! Data structures to hold insertion and internal tree indices that may be either `u16` or `u32`
//! to save space.

use bytemuck::{cast_slice, cast_slice_mut};

/// A mutable slice of indices that may be either `u16` or `u32`.
#[derive(Debug)]
pub enum MutableIndices<'a> {
    /// Indices stored as a u16 byte slice
    U16(&'a mut [u16]),
    /// Indices stored as a u32 byte slice
    U32(&'a mut [u32]),
}

impl<'a> MutableIndices<'a> {
    pub(crate) fn new(slice: &'a mut [u8], num_nodes: usize) -> Self {
        if num_nodes < 16384 {
            Self::U16(cast_slice_mut(slice))
        } else {
            Self::U32(cast_slice_mut(slice))
        }
    }
}

impl MutableIndices<'_> {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn bytes_per_element(&self) -> usize {
        match self {
            Self::U16(_) => 2,
            Self::U32(_) => 4,
        }
    }

    #[inline]
    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        match self {
            Self::U16(arr) => arr.swap(a, b),
            Self::U32(arr) => arr.swap(a, b),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn get(&self, index: usize) -> usize {
        match self {
            Self::U16(arr) => arr[index] as usize,
            Self::U32(arr) => arr[index] as usize,
        }
    }

    #[inline]
    pub(crate) fn set(&mut self, index: usize, value: usize) {
        match self {
            Self::U16(arr) => arr[index] = value.try_into().unwrap(),
            Self::U32(arr) => arr[index] = value.try_into().unwrap(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn split_at_mut(&mut self, mid: usize) -> (MutableIndices<'_>, MutableIndices<'_>) {
        match self {
            Self::U16(arr) => {
                let (left, right) = arr.split_at_mut(mid);
                (MutableIndices::U16(left), MutableIndices::U16(right))
            }
            Self::U32(arr) => {
                let (left, right) = arr.split_at_mut(mid);
                (MutableIndices::U32(left), MutableIndices::U32(right))
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn chunks_mut(&mut self, chunk_size: usize) -> Vec<MutableIndices<'_>> {
        match self {
            Self::U16(arr) => arr
                .chunks_mut(chunk_size)
                .map(MutableIndices::U16)
                .collect(),
            Self::U32(arr) => arr
                .chunks_mut(chunk_size)
                .map(MutableIndices::U32)
                .collect(),
        }
    }
}

/// A slice of indices that may be either `u16` or `u32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Indices<'a> {
    /// Indices stored as a u16 byte slice
    U16(&'a [u16]),
    /// Indices stored as a u32 byte slice
    U32(&'a [u32]),
}

impl<'a> Indices<'a> {
    pub(crate) fn new(slice: &'a [u8], num_nodes: usize) -> Self {
        if num_nodes < 16384 {
            Self::U16(cast_slice(slice))
        } else {
            Self::U32(cast_slice(slice))
        }
    }
}

impl Indices<'_> {
    /// The number of indices in this byte slice
    pub fn len(&self) -> usize {
        match self {
            Self::U16(arr) => arr.len(),
            Self::U32(arr) => arr.len(),
        }
    }

    /// Whether this slice is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A helper to access a single index from this slice.
    ///
    /// Values are casted from u16 or u32 to usize.
    #[inline]
    pub fn get(&self, index: usize) -> usize {
        match self {
            Self::U16(arr) => arr[index] as usize,
            Self::U32(arr) => arr[index] as usize,
        }
    }
}
