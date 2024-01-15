//! Data structures to hold indices that may be either `u16` or `u32` to save space.

use bytemuck::{cast_slice, cast_slice_mut};

/// A mutable slice of indices that may be either `u16` or `u32`.
#[derive(Debug)]
pub enum MutableIndices<'a> {
    U16(&'a mut [u16]),
    U32(&'a mut [u32]),
}

impl<'a> MutableIndices<'a> {
    pub fn new(slice: &'a mut [u8], num_nodes: usize) -> Self {
        if num_nodes < 16384 {
            Self::U16(cast_slice_mut(slice))
        } else {
            Self::U32(cast_slice_mut(slice))
        }
    }
}

impl MutableIndices<'_> {
    #[inline]
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::U16(_) => 2,
            Self::U32(_) => 4,
        }
    }

    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        match self {
            Self::U16(arr) => arr.swap(a, b),
            Self::U32(arr) => arr.swap(a, b),
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> usize {
        match self {
            Self::U16(arr) => arr[index] as usize,
            Self::U32(arr) => arr[index] as usize,
        }
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: usize) {
        match self {
            Self::U16(arr) => arr[index] = value.try_into().unwrap(),
            Self::U32(arr) => arr[index] = value.try_into().unwrap(),
        }
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (MutableIndices, MutableIndices) {
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

    pub fn chunks_mut(&mut self, chunk_size: usize) -> Vec<MutableIndices> {
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
    U16(&'a [u16]),
    U32(&'a [u32]),
}

impl<'a> Indices<'a> {
    pub fn new(slice: &'a [u8], num_nodes: usize) -> Self {
        if num_nodes < 16384 {
            Self::U16(cast_slice(slice))
        } else {
            Self::U32(cast_slice(slice))
        }
    }
}

impl Indices<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::U16(arr) => arr.len(),
            Self::U32(arr) => arr.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> usize {
        match self {
            Self::U16(arr) => arr[index] as usize,
            Self::U32(arr) => arr[index] as usize,
        }
    }
}
