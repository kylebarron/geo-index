use bytemuck::{cast_slice_mut, cast_slice};

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
}

#[derive(Debug, Clone, PartialEq)]
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
    #[inline]
    pub fn get(&self, index: usize) -> usize {
        match self {
            Self::U16(arr) => arr[index] as usize,
            Self::U32(arr) => arr[index] as usize,
        }
    }
}
