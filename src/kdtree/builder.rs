use std::cmp;
use std::marker::PhantomData;

use bytemuck::cast_slice_mut;

use crate::indices::MutableIndices;
use crate::kdtree::constants::{KDBUSH_HEADER_SIZE, KDBUSH_MAGIC, KDBUSH_VERSION};
use crate::kdtree::OwnedKDTree;
use crate::r#type::IndexableNum;

const DEFAULT_NODE_SIZE: usize = 64;

/// A builder to create an [`OwnedKDTree`].
pub struct KDTreeBuilder<N: IndexableNum> {
    /// data buffer
    data: Vec<u8>,

    num_items: usize,
    node_size: usize,

    coords_byte_size: usize,
    indices_byte_size: usize,
    pad_coords_byte_size: usize,

    pos: usize,

    phantom: PhantomData<N>,
}

impl<N: IndexableNum> KDTreeBuilder<N> {
    /// Create a new builder with the provided number of items and the default node size.
    pub fn new(num_items: usize) -> Self {
        Self::new_with_node_size(num_items, DEFAULT_NODE_SIZE)
    }

    /// Create a new builder with the provided number of items and node size.
    pub fn new_with_node_size(num_items: usize, node_size: usize) -> Self {
        assert!((2..=65535).contains(&node_size));
        assert!(num_items <= u32::MAX.try_into().unwrap());

        let coords_byte_size = num_items * 2 * N::BYTES_PER_ELEMENT;
        let indices_bytes_per_element = if num_items < 65536 { 2 } else { 4 };
        let indices_byte_size = num_items * indices_bytes_per_element;
        let pad_coords_byte_size = (8 - (indices_byte_size % 8)) % 8;

        let data_buffer_length =
            KDBUSH_HEADER_SIZE + coords_byte_size + indices_byte_size + pad_coords_byte_size;
        let mut data = vec![0; data_buffer_length];

        // Set data header;
        data[0] = KDBUSH_MAGIC;
        data[1] = (KDBUSH_VERSION << 4) + N::TYPE_INDEX;
        cast_slice_mut(&mut data[2..4])[0] = node_size as u16;
        cast_slice_mut(&mut data[4..8])[0] = num_items as u32;

        Self {
            data,
            num_items,
            node_size,
            coords_byte_size,
            indices_byte_size,
            pad_coords_byte_size,
            pos: 0,
            phantom: PhantomData,
        }
    }

    /// Add a point to the index.
    pub fn add(&mut self, x: N, y: N) -> usize {
        let index = self.pos >> 1;
        let (coords, mut ids) = split_data_borrow(
            &mut self.data,
            self.num_items,
            self.indices_byte_size,
            self.coords_byte_size,
            self.pad_coords_byte_size,
        );

        ids.set(index, index);
        coords[self.pos] = x;
        self.pos += 1;
        coords[self.pos] = y;
        self.pos += 1;

        index
    }

    /// Consume this builder, perfoming the k-d sort and generating a KDTree ready for queries.
    pub fn finish(mut self) -> OwnedKDTree<N> {
        assert_eq!(
            self.pos >> 1,
            self.num_items,
            "Added {} items when expected {}.",
            self.pos >> 1,
            self.num_items
        );

        let (coords, mut ids) = split_data_borrow::<N>(
            &mut self.data,
            self.num_items,
            self.indices_byte_size,
            self.coords_byte_size,
            self.pad_coords_byte_size,
        );

        // kd-sort both arrays for efficient search
        sort(&mut ids, coords, self.node_size, 0, self.num_items - 1, 0);

        OwnedKDTree {
            buffer: self.data,
            node_size: self.node_size,
            num_items: self.num_items,
            phantom: PhantomData,
        }
    }
}

/// Mutable borrow of coords and ids
fn split_data_borrow<N: IndexableNum>(
    data: &mut [u8],
    num_items: usize,
    indices_byte_size: usize,
    coords_byte_size: usize,
    pad_coords: usize,
) -> (&mut [N], MutableIndices) {
    let (ids_buf, padded_coords_buf) = data[KDBUSH_HEADER_SIZE..].split_at_mut(indices_byte_size);
    let coords_buf = &mut padded_coords_buf[pad_coords..];
    debug_assert_eq!(coords_buf.len(), coords_byte_size);

    let ids = if num_items < 65536 {
        MutableIndices::U16(cast_slice_mut(ids_buf))
    } else {
        MutableIndices::U32(cast_slice_mut(ids_buf))
    };
    let coords = cast_slice_mut(coords_buf);

    (coords, ids)
}

fn sort<N: IndexableNum>(
    ids: &mut MutableIndices,
    coords: &mut [N],
    node_size: usize,
    left: usize,
    right: usize,
    axis: usize,
) {
    if right - left <= node_size {
        return;
    }

    // middle index
    let m = (left + right) >> 1;

    // sort ids and coords around the middle index so that the halves lie either left/right or
    // top/bottom correspondingly (taking turns)
    select(ids, coords, m, left, right, axis);

    // recursively kd-sort first half and second half on the opposite axis
    sort(ids, coords, node_size, left, m - 1, 1 - axis);
    sort(ids, coords, node_size, m + 1, right, 1 - axis);
}

/// Custom Floyd-Rivest selection algorithm: sort ids and coords so that [left..k-1] items are
/// smaller than k-th item (on either x or y axis)
#[inline]
fn select<N: IndexableNum>(
    ids: &mut MutableIndices,
    coords: &mut [N],
    k: usize,
    mut left: usize,
    mut right: usize,
    axis: usize,
) {
    while right > left {
        if right - left > 600 {
            let n = (right - left + 1) as f64;
            let m = (k - left + 1) as f64;
            let z = f64::ln(n);
            let s = 0.5 * f64::exp((2.0 * z) / 3.0);
            let sd = 0.5
                * f64::sqrt((z * s * (n - s)) / n)
                * (if m - n / 2.0 < 0.0 { -1.0 } else { 1.0 });
            let new_left = cmp::max(left, f64::floor(k as f64 - (m * s) / n + sd) as usize);
            let new_right = cmp::min(
                right,
                f64::floor(k as f64 + ((n - m) * s) / n + sd) as usize,
            );
            select(ids, coords, k, new_left, new_right, axis);
        }

        let t = coords[2 * k + axis];
        let mut i = left;
        let mut j = right;

        swap_item(ids, coords, left, k);
        if coords[2 * right + axis] > t {
            swap_item(ids, coords, left, right);
        }

        while i < j {
            swap_item(ids, coords, i, j);
            i += 1;
            j -= 1;
            while coords[2 * i + axis] < t {
                i += 1;
            }
            while coords[2 * j + axis] > t {
                j -= 1;
            }
        }

        if coords[2 * left + axis] == t {
            swap_item(ids, coords, left, j);
        } else {
            j += 1;
            swap_item(ids, coords, j, right);
        }

        if j <= k {
            left = j + 1;
        }
        if k <= j {
            right = j - 1;
        }
    }
}

#[inline]
fn swap_item<N: IndexableNum>(ids: &mut MutableIndices, coords: &mut [N], i: usize, j: usize) {
    ids.swap(i, j);
    coords.swap(2 * i, 2 * j);
    coords.swap(2 * i + 1, 2 * j + 1);
}
