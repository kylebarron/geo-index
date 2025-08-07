use std::cmp;

use bytemuck::cast_slice_mut;
use geo_traits::{CoordTrait, PointTrait};

use crate::error::Result;
use crate::indices::MutableIndices;
use crate::kdtree::constants::{KDBUSH_HEADER_SIZE, KDBUSH_MAGIC, KDBUSH_VERSION};
use crate::kdtree::index::KDTreeMetadata;
use crate::kdtree::KDTree;
use crate::r#type::IndexableNum;
use crate::GeoIndexError;

/// Default node size in [`KDTreeBuilder::new`].
pub const DEFAULT_KDTREE_NODE_SIZE: u16 = 64;

/// A builder to create an [`KDTree`].
#[derive(Debug)]
pub struct KDTreeBuilder<N: IndexableNum> {
    /// data buffer
    data: Vec<u8>,
    metadata: KDTreeMetadata<N>,
    pos: usize,
}

impl<N: IndexableNum> KDTreeBuilder<N> {
    /// Create a new builder with the provided number of items and the default node size.
    pub fn new(num_items: u32) -> Self {
        Self::new_with_node_size(num_items, DEFAULT_KDTREE_NODE_SIZE)
    }

    /// Create a new builder with the provided number of items and node size.
    pub fn new_with_node_size(num_items: u32, node_size: u16) -> Self {
        let metadata = KDTreeMetadata::new(num_items, node_size);
        Self::from_metadata(metadata)
    }

    /// Create a new builder with the provided metadata
    pub fn from_metadata(metadata: KDTreeMetadata<N>) -> Self {
        let data_buffer_length = metadata.data_buffer_length();
        let mut data = vec![0; data_buffer_length];

        // Set data header;
        data[0] = KDBUSH_MAGIC;
        data[1] = (KDBUSH_VERSION << 4) + N::TYPE_INDEX;
        cast_slice_mut(&mut data[2..4])[0] = metadata.node_size();
        cast_slice_mut(&mut data[4..8])[0] = metadata.num_items();

        Self {
            data,
            pos: 0,
            metadata,
        }
    }

    /// Access the underlying [KDTreeMetadata] of this instance.
    pub fn metadata(&self) -> &KDTreeMetadata<N> {
        &self.metadata
    }

    /// Add a point to the KDTree.
    ///
    /// This returns a positional index that provides a lookup back into the original data.
    #[inline]
    pub fn add(&mut self, x: N, y: N) -> u32 {
        let index = self.pos >> 1;
        let (coords, mut ids) = split_data_borrow(&mut self.data, self.metadata);

        ids.set(index, index);
        coords[self.pos] = x;
        self.pos += 1;
        coords[self.pos] = y;
        self.pos += 1;

        index.try_into().unwrap()
    }

    /// Add a coord to the KDTree.
    ///
    /// This returns a positional index that provides a lookup back into the original data.
    #[inline]
    pub fn add_coord(&mut self, coord: &impl CoordTrait<T = N>) -> u32 {
        self.add(coord.x(), coord.y())
    }

    /// Add a point to the KDTree.
    ///
    /// This returns a positional index that provides a lookup back into the original data.
    ///
    /// ## Errors
    ///
    /// - If the point is empty.
    #[inline]
    pub fn add_point(&mut self, point: &impl PointTrait<T = N>) -> Result<u32> {
        let coord = point.coord().ok_or(GeoIndexError::General(
            "Unable to add empty point to KDTree".to_string(),
        ))?;
        Ok(self.add_coord(&coord))
    }

    /// Consume this builder, perfoming the k-d sort and generating a KDTree ready for queries.
    pub fn finish(mut self) -> KDTree<N> {
        assert_eq!(
            self.pos >> 1,
            self.metadata.num_items() as usize,
            "Added {} items when expected {}.",
            self.pos >> 1,
            self.metadata.num_items()
        );

        let (coords, mut ids) = split_data_borrow::<N>(&mut self.data, self.metadata);

        // kd-sort both arrays for efficient search
        sort(
            &mut ids,
            coords,
            self.metadata.node_size() as usize,
            0,
            self.metadata.num_items() as usize - 1,
            0,
        );

        KDTree {
            buffer: self.data,
            metadata: self.metadata,
        }
    }
}

/// Mutable borrow of coords and ids
fn split_data_borrow<N: IndexableNum>(
    data: &mut [u8],
    metadata: KDTreeMetadata<N>,
) -> (&mut [N], MutableIndices<'_>) {
    let (ids_buf, padded_coords_buf) =
        data[KDBUSH_HEADER_SIZE..].split_at_mut(metadata.indices_byte_size);
    let coords_buf = &mut padded_coords_buf[metadata.pad_coords_byte_size..];
    debug_assert_eq!(coords_buf.len(), metadata.coords_byte_size);

    let ids = if metadata.num_items() < 65536 {
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
