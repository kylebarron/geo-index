use bytemuck::cast_slice_mut;
use geo_traits::{CoordTrait, RectTrait};

use crate::indices::MutableIndices;
use crate::r#type::IndexableNum;
use crate::rtree::constants::VERSION;
use crate::rtree::index::{RTree, RTreeMetadata};
use crate::rtree::sort::{Sort, SortParams};

/// The default node size used by [`RTreeBuilder::new`]
pub const DEFAULT_RTREE_NODE_SIZE: u16 = 16;

/// A builder to create an [`RTree`].
///
/// ```
/// use geo_index::rtree::RTreeBuilder;
/// use geo_index::rtree::sort::HilbertSort;
///
/// let mut builder = RTreeBuilder::<f64>::new(3);
/// builder.add(0., 0., 2., 2.);
/// builder.add(1., 1., 3., 3.);
/// builder.add(2., 2., 4., 4.);
/// let tree = builder.finish::<HilbertSort>();
/// ```
pub struct RTreeBuilder<N: IndexableNum> {
    /// data buffer
    data: Vec<u8>,
    metadata: RTreeMetadata<N>,
    pos: usize,
    min_x: N,
    min_y: N,
    max_x: N,
    max_y: N,
}

impl<N: IndexableNum> RTreeBuilder<N> {
    /// Create a new builder with the provided number of items and the default node size.
    pub fn new(num_items: u32) -> Self {
        Self::new_with_node_size(num_items, DEFAULT_RTREE_NODE_SIZE)
    }

    /// Create a new builder with the provided number of items and node size.
    pub fn new_with_node_size(num_items: u32, node_size: u16) -> Self {
        let metadata = RTreeMetadata::new(num_items, node_size);
        Self::from_metadata(metadata)
    }

    /// Create a new builder with the provided metadata
    pub fn from_metadata(metadata: RTreeMetadata<N>) -> Self {
        let mut data = vec![0; metadata.data_buffer_length()];

        // Set data header
        data[0] = 0xfb;
        data[1] = (VERSION << 4) + N::TYPE_INDEX;
        cast_slice_mut(&mut data[2..4])[0] = metadata.node_size();
        cast_slice_mut(&mut data[4..8])[0] = metadata.num_items();

        Self {
            data,
            metadata,
            pos: 0,
            min_x: N::max_value(),
            min_y: N::max_value(),
            max_x: N::min_value(),
            max_y: N::min_value(),
        }
    }

    /// Access the underlying [RTreeMetadata] of this instance.
    pub fn metadata(&self) -> &RTreeMetadata<N> {
        &self.metadata
    }

    /// Add a given rectangle to the RTree.
    ///
    /// This returns the insertion index, which provides a lookup back into the original data.
    ///
    /// `RTreeIndex::search` will return this same insertion index, which allows you to reference
    /// your original collection.
    #[inline]
    pub fn add(&mut self, min_x: N, min_y: N, max_x: N, max_y: N) -> u32 {
        let index = self.pos >> 2;
        let (boxes, mut indices) = split_data_borrow(&mut self.data, &self.metadata);

        indices.set(index, index);
        boxes[self.pos] = min_x;
        self.pos += 1;
        boxes[self.pos] = min_y;
        self.pos += 1;
        boxes[self.pos] = max_x;
        self.pos += 1;
        boxes[self.pos] = max_y;
        self.pos += 1;

        if min_x < self.min_x {
            self.min_x = min_x
        };
        if min_y < self.min_y {
            self.min_y = min_y
        };
        if max_x > self.max_x {
            self.max_x = max_x
        };
        if max_y > self.max_y {
            self.max_y = max_y
        };

        index.try_into().unwrap()
    }

    /// Add a given rectangle to the RTree.
    ///
    /// This returns the insertion index, which provides a lookup back into the original data.
    ///
    /// `RTreeIndex::search` will return this same insertion index, which allows you to reference
    /// your original collection.
    #[inline]
    pub fn add_slice(
        &mut self,
        mut min_x: impl ExactSizeIterator<Item = N>,
        mut min_y: impl ExactSizeIterator<Item = N>,
        mut max_x: impl ExactSizeIterator<Item = N>,
        mut max_y: impl ExactSizeIterator<Item = N>,
    ) -> Vec<u32> {
        let (boxes, mut indices) = split_data_borrow(&mut self.data, &self.metadata);
        assert_eq!(min_x.len(), min_y.len());
        assert_eq!(min_x.len(), max_x.len());
        assert_eq!(min_x.len(), max_y.len());

        let mut out = Vec::with_capacity(min_x.len());

        for i in 0..min_x.len() {
            let index = self.pos >> 2;

            let this_min_x = min_x.next().unwrap();
            let this_min_y = min_y.next().unwrap();
            let this_max_x = max_x.next().unwrap();
            let this_max_y = max_y.next().unwrap();

            indices.set(index, index);
            boxes[self.pos] = this_min_x;
            self.pos += 1;
            boxes[self.pos] = this_min_y;
            self.pos += 1;
            boxes[self.pos] = this_max_x;
            self.pos += 1;
            boxes[self.pos] = this_max_y;
            self.pos += 1;

            if this_min_x < self.min_x {
                self.min_x = this_min_x
            };
            if this_min_y < self.min_y {
                self.min_y = this_min_y
            };
            if this_max_x > self.max_x {
                self.max_x = this_max_x
            };
            if this_max_y > self.max_y {
                self.max_y = this_max_y
            };

            out.push(index.try_into().unwrap());
        }
        out
    }

    /// Add a given rectangle to the RTree.
    ///
    /// This returns the insertion index, which provides a lookup back into the original data.
    ///
    /// `RTreeIndex::search` will return this same insertion index, which allows you to reference
    /// your original collection.
    #[inline]
    pub fn add_rect(&mut self, rect: &impl RectTrait<T = N>) -> u32 {
        self.add(
            rect.min().x(),
            rect.min().y(),
            rect.max().x(),
            rect.max().y(),
        )
    }

    /// Consume this builder, perfoming the sort and generating an RTree ready for queries.
    ///
    /// [`HilbertSort`] and [`STRSort`] both implement [`Sort`], allowing you to choose the method
    /// used.
    ///
    /// [`HilbertSort`]: crate::rtree::sort::HilbertSort
    /// [`STRSort`]: crate::rtree::sort::STRSort
    pub fn finish<S: Sort<N>>(mut self) -> RTree<N> {
        assert_eq!(
            self.pos >> 2,
            self.metadata.num_items() as usize,
            "Added {} items when expected {}.",
            self.pos >> 2,
            self.metadata.num_items()
        );

        let (boxes, mut indices) = split_data_borrow(&mut self.data, &self.metadata);

        if self.metadata.num_items() == 1 {
            // Only one item, we don't even have a root node to fill
            return RTree {
                buffer: self.data,
                metadata: self.metadata,
            };
        }

        if self.metadata.num_items() as usize <= self.metadata.node_size() as usize {
            // only one node, skip sorting and just fill the root box
            boxes[self.pos] = self.min_x;
            self.pos += 1;
            boxes[self.pos] = self.min_y;
            self.pos += 1;
            boxes[self.pos] = self.max_x;
            self.pos += 1;
            boxes[self.pos] = self.max_y;
            self.pos += 1;

            return RTree {
                buffer: self.data,
                metadata: self.metadata,
            };
        }

        let mut sort_params = SortParams {
            num_items: self.metadata.num_items() as usize,
            node_size: self.metadata.node_size() as usize,
            min_x: self.min_x,
            min_y: self.min_y,
            max_x: self.max_x,
            max_y: self.max_y,
        };
        S::sort(&mut sort_params, boxes, &mut indices);

        {
            // generate nodes at each tree level, bottom-up
            let mut pos = 0;
            for end in self.metadata.level_bounds()[..self.metadata.level_bounds().len() - 1].iter()
            {
                while pos < *end {
                    let node_index = pos;

                    // calculate bbox for the new node
                    let mut node_min_x = boxes[pos];
                    pos += 1;
                    let mut node_min_y = boxes[pos];
                    pos += 1;
                    let mut node_max_x = boxes[pos];
                    pos += 1;
                    let mut node_max_y = boxes[pos];
                    pos += 1;
                    for _ in 1..self.metadata.node_size() {
                        if pos >= *end {
                            break;
                        }

                        if boxes[pos] < node_min_x {
                            node_min_x = boxes[pos];
                        }
                        pos += 1;
                        if boxes[pos] < node_min_y {
                            node_min_y = boxes[pos];
                        }
                        pos += 1;
                        if boxes[pos] > node_max_x {
                            node_max_x = boxes[pos]
                        }
                        pos += 1;
                        if boxes[pos] > node_max_y {
                            node_max_y = boxes[pos]
                        }
                        pos += 1;
                    }

                    // add the new node to the tree data
                    indices.set(self.pos >> 2, node_index);
                    boxes[self.pos] = node_min_x;
                    self.pos += 1;
                    boxes[self.pos] = node_min_y;
                    self.pos += 1;
                    boxes[self.pos] = node_max_x;
                    self.pos += 1;
                    boxes[self.pos] = node_max_y;
                    self.pos += 1;
                }
            }
        }

        RTree {
            buffer: self.data,
            metadata: self.metadata,
        }
    }
}

/// Mutable borrow of boxes and indices
#[inline]
fn split_data_borrow<'a, N: IndexableNum>(
    data: &'a mut [u8],
    metadata: &'a RTreeMetadata<N>,
) -> (&'a mut [N], MutableIndices<'a>) {
    let (boxes_buf, indices_buf) = data[8..].split_at_mut(metadata.nodes_byte_length);
    debug_assert_eq!(indices_buf.len(), metadata.indices_byte_length);

    let boxes = cast_slice_mut(boxes_buf);
    let indices = MutableIndices::new(indices_buf, metadata.num_nodes());
    (boxes, indices)
}

#[cfg(test)]
mod test {
    use crate::rtree::sort::HilbertSort;
    use crate::rtree::RTreeIndex;

    use super::*;

    #[test]
    fn does_not_panic_length_1_tree() {
        let mut builder = RTreeBuilder::<f64>::new(1);
        builder.add(-20., -20., 1020., 1020.);
        let tree = builder.finish::<HilbertSort>();
        let result = tree.search(0., 0., 0., 0.);
        assert_eq!(result, vec![0]);
    }
}
