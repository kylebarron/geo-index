use std::cmp::Reverse;
use std::collections::BinaryHeap;

use geo_traits::{CoordTrait, RectTrait};

use crate::error::Result;
use crate::indices::Indices;
use crate::r#type::IndexableNum;
use crate::rtree::index::{RTree, RTreeRef};
use crate::rtree::traversal::{IntersectionIterator, Node};
use crate::rtree::util::upper_bound;
use crate::rtree::RTreeMetadata;
use crate::GeoIndexError;

/// A trait for searching and accessing data out of an RTree.
pub trait RTreeIndex<N: IndexableNum>: Sized {
    /// A slice representing all the bounding boxes of all elements contained within this tree,
    /// including the bounding boxes of each internal node.
    fn boxes(&self) -> &[N];

    /// A slice representing the indices within the `boxes` slice, including internal nodes.
    fn indices(&self) -> Indices<'_>;

    /// Access the metadata describing this RTree
    fn metadata(&self) -> &RTreeMetadata<N>;

    /// The total number of items contained in this RTree.
    fn num_items(&self) -> u32 {
        self.metadata().num_items()
    }

    /// The total number of nodes in this RTree, including both leaf and intermediate nodes.
    fn num_nodes(&self) -> usize {
        self.metadata().num_nodes()
    }

    /// The maximum number of elements in each node.
    fn node_size(&self) -> u16 {
        self.metadata().node_size()
    }

    /// The offsets into [RTreeIndex::boxes] where each level's boxes starts and ends. The tree is
    /// laid out bottom-up, and there's an implicit initial 0. So the boxes of the lowest level of
    /// the tree are located from `boxes[0..self.level_bounds()[0]]`.
    fn level_bounds(&self) -> &[usize] {
        self.metadata().level_bounds()
    }

    /// The number of levels (height) of the tree.
    fn num_levels(&self) -> usize {
        self.level_bounds().len()
    }

    /// The tree is laid out from bottom to top. Level 0 is the _base_ of the tree. Each integer
    /// higher is one level higher of the tree.
    fn boxes_at_level(&self, level: usize) -> Result<&[N]> {
        let level_bounds = self.level_bounds();
        if level >= level_bounds.len() {
            return Err(GeoIndexError::General("Level out of bounds".to_string()));
        }
        let result = if level == 0 {
            &self.boxes()[0..level_bounds[0]]
        } else if level == level_bounds.len() {
            &self.boxes()[level_bounds[level]..]
        } else {
            &self.boxes()[level_bounds[level - 1]..level_bounds[level]]
        };
        Ok(result)
    }

    /// Search an RTree given the provided bounding box.
    ///
    /// Results are the indexes of the inserted objects in insertion order.
    fn search(&self, min_x: N, min_y: N, max_x: N, max_y: N) -> Vec<u32> {
        let boxes = self.boxes();
        let indices = self.indices();
        if boxes.is_empty() {
            return vec![];
        }

        let mut outer_node_index = boxes.len().checked_sub(4);

        let mut queue = vec![];
        let mut results = vec![];

        while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() as usize * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // search through child nodes
            for pos in (node_index..end).step_by(4) {
                // check if node bbox intersects with query bbox
                if max_x < boxes[pos] {
                    continue; // maxX < nodeMinX
                }
                if max_y < boxes[pos + 1] {
                    continue; // maxY < nodeMinY
                }
                if min_x > boxes[pos + 2] {
                    continue; // minX > nodeMaxX
                }
                if min_y > boxes[pos + 3] {
                    continue; // minY > nodeMaxY
                }

                let index = indices.get(pos >> 2);

                if node_index >= self.num_items() as usize * 4 {
                    queue.push(index); // node; add it to the search queue
                } else {
                    // Since the max items of the index is u32, we can coerce to u32
                    results.push(index.try_into().unwrap()); // leaf item
                }
            }

            outer_node_index = queue.pop();
        }

        results
    }

    /// Search an RTree given the provided bounding box.
    ///
    /// Results are the indexes of the inserted objects in insertion order.
    fn search_rect(&self, rect: &impl RectTrait<T = N>) -> Vec<u32> {
        self.search(
            rect.min().x(),
            rect.min().y(),
            rect.max().x(),
            rect.max().y(),
        )
    }

    /// Search items in order of distance from the given point.
    ///
    /// ```
    /// use geo_index::rtree::{RTreeBuilder, RTreeIndex, RTreeRef};
    /// use geo_index::rtree::sort::HilbertSort;
    ///
    /// // Create an RTree
    /// let mut builder = RTreeBuilder::<f64>::new(3);
    /// builder.add(0., 0., 2., 2.);
    /// builder.add(1., 1., 3., 3.);
    /// builder.add(2., 2., 4., 4.);
    /// let tree = builder.finish::<HilbertSort>();
    ///
    /// let results = tree.neighbors(5., 5., None, None);
    /// assert_eq!(results, vec![2, 1, 0]);
    /// ```
    fn neighbors(
        &self,
        x: N,
        y: N,
        max_results: Option<usize>,
        max_distance: Option<N>,
    ) -> Vec<u32> {
        let boxes = self.boxes();
        let indices = self.indices();
        let max_distance = max_distance.unwrap_or(N::max_value());

        let mut outer_node_index = Some(boxes.len() - 4);
        let mut queue = BinaryHeap::new();
        let mut results: Vec<u32> = vec![];
        let max_dist_squared = max_distance * max_distance;

        'outer: while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() as usize * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // add child nodes to the queue
            for pos in (node_index..end).step_by(4) {
                let index = indices.get(pos >> 2);

                let dx = axis_dist(x, boxes[pos], boxes[pos + 2]);
                let dy = axis_dist(y, boxes[pos + 1], boxes[pos + 3]);
                let dist = dx * dx + dy * dy;
                if dist > max_dist_squared {
                    continue;
                }

                if node_index >= self.num_items() as usize * 4 {
                    // node (use even id)
                    queue.push(Reverse(NeighborNode {
                        id: index << 1,
                        dist,
                    }));
                } else {
                    // leaf item (use odd id)
                    queue.push(Reverse(NeighborNode {
                        id: (index << 1) + 1,
                        dist,
                    }));
                }
            }

            // pop items from the queue
            while !queue.is_empty() && queue.peek().is_some_and(|val| (val.0.id & 1) != 0) {
                let dist = queue.peek().unwrap().0.dist;
                if dist > max_dist_squared {
                    break 'outer;
                }
                let item = queue.pop().unwrap();
                results.push((item.0.id >> 1).try_into().unwrap());
                if max_results.is_some_and(|max_results| results.len() == max_results) {
                    break 'outer;
                }
            }

            if let Some(item) = queue.pop() {
                outer_node_index = Some(item.0.id >> 1);
            } else {
                outer_node_index = None;
            }
        }

        results
    }

    /// Search items in order of distance from the given coordinate.
    fn neighbors_coord(
        &self,
        coord: &impl CoordTrait<T = N>,
        max_results: Option<usize>,
        max_distance: Option<N>,
    ) -> Vec<u32> {
        self.neighbors(coord.x(), coord.y(), max_results, max_distance)
    }

    /// Returns an iterator over the indexes of objects in this and another tree that intersect.
    ///
    /// Each returned object is of the form `(u32, u32)`, where the first is the positional
    /// index of the "left" tree and the second is the index of the "right" tree.
    fn intersection_candidates_with_other_tree<'a>(
        &'a self,
        other: &'a impl RTreeIndex<N>,
    ) -> impl Iterator<Item = (u32, u32)> + 'a {
        IntersectionIterator::from_trees(self, other)
    }

    /// Access the root node of the RTree for manual traversal.
    fn root(&self) -> Node<'_, N, Self> {
        Node::from_root(self)
    }
}

/// A wrapper around a node and its distance for use in the priority queue.
#[derive(Debug, Clone, Copy, PartialEq)]
struct NeighborNode<N: IndexableNum> {
    id: usize,
    dist: N,
}

impl<N: IndexableNum> Eq for NeighborNode<N> {}

impl<N: IndexableNum> Ord for NeighborNode<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We don't allow NaN. This should only panic on NaN
        self.dist.partial_cmp(&other.dist).unwrap()
    }
}

impl<N: IndexableNum> PartialOrd for NeighborNode<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: IndexableNum> RTreeIndex<N> for RTree<N> {
    fn boxes(&self) -> &[N] {
        self.metadata.boxes_slice(&self.buffer)
    }

    fn indices(&self) -> Indices<'_> {
        self.metadata.indices_slice(&self.buffer)
    }

    fn metadata(&self) -> &RTreeMetadata<N> {
        &self.metadata
    }
}

impl<N: IndexableNum> RTreeIndex<N> for RTreeRef<'_, N> {
    fn boxes(&self) -> &[N] {
        self.boxes
    }

    fn indices(&self) -> Indices<'_> {
        self.indices
    }

    fn metadata(&self) -> &RTreeMetadata<N> {
        &self.metadata
    }
}

/// 1D distance from a value to a range.
#[allow(dead_code)]
#[inline]
fn axis_dist<N: IndexableNum>(k: N, min: N, max: N) -> N {
    if k < min {
        min - k
    } else if k <= max {
        N::zero()
    } else {
        k - max
    }
}

#[cfg(test)]
mod test {
    // Replication of tests from flatbush js
    mod js {
        use crate::rtree::RTreeIndex;
        use crate::test::{flatbush_js_test_data, flatbush_js_test_index};

        #[test]
        fn performs_bbox_search() {
            let data = flatbush_js_test_data();
            let index = flatbush_js_test_index();
            let ids = index.search(40., 40., 60., 60.);

            let mut results: Vec<usize> = vec![];
            for id in ids {
                results.push(data[4 * id as usize] as usize);
                results.push(data[4 * id as usize + 1] as usize);
                results.push(data[4 * id as usize + 2] as usize);
                results.push(data[4 * id as usize + 3] as usize);
            }

            results.sort();

            let mut expected = vec![
                57, 59, 58, 59, 48, 53, 52, 56, 40, 42, 43, 43, 43, 41, 47, 43,
            ];
            expected.sort();

            assert_eq!(results, expected);
        }
    }
}
