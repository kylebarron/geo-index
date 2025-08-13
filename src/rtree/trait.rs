use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::vec;

use geo::algorithm::BoundingRect;
use geo_traits::{CoordTrait, RectTrait};

use crate::error::Result;
use crate::indices::Indices;
use crate::r#type::IndexableNum;
use crate::rtree::distance::{DistanceMetric, EuclideanDistance};
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

        let mut queue = VecDeque::with_capacity(self.node_size() as usize);
        let mut results = vec![];

        while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() as usize * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // search through child nodes
            for pos in (node_index..end).step_by(4) {
                // Safety: pos was checked before to be within bounds
                // Justification: avoiding bounds check improves performance by up to 30%
                let (node_min_x, node_min_y, node_max_x, node_max_y) = unsafe {
                    let node_min_x = *boxes.get_unchecked(pos);
                    let node_min_y = *boxes.get_unchecked(pos + 1);
                    let node_max_x = *boxes.get_unchecked(pos + 2);
                    let node_max_y = *boxes.get_unchecked(pos + 3);
                    (node_min_x, node_min_y, node_max_x, node_max_y)
                };

                // check if the query box disjoint with the node box
                if max_x < node_min_x
                    || max_y < node_min_y
                    || min_x > node_max_x
                    || min_y > node_max_y
                {
                    continue;
                }

                let index = indices.get(pos >> 2);

                if node_index >= self.num_items() as usize * 4 {
                    queue.push_back(index); // node; add it to the search queue
                } else {
                    // Since the max items of the index is u32, we can coerce to u32
                    results.push(index.try_into().unwrap()); // leaf item
                }
            }

            outer_node_index = queue.pop_front();
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
    /// This method uses Euclidean distance by default. For other distance metrics,
    /// use [`neighbors_with_distance`].
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
        // Use Euclidean distance by default for backward compatibility
        let euclidean_distance = EuclideanDistance;
        self.neighbors_with_distance(x, y, max_results, max_distance, &euclidean_distance)
    }

    /// Search items in order of distance from the given point using a custom distance metric.
    ///
    /// This method allows you to specify a custom distance calculation method, such as
    /// Euclidean, Haversine, or Spheroid distance.
    ///
    /// ```
    /// use geo_index::rtree::{RTreeBuilder, RTreeIndex};
    /// use geo_index::rtree::distance::{EuclideanDistance, HaversineDistance};
    /// use geo_index::rtree::sort::HilbertSort;
    ///
    /// // Create an RTree with geographic coordinates (longitude, latitude)
    /// let mut builder = RTreeBuilder::<f64>::new(3);
    /// builder.add(-74.0, 40.7, -74.0, 40.7); // New York
    /// builder.add(-0.1, 51.5, -0.1, 51.5);   // London
    /// builder.add(139.7, 35.7, 139.7, 35.7); // Tokyo
    /// let tree = builder.finish::<HilbertSort>();
    ///
    /// // Find nearest neighbors using Haversine distance (great-circle distance)
    /// let haversine = HaversineDistance::default();
    /// let results = tree.neighbors_with_distance(-74.0, 40.7, Some(2), None, &haversine);
    /// ```
    fn neighbors_with_distance(
        &self,
        x: N,
        y: N,
        max_results: Option<usize>,
        max_distance: Option<N>,
        distance_metric: &dyn DistanceMetric<N>,
    ) -> Vec<u32> {
        let boxes = self.boxes();
        let indices = self.indices();
        let max_distance = max_distance.unwrap_or(distance_metric.max_distance());

        let mut outer_node_index = Some(boxes.len() - 4);
        let mut queue = BinaryHeap::new();
        let mut results: Vec<u32> = vec![];

        'outer: while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() as usize * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // add child nodes to the queue
            for pos in (node_index..end).step_by(4) {
                let index = indices.get(pos >> 2);

                // Use the custom distance metric for bbox distance calculation
                let dist = distance_metric.distance_to_bbox(
                    x,
                    y,
                    boxes[pos],
                    boxes[pos + 1],
                    boxes[pos + 2],
                    boxes[pos + 3],
                );

                if dist > max_distance {
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
                    // Use consistent distance calculation for both nodes and leaf items
                    queue.push(Reverse(NeighborNode {
                        id: (index << 1) + 1,
                        dist,
                    }));
                }
            }

            // pop items from the queue
            while !queue.is_empty() && queue.peek().is_some_and(|val| (val.0.id & 1) != 0) {
                let dist = queue.peek().unwrap().0.dist;
                if dist > max_distance {
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

    /// Search items in order of distance from the given coordinate using a custom distance metric.
    fn neighbors_coord_with_distance(
        &self,
        coord: &impl CoordTrait<T = N>,
        max_results: Option<usize>,
        max_distance: Option<N>,
        distance_metric: &dyn DistanceMetric<N>,
    ) -> Vec<u32> {
        self.neighbors_with_distance(
            coord.x(),
            coord.y(),
            max_results,
            max_distance,
            distance_metric,
        )
    }

    /// Search items in order of distance from a query geometry using a custom distance metric.
    ///
    /// This method finds nearest neighbors based on the distance between the query
    /// geometry and the indexed geometries. The geometries parameter provides access
    /// to the actual geometric objects stored in the index.
    ///
    /// ```
    /// use geo_index::rtree::{RTreeBuilder, RTreeIndex};
    /// use geo_index::rtree::distance::EuclideanDistance;
    /// use geo_index::rtree::sort::HilbertSort;
    /// use geo::{Point, Geometry};
    ///
    /// // Create an RTree (in practice, you'd store geometries separately)
    /// let mut builder = RTreeBuilder::<f64>::new(3);
    /// builder.add(0., 0., 2., 2.);
    /// builder.add(5., 5., 7., 7.);
    /// builder.add(10., 10., 12., 12.);
    /// let tree = builder.finish::<HilbertSort>();
    ///
    /// // Example geometries corresponding to the bboxes (in practice from your data)
    /// let geometries: Vec<Geometry<f64>> = vec![
    ///     Geometry::Point(Point::new(1.0, 1.0)),
    ///     Geometry::Point(Point::new(6.0, 6.0)),
    ///     Geometry::Point(Point::new(11.0, 11.0)),
    /// ];
    ///
    /// // Query geometry
    /// let query_geom = Geometry::Point(Point::new(3.0, 3.0));
    /// let euclidean = EuclideanDistance;
    ///
    /// let results = tree.neighbors_geometry(&query_geom, None, None, &euclidean, &geometries);
    /// ```
    fn neighbors_geometry(
        &self,
        query_geometry: &geo::Geometry<f64>,
        max_results: Option<usize>,
        max_distance: Option<N>,
        distance_metric: &dyn DistanceMetric<N>,
        geometries: &[geo::Geometry<f64>],
    ) -> Vec<u32> {
        let boxes = self.boxes();
        let indices = self.indices();
        let max_distance = max_distance.unwrap_or(distance_metric.max_distance());

        // Get the bounding box of the query geometry
        let bounds = query_geometry.bounding_rect();
        let (query_min_x, query_min_y, query_max_x, query_max_y) = if let Some(rect) = bounds {
            let min = rect.min();
            let max = rect.max();
            (
                N::from_f64(min.x).unwrap_or(N::zero()),
                N::from_f64(min.y).unwrap_or(N::zero()),
                N::from_f64(max.x).unwrap_or(N::zero()),
                N::from_f64(max.y).unwrap_or(N::zero()),
            )
        } else {
            // If no bounding box, use origin
            (N::zero(), N::zero(), N::zero(), N::zero())
        };

        let mut outer_node_index = Some(boxes.len() - 4);
        let mut queue = BinaryHeap::new();
        let mut results: Vec<u32> = vec![];

        'outer: while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() as usize * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // add child nodes to the queue
            for pos in (node_index..end).step_by(4) {
                let index = indices.get(pos >> 2);

                let dist = if node_index >= self.num_items() as usize * 4 {
                    // For internal nodes, use bbox-to-bbox distance as approximation
                    // Convert query bbox center to point for bbox distance calculation
                    let center_x = (query_min_x + query_max_x) / (N::one() + N::one());
                    let center_y = (query_min_y + query_max_y) / (N::one() + N::one());

                    distance_metric.distance_to_bbox(
                        center_x,
                        center_y,
                        boxes[pos],
                        boxes[pos + 1],
                        boxes[pos + 2],
                        boxes[pos + 3],
                    )
                } else {
                    // For leaf items, use actual geometry-to-geometry distance
                    let item_index = index;
                    if item_index < geometries.len() {
                        distance_metric
                            .geometry_to_geometry_distance(query_geometry, &geometries[item_index])
                    } else {
                        // Fallback to bbox distance if geometry not available
                        let center_x = (query_min_x + query_max_x) / (N::one() + N::one());
                        let center_y = (query_min_y + query_max_y) / (N::one() + N::one());

                        distance_metric.distance_to_bbox(
                            center_x,
                            center_y,
                            boxes[pos],
                            boxes[pos + 1],
                            boxes[pos + 2],
                            boxes[pos + 3],
                        )
                    }
                };

                if dist > max_distance {
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
                if dist > max_distance {
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

    mod distance_metrics {
        use crate::rtree::distance::{EuclideanDistance, HaversineDistance, SpheroidDistance};
        use crate::rtree::sort::HilbertSort;
        use crate::rtree::{RTreeBuilder, RTreeIndex};

        #[test]
        fn test_euclidean_distance_neighbors() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 1., 1.);
            builder.add(2., 2., 3., 3.);
            builder.add(4., 4., 5., 5.);
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            let results = tree.neighbors_with_distance(0., 0., None, None, &euclidean);

            // Should return items in order of distance from (0,0)
            assert_eq!(results, vec![0, 1, 2]);
        }

        #[test]
        fn test_haversine_distance_neighbors() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            // Add some geographic points (longitude, latitude)
            builder.add(-74.0, 40.7, -74.0, 40.7); // New York
            builder.add(-0.1, 51.5, -0.1, 51.5); // London
            builder.add(139.7, 35.7, 139.7, 35.7); // Tokyo
            let tree = builder.finish::<HilbertSort>();

            let haversine = HaversineDistance::default();
            let results = tree.neighbors_with_distance(-74.0, 40.7, None, None, &haversine);

            // From New York, should find New York first, then London, then Tokyo
            assert_eq!(results, vec![0, 1, 2]);
        }

        #[test]
        fn test_spheroid_distance_neighbors() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            // Add some geographic points (longitude, latitude)
            builder.add(-74.0, 40.7, -74.0, 40.7); // New York
            builder.add(-0.1, 51.5, -0.1, 51.5); // London
            builder.add(139.7, 35.7, 139.7, 35.7); // Tokyo
            let tree = builder.finish::<HilbertSort>();

            let spheroid = SpheroidDistance::default();
            let results = tree.neighbors_with_distance(-74.0, 40.7, None, None, &spheroid);

            // From New York, should find New York first, then London, then Tokyo
            assert_eq!(results, vec![0, 1, 2]);
        }

        #[test]
        fn test_backward_compatibility() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 1., 1.);
            builder.add(2., 2., 3., 3.);
            builder.add(4., 4., 5., 5.);
            let tree = builder.finish::<HilbertSort>();

            // Test that original neighbors method still works
            let results_original = tree.neighbors(0., 0., None, None);

            // Test that new method with Euclidean distance gives same results
            let euclidean = EuclideanDistance;
            let results_new = tree.neighbors_with_distance(0., 0., None, None, &euclidean);

            assert_eq!(results_original, results_new);
        }

        #[test]
        fn test_max_distance_filtering() {
            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 1., 1.);
            builder.add(2., 2., 3., 3.);
            builder.add(10., 10., 11., 11.);
            let tree = builder.finish::<HilbertSort>();

            let euclidean = EuclideanDistance;
            // Only find neighbors within distance 5
            let results = tree.neighbors_with_distance(0., 0., None, Some(5.0), &euclidean);

            // Should only find first two items, not the distant third one
            assert_eq!(results.len(), 2);
            assert_eq!(results, vec![0, 1]);
        }

        #[test]
        fn test_geometry_neighbors_euclidean() {
            use geo::{Geometry, Point};

            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 2., 2.); // Item 0
            builder.add(5., 5., 7., 7.); // Item 1
            builder.add(10., 10., 12., 12.); // Item 2
            let tree = builder.finish::<HilbertSort>();

            // Geometries corresponding to the bboxes
            let geometries: Vec<Geometry<f64>> = vec![
                Geometry::Point(Point::new(1.0, 1.0)),   // Item 0
                Geometry::Point(Point::new(6.0, 6.0)),   // Item 1
                Geometry::Point(Point::new(11.0, 11.0)), // Item 2
            ];

            let query_geom = Geometry::Point(Point::new(3.0, 3.0));
            let euclidean = EuclideanDistance;
            let results = tree.neighbors_geometry(&query_geom, None, None, &euclidean, &geometries);

            // Item 0 should be closest to query point (3,3)
            assert_eq!(results[0], 0);
            assert_eq!(results[1], 1);
            assert_eq!(results[2], 2);
        }

        #[test]
        fn test_geometry_neighbors_linestring() {
            use geo::{Geometry, LineString, Point};
            use geo_types::coord;

            let mut builder = RTreeBuilder::<f64>::new(3);
            builder.add(0., 0., 10., 0.); // Item 0 - horizontal line
            builder.add(5., 5., 15., 5.); // Item 1 - horizontal line higher up
            builder.add(0., 10., 10., 10.); // Item 2 - horizontal line at top
            let tree = builder.finish::<HilbertSort>();

            // Geometries corresponding to the bboxes
            let geometries: Vec<Geometry<f64>> = vec![
                Geometry::LineString(LineString::new(vec![
                    coord! { x: 0.0, y: 0.0 },
                    coord! { x: 10.0, y: 0.0 },
                ])),
                Geometry::LineString(LineString::new(vec![
                    coord! { x: 5.0, y: 5.0 },
                    coord! { x: 15.0, y: 5.0 },
                ])),
                Geometry::LineString(LineString::new(vec![
                    coord! { x: 0.0, y: 10.0 },
                    coord! { x: 10.0, y: 10.0 },
                ])),
            ];

            let query_geom = Geometry::Point(Point::new(5.0, 2.0));
            let euclidean = EuclideanDistance;
            let results = tree.neighbors_geometry(&query_geom, None, None, &euclidean, &geometries);

            // Item 0 (bottom line) should be closest to point (5, 2)
            assert_eq!(results[0], 0);
        }

        #[test]
        fn test_geometry_neighbors_with_max_results() {
            use geo::{Geometry, Point};

            let mut builder = RTreeBuilder::<f64>::new(5);
            for i in 0..5 {
                let x = (i * 3) as f64;
                builder.add(x, x, x + 1., x + 1.);
            }
            let tree = builder.finish::<HilbertSort>();

            // Create geometries for each bbox
            let geometries: Vec<Geometry<f64>> = (0..5)
                .map(|i| {
                    let x = (i * 3) as f64;
                    Geometry::Point(Point::new(x + 0.5, x + 0.5))
                })
                .collect();

            let query_geom = Geometry::Point(Point::new(5.0, 5.0));
            let euclidean = EuclideanDistance;
            let results =
                tree.neighbors_geometry(&query_geom, Some(3), None, &euclidean, &geometries);

            assert_eq!(results.len(), 3);
            // Should get the 3 closest items
        }

        #[test]
        fn test_geometry_neighbors_haversine() {
            use geo::{Geometry, Point};

            let mut builder = RTreeBuilder::<f64>::new(3);
            // Geographic bounding boxes (lon, lat)
            builder.add(-74.1, 40.6, -74.0, 40.7); // New York area
            builder.add(-0.2, 51.4, -0.1, 51.5); // London area
            builder.add(139.6, 35.6, 139.7, 35.7); // Tokyo area
            let tree = builder.finish::<HilbertSort>();

            let geometries: Vec<Geometry<f64>> = vec![
                Geometry::Point(Point::new(-74.0, 40.7)), // New York
                Geometry::Point(Point::new(-0.1, 51.5)),  // London
                Geometry::Point(Point::new(139.7, 35.7)), // Tokyo
            ];

            let query_geom = Geometry::Point(Point::new(-74.0, 40.7)); // New York
            let haversine = HaversineDistance::default();
            let results = tree.neighbors_geometry(&query_geom, None, None, &haversine, &geometries);

            // New York should be closest (distance 0)
            assert_eq!(results[0], 0);
        }
    }
}
