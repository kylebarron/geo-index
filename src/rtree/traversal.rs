//! Utilities to traverse the RTree structure.

use geo_traits::{CoordTrait, RectTrait};

use crate::r#type::IndexableNum;
use crate::rtree::RTreeIndex;
use core::mem::take;
use std::marker::PhantomData;

/// An internal node in the RTree.
#[derive(Debug, Clone)]
pub struct Node<'a, N: IndexableNum, T: RTreeIndex<N>> {
    /// The tree that this node is a reference onto
    tree: &'a T,

    /// This points to the position in the full `boxes` slice of the **first** coordinate of
    /// this node. So
    /// ```notest
    /// self.tree.boxes()[self.pos]
    /// ```
    /// accesses the `min_x` coordinate of this node.
    ///
    /// This also relates to the children and the insertion index. When this is `<
    /// self.tree.num_items() * 4`, it means it's a leaf node at the bottom of the tree. In this
    /// case, calling `>> 2` on this finds the original insertion index.
    ///
    /// When this is `>= self.tree.num_items() * 4`, it means it's _not_ a leaf node, and calling
    /// `>> 2` retrieves the `pos` of the first of its children.
    pos: usize,

    phantom: PhantomData<N>,
}

impl<'a, N: IndexableNum, T: RTreeIndex<N>> Node<'a, N, T> {
    fn new(tree: &'a T, pos: usize) -> Self {
        Self {
            tree,
            pos,
            phantom: PhantomData,
        }
    }

    pub(crate) fn from_root(tree: &'a T) -> Self {
        let root_index = tree.boxes().len() - 4;
        Self {
            tree,
            pos: root_index,
            phantom: PhantomData,
        }
    }

    /// Get the minimum `x` value of this node.
    pub fn min_x(&self) -> N {
        self.tree.boxes()[self.pos]
    }

    /// Get the minimum `y` value of this node.
    pub fn min_y(&self) -> N {
        self.tree.boxes()[self.pos + 1]
    }

    /// Get the maximum `x` value of this node.
    pub fn max_x(&self) -> N {
        self.tree.boxes()[self.pos + 2]
    }

    /// Get the maximum `y` value of this node.
    pub fn max_y(&self) -> N {
        self.tree.boxes()[self.pos + 3]
    }

    /// Returns `true` if this is a leaf node without children.
    pub fn is_leaf(&self) -> bool {
        self.pos < self.tree.num_items() * 4
    }

    /// Returns `true` if this is an intermediate node with children.
    pub fn is_parent(&self) -> bool {
        !self.is_leaf()
    }

    /// Returns `true` if this node intersects another node.
    pub fn intersects<T2: RTreeIndex<N>>(&self, other: &Node<N, T2>) -> bool {
        if self.max_x() < other.min_x() {
            return false;
        }

        if self.max_y() < other.min_y() {
            return false;
        }

        if self.min_x() > other.max_x() {
            return false;
        }

        if self.min_y() > other.max_y() {
            return false;
        }

        true
    }

    /// Returns an iterator over the child nodes of this node. This must only be called if
    /// `is_parent` is `true`.
    pub fn children(&self) -> impl Iterator<Item = Node<'_, N, T>> {
        debug_assert!(self.is_parent());

        // find the start and end indexes of the children of this node
        let start_child_pos = self.tree.indices().get(self.pos >> 2);
        let end_children_pos = (start_child_pos + self.tree.node_size() * 4)
            .min(upper_bound(start_child_pos, self.tree.level_bounds()));

        (start_child_pos..end_children_pos)
            .step_by(4)
            .map(|pos| Node::new(self.tree, pos))
    }

    /// The original insertion index. This is only valid when this is a leaf node, which you can
    /// check with `Self::is_leaf`.
    pub fn index(&self) -> usize {
        debug_assert!(self.is_leaf());
        self.tree.indices().get(self.pos >> 2)
    }
}

/// A single coordinate.
///
/// Used in the implementation of RectTrait for Node.
pub struct Coord<N: IndexableNum> {
    x: N,
    y: N,
}

impl<N: IndexableNum> CoordTrait for Coord<N> {
    type T = N;

    fn dim(&self) -> geo_traits::Dimensions {
        geo_traits::Dimensions::Xy
    }

    fn x(&self) -> Self::T {
        self.x
    }

    fn y(&self) -> Self::T {
        self.y
    }

    fn nth_or_panic(&self, n: usize) -> Self::T {
        match n {
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid index of coord"),
        }
    }
}

impl<N: IndexableNum, T: RTreeIndex<N>> RectTrait for Node<'_, N, T> {
    type T = N;
    type CoordType<'a>
        = Coord<N>
    where
        Self: 'a;

    fn dim(&self) -> geo_traits::Dimensions {
        geo_traits::Dimensions::Xy
    }

    fn min(&self) -> Self::CoordType<'_> {
        Coord {
            x: self.min_x(),
            y: self.min_y(),
        }
    }

    fn max(&self) -> Self::CoordType<'_> {
        Coord {
            x: self.max_x(),
            y: self.max_y(),
        }
    }
}

// This is copied from rstar under the MIT/Apache 2 license
// https://github.com/georust/rstar/blob/6c23af0f3acc0c4668ce6c368820e0fa986a65b4/rstar/src/algorithm/intersection_iterator.rs
pub(crate) struct IntersectionIterator<'a, N, T1, T2>
where
    N: IndexableNum,
    T1: RTreeIndex<N>,
    T2: RTreeIndex<N>,
{
    left: &'a T1,
    right: &'a T2,
    todo_list: Vec<(usize, usize)>,
    candidates: Vec<usize>,
    phantom: PhantomData<N>,
}

impl<'a, N, T1, T2> IntersectionIterator<'a, N, T1, T2>
where
    N: IndexableNum,
    T1: RTreeIndex<N>,
    T2: RTreeIndex<N>,
{
    pub(crate) fn from_trees(root1: &'a T1, root2: &'a T2) -> Self {
        let mut intersections = IntersectionIterator {
            left: root1,
            right: root2,
            todo_list: Vec::new(),
            candidates: Vec::new(),
            phantom: PhantomData,
        };
        intersections.add_intersecting_children(&root1.root(), &root2.root());
        intersections
    }

    #[allow(dead_code)]
    pub(crate) fn new(root1: &'a Node<N, T1>, root2: &'a Node<N, T2>) -> Self {
        let mut intersections = IntersectionIterator {
            left: root1.tree,
            right: root2.tree,
            todo_list: Vec::new(),
            candidates: Vec::new(),
            phantom: PhantomData,
        };
        intersections.add_intersecting_children(root1, root2);
        intersections
    }

    fn push_if_intersecting(&mut self, node1: &'_ Node<N, T1>, node2: &'_ Node<N, T2>) {
        if node1.intersects(node2) {
            self.todo_list.push((node1.pos, node2.pos));
        }
    }

    fn add_intersecting_children(&mut self, parent1: &'_ Node<N, T1>, parent2: &'_ Node<N, T2>) {
        if !parent1.intersects(parent2) {
            return;
        }

        let children1 = parent1.children().filter(|c1| c1.intersects(parent2));

        let mut children2 = take(&mut self.candidates);
        children2.extend(
            parent2
                .children()
                .filter(|c2| c2.intersects(parent1))
                .map(|c| c.pos),
        );

        for child1 in children1 {
            for child2 in &children2 {
                self.push_if_intersecting(&child1, &Node::new(self.right, *child2));
            }
        }

        children2.clear();
        self.candidates = children2;
    }
}

impl<N, T1, T2> Iterator for IntersectionIterator<'_, N, T1, T2>
where
    N: IndexableNum,
    T1: RTreeIndex<N>,
    T2: RTreeIndex<N>,
{
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((left_index, right_index)) = self.todo_list.pop() {
            let left = Node::new(self.left, left_index);
            let right = Node::new(self.right, right_index);
            match (left.is_leaf(), right.is_leaf()) {
                (true, true) => return Some((left.index(), right.index())),
                (true, false) => right
                    .children()
                    .for_each(|c| self.push_if_intersecting(&left, &c)),
                (false, true) => left
                    .children()
                    .for_each(|c| self.push_if_intersecting(&c, &right)),
                (false, false) => self.add_intersecting_children(&left, &right),
            }
        }
        None
    }
}

/**
 * Binary search for the first value in the array bigger than the given.
 * @param {number} value
 * @param {number[]} arr
 */
#[inline]
fn upper_bound(value: usize, arr: &[usize]) -> usize {
    let mut i = 0;
    let mut j = arr.len() - 1;

    while i < j {
        let m = (i + j) >> 1;
        if arr[m] > value {
            j = m;
        } else {
            i = m + 1;
        }
    }

    arr[i]
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::flatbush_js_test_index;

    #[test]
    fn test_node() {
        let tree = flatbush_js_test_index();

        let top_box = tree.boxes_at_level(2).unwrap();

        // Should only be one box
        assert_eq!(top_box.len(), 4);

        // Root node should match that one box in the top level
        let root_node = tree.root();
        assert_eq!(root_node.min_x(), top_box[0]);
        assert_eq!(root_node.min_y(), top_box[1]);
        assert_eq!(root_node.max_x(), top_box[2]);
        assert_eq!(root_node.max_y(), top_box[3]);

        assert!(root_node.is_parent());

        let level_1_boxes = tree.boxes_at_level(1).unwrap();
        let level_1 = root_node.children().collect::<Vec<_>>();
        assert_eq!(level_1.len(), level_1_boxes.len() / 4);
    }
}

#[cfg(test)]
mod test_issue_42 {
    use std::collections::HashSet;

    use crate::rtree::sort::HilbertSort;
    use crate::rtree::{RTreeBuilder, RTreeIndex};
    use geo::Polygon;
    use geo::{BoundingRect, Geometry};
    use geozero::geo_types::GeoWriter;
    use geozero::geojson::read_geojson_fc;
    use rstar::primitives::GeomWithData;
    use rstar::{primitives::Rectangle, AABB};
    use zip::ZipArchive;

    // Find tree self-intersection canddiates using rstar
    fn geo_contiguity(geom: &[Polygon]) -> HashSet<(usize, usize)> {
        let to_insert = geom
            .iter()
            .enumerate()
            .map(|(i, gi)| {
                let rect = gi.bounding_rect().unwrap();
                let aabb =
                    AABB::from_corners([rect.min().x, rect.min().y], [rect.max().x, rect.max().y]);

                GeomWithData::new(Rectangle::from_aabb(aabb), i)
            })
            .collect::<Vec<_>>();

        let tree = rstar::RTree::bulk_load(to_insert);
        let candidates = tree
            .intersection_candidates_with_other_tree(&tree)
            .map(|(left_candidate, right_candidate)| (left_candidate.data, right_candidate.data));

        HashSet::from_iter(candidates)
    }

    // Find tree self-intersection canddiates using geo-index
    fn geo_index_contiguity(geoms: &Vec<Polygon>, node_size: u16) -> HashSet<(usize, usize)> {
        let mut tree_builder = RTreeBuilder::new_with_node_size(geoms.len() as _, node_size);
        for geom in geoms {
            tree_builder.add_rect(&geom.bounding_rect().unwrap());
        }
        let tree = tree_builder.finish::<HilbertSort>();

        let candidates = tree.intersection_candidates_with_other_tree(&tree);

        HashSet::from_iter(candidates)
    }

    #[test]
    fn test_repro_issue_42() {
        let file = std::fs::File::open("fixtures/issue_42.geojson.zip").unwrap();
        let mut zip_archive = ZipArchive::new(file).unwrap();
        let zipped_file = zip_archive.by_name("guerry.geojson").unwrap();
        let reader = std::io::BufReader::new(zipped_file);

        let mut geo_writer = GeoWriter::new();
        read_geojson_fc(reader, &mut geo_writer).unwrap();

        let geoms = match geo_writer.take_geometry().unwrap() {
            Geometry::GeometryCollection(gc) => gc.0,
            _ => panic!(),
        };

        let mut polys = vec![];
        for geom in geoms {
            let poly = match geom {
                Geometry::Polygon(poly) => poly,
                _ => panic!(),
            };
            polys.push(poly);
        }

        let geo_index_self_intersection = geo_index_contiguity(&polys, 10);
        let geo_self_intersection = geo_contiguity(&polys);

        assert_eq!(
            geo_index_self_intersection, geo_self_intersection,
            "The two intersections should match!"
        );
    }
}
