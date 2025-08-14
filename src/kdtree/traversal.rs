//! Utilities to traverse the KDTree structure.

use geo_traits::{
    GeometryTrait, RectTrait, UnimplementedGeometryCollection, UnimplementedLine,
    UnimplementedLineString, UnimplementedMultiLineString, UnimplementedMultiPoint,
    UnimplementedMultiPolygon, UnimplementedPoint, UnimplementedPolygon, UnimplementedTriangle,
};

use crate::kdtree::KDTreeIndex;
use crate::r#type::{Coord, IndexableNum};
use std::marker::PhantomData;

/// An internal node in the KDTree.
#[derive(Debug, Clone)]
pub struct Node<'a, N: IndexableNum, T: KDTreeIndex<N>> {
    /// The tree that this node is a reference onto
    tree: &'a T,

    /// The axis that the children of this node are split over.
    /// 0 for x axis, 1 for y axis
    /// TODO: switch to bool
    axis: usize,

    /// The index of the right child.
    right_child: usize,

    /// The index of the left child.
    left_child: usize,

    /// The min_x of this node.
    min_x: N,
    /// The min_y of this node.
    min_y: N,
    /// The max_x of this node.
    max_x: N,
    /// The max_y of this node.
    max_y: N,

    phantom: PhantomData<N>,
}

impl<'a, N: IndexableNum, T: KDTreeIndex<N>> Node<'a, N, T> {
    pub(crate) fn from_root(tree: &'a T) -> Self {
        Self {
            tree,
            axis: 0,
            right_child: tree.indices().len() - 1,
            left_child: 0,
            min_x: N::max_value(),
            min_y: N::max_value(),
            max_x: N::min_value(),
            max_y: N::min_value(),
            phantom: PhantomData,
        }
    }

    // TODO: perhaps this should be state stored on the node, so we don't have to recompute it
    // But it's only valid for nodes that have children
    /// Note: this is the index into the coords array, not the insertion index.
    #[inline]
    pub(crate) fn middle_index(&self) -> usize {
        (self.left_child + self.right_child) >> 1
    }

    // TODO: perhaps this should be state stored on the node, so we don't have to recompute it
    // But it's only valid for nodes that have children
    #[inline]
    pub(crate) fn middle_xy(&self, m: usize) -> (N, N) {
        let x = self.tree.coords()[2 * m];
        let y = self.tree.coords()[2 * m + 1];
        (x, y)
    }

    /// The child node representing the "left" half.
    ///
    /// Returns `None` if [`Self::is_parent`] is `false`.
    pub fn left_child(&self) -> Option<Node<'_, N, T>> {
        if self.is_parent() {
            Some(self.left_child_unchecked())
        } else {
            None
        }
    }

    /// The child node representing the "left" half.
    ///
    /// Note that this **does not include** the middle index of the current node.
    pub fn left_child_unchecked(&self) -> Node<'_, N, T> {
        debug_assert!(self.is_parent());

        let m = self.middle_index();
        let (x, y) = self.middle_xy(m);

        let mut max_x = self.max_x;
        let mut max_y = self.max_y;
        if self.axis == 0 {
            max_x = x;
        } else {
            max_y = y;
        };

        Self {
            tree: self.tree,
            axis: 1 - self.axis,
            right_child: m - 1,
            left_child: self.left_child,
            min_x: self.min_x,
            min_y: self.min_y,
            max_x,
            max_y,
            phantom: self.phantom,
        }
    }

    /// The child node representing the "right" half.
    ///
    /// Returns `None` if [`Self::is_parent`] is `false`.
    pub fn right_child(&self) -> Option<Node<'_, N, T>> {
        if self.is_parent() {
            Some(self.right_child_unchecked())
        } else {
            None
        }
    }

    /// The child node representing the "right" half.
    ///
    /// Note that this **does not include** the middle index of the current node.
    pub fn right_child_unchecked(&self) -> Node<'_, N, T> {
        debug_assert!(self.is_parent());

        let m = self.middle_index();
        let (x, y) = self.middle_xy(m);

        let mut min_x = self.min_x;
        let mut min_y = self.min_y;
        if self.axis == 0 {
            min_x = x;
        } else {
            min_y = y;
        };

        Self {
            tree: self.tree,
            axis: 1 - self.axis,
            right_child: self.right_child,
            left_child: m + 1,
            min_x,
            min_y,
            max_x: self.max_x,
            max_y: self.max_y,
            phantom: self.phantom,
        }
    }

    /// Returns `true` if this is a leaf node without children.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.right_child - self.left_child <= self.tree.node_size() as usize
    }

    /// Returns `true` if this is an intermediate node with children.
    #[inline]
    pub fn is_parent(&self) -> bool {
        !self.is_leaf()
    }

    /// The original insertion index of the "middle child" of this node. This is only valid when
    /// this is a parent node, which you can check with `Self::is_parent`.
    ///
    /// Returns `None` if [`Self::is_parent`] is `false`.
    #[inline]
    pub fn middle_insertion_index(&self) -> Option<u32> {
        if self.is_parent() {
            Some(self.middle_insertion_index_unchecked())
        } else {
            None
        }
    }

    /// The original insertion index of the "middle child" of this node. This is only valid when
    /// this is a parent node, which you can check with `Self::is_parent`.
    #[inline]
    pub fn middle_insertion_index_unchecked(&self) -> u32 {
        debug_assert!(self.is_parent());

        let m = self.middle_index();
        let indices = self.tree.indices();
        indices.get(m) as u32
    }

    /// The original insertion indices. This is only valid when this is a leaf node, which you can
    /// check with `Self::is_leaf`.
    ///
    /// Returns `None` if [`Self::is_leaf`] is `false`.
    #[inline]
    pub fn leaf_insertion_indices(&self) -> Option<Vec<u32>> {
        if self.is_leaf() {
            Some(self.leaf_insertion_indices_unchecked())
        } else {
            None
        }
    }

    /// The original insertion indices. This is only valid when this is a leaf node, which you can
    /// check with `Self::is_leaf`.
    #[inline]
    pub fn leaf_insertion_indices_unchecked(&self) -> Vec<u32> {
        debug_assert!(self.is_leaf());

        let mut result = Vec::with_capacity(self.tree.node_size() as _);

        let indices = self.tree.indices();
        for i in self.left_child..=self.right_child {
            result.push(indices.get(i) as u32);
        }

        result
    }
}

impl<N: IndexableNum, T: KDTreeIndex<N>> RectTrait for Node<'_, N, T> {
    type CoordType<'a>
        = Coord<N>
    where
        Self: 'a;

    fn min(&self) -> Self::CoordType<'_> {
        Coord {
            x: self.min_x,
            y: self.min_y,
        }
    }

    fn max(&self) -> Self::CoordType<'_> {
        Coord {
            x: self.max_x,
            y: self.max_y,
        }
    }
}

impl<N: IndexableNum, T: KDTreeIndex<N>> GeometryTrait for Node<'_, N, T> {
    type T = N;

    type PointType<'a>
        = UnimplementedPoint<N>
    where
        Self: 'a;

    type LineStringType<'a>
        = UnimplementedLineString<N>
    where
        Self: 'a;

    type PolygonType<'a>
        = UnimplementedPolygon<N>
    where
        Self: 'a;

    type MultiPointType<'a>
        = UnimplementedMultiPoint<N>
    where
        Self: 'a;

    type MultiLineStringType<'a>
        = UnimplementedMultiLineString<N>
    where
        Self: 'a;

    type MultiPolygonType<'a>
        = UnimplementedMultiPolygon<N>
    where
        Self: 'a;

    type GeometryCollectionType<'a>
        = UnimplementedGeometryCollection<N>
    where
        Self: 'a;

    type RectType<'a>
        = Node<'a, N, T>
    where
        Self: 'a;

    type TriangleType<'a>
        = UnimplementedTriangle<N>
    where
        Self: 'a;

    type LineType<'a>
        = UnimplementedLine<N>
    where
        Self: 'a;

    fn dim(&self) -> geo_traits::Dimensions {
        geo_traits::Dimensions::Xy
    }

    fn as_type(
        &self,
    ) -> geo_traits::GeometryType<
        '_,
        Self::PointType<'_>,
        Self::LineStringType<'_>,
        Self::PolygonType<'_>,
        Self::MultiPointType<'_>,
        Self::MultiLineStringType<'_>,
        Self::MultiPolygonType<'_>,
        Self::GeometryCollectionType<'_>,
        Self::RectType<'_>,
        Self::TriangleType<'_>,
        Self::LineType<'_>,
    > {
        geo_traits::GeometryType::Rect(self)
    }
}
