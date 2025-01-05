//! Utilities to traverse the KDTree structure.

use geo_traits::RectTrait;

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

    right_child: usize,
    left_child: usize,

    phantom: PhantomData<N>,

    min_x: N,
    min_y: N,
    max_x: N,
    max_y: N,
}

impl<'a, N: IndexableNum, T: KDTreeIndex<N>> Node<'a, N, T> {
    pub(crate) fn from_root(tree: &'a T) -> Self {
        Self {
            tree,
            axis: 0,
            right_child: tree.indices().len() - 1,
            left_child: 0,
            phantom: PhantomData,
            min_x: N::max_value(),
            min_y: N::max_value(),
            max_x: N::min_value(),
            max_y: N::min_value(),
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
    /// Note that this **does not include** the middle index of the current node.
    pub fn left_child(&self) -> Node<'_, N, T> {
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
    /// Note that this **does not include** the middle index of the current node.
    pub fn right_child(&self) -> Node<'_, N, T> {
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
}

impl<N: IndexableNum, T: KDTreeIndex<N>> RectTrait for Node<'_, N, T> {
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
