use crate::indices::MutableIndices;
use crate::r#type::IndexableNum;

/// Parameters that are passed in to the `sort` function of the `Sort` trait.
pub struct SortParams<N: IndexableNum> {
    /// The number of items in this RTree
    pub num_items: usize,
    /// The node size of this RTree
    pub node_size: usize,
    /// The global min_x of this RTree
    pub min_x: N,
    /// The global min_y of this RTree
    pub min_y: N,
    /// The global max_x of this RTree
    pub max_x: N,
    /// The global max_y of this RTree
    pub max_y: N,
}

/// A trait for sorting the elements of an RTree.
pub trait Sort<N: IndexableNum> {
    /// Sort the mutable slice of `boxes` and `indices`.
    ///
    /// ## Invariants:
    ///
    /// - Each element in `boxes` consists of four numbers.
    /// - Each element in `boxes` is ordered `[min_x, min_y, max_x, max_y]`.
    /// - The relative order of elements within `boxes` and `indices` must be maintained. If you're
    ///   swapping the first box with the second box, you must also swap the first index with the
    ///   second index.
    fn sort(sort_params: &mut SortParams<N>, boxes: &mut [N], indices: &mut MutableIndices);
}
