//! An immutable, ABI-stable K-D Tree.
//!
//! ## Creation
//!
//! Use [`KDTreeBuilder`] to construct an [`KDTree`], which allows you to make queries.
//!
//! ## Search
//!
//! Use [`KDTreeIndex::range`] to search a KDTree given a bounding box query. Use
//! [`KDTreeIndex::within`] to search a KDTree given a point and radius.
//!
//! ## Persisting
//!
//! You can use [`KDTree::into_inner`] to access the underlying `Vec<u8>` it contains.
//!
//! ## Recovering the index
//!
//! You can use [`KDTreeRef::try_new`] to construct a KDTree as a reference on an external byte
//! slice. If you don't know the coordinate type used in the index, you can use
//! [`CoordType::from_buffer`][crate::CoordType::from_buffer] to infer the coordinate type.
//!
//! ## Coordinate types
//!
//! Supported coordinate types implement [`IndexableNum`][crate::IndexableNum]. Note that float
//! `NaN` is not supported and may panic.
//!
//! ## Example
//!
//! ```
//! use geo_index::kdtree::{KDTreeBuilder, KDTreeIndex, KDTreeRef};
//!
//! // Create a KDTree
//! let mut builder = KDTreeBuilder::<f64>::new(3);
//! builder.add(0., 0.);
//! builder.add(1., 1.);
//! builder.add(2., 2.);
//! let tree = builder.finish();
//!
//! // Perform a search
//! assert_eq!(tree.range(0.5, 0.5, 1.5, 1.5), vec![1]);
//!
//! // Convert to underlying buffer
//! let buffer = tree.into_inner();
//!
//! // Create tree as a reference onto this buffer
//! let tree_ref = KDTreeRef::<f64>::try_new(&buffer).unwrap();
//!
//! // Perform search again
//! assert_eq!(tree_ref.range(0.5, 0.5, 1.5, 1.5), vec![1]);
//! ```

mod builder;
pub(crate) mod constants;
mod index;
mod r#trait;

pub use builder::{KDTreeBuilder, DEFAULT_KDTREE_NODE_SIZE};
pub use index::{KDTreeMetadata, KDTreeRef, KDTree};
pub use r#trait::KDTreeIndex;

#[cfg(test)]
mod test;
