//! An implementation of a static, ABI-stable RTree.

mod builder;
mod constants;
mod index;
pub mod sort;
mod r#trait;
pub mod traversal;
pub mod util;

pub use builder::RTreeBuilder;
pub use index::{OwnedRTree, RTreeRef};
pub use r#trait::RTreeIndex;
