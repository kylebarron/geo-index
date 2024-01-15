#![doc = include_str!("../README.md")]

pub mod indices;
pub mod kdtree;
pub mod rtree;
pub mod r#type;

pub use kdtree::{KdbushBuilder, KdbushIndex, KdbushRef, OwnedKdbush};
pub use rtree::{OwnedRTree, RTreeBuilder, RTreeIndex, RTreeRef};

#[cfg(test)]
pub(crate) mod test;
