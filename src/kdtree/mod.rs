//! An implementation of an immutable, ABI-stable K-D Tree.

#![warn(missing_docs)]

mod builder;
pub(crate) mod constants;
mod index;
mod r#trait;

pub use builder::KDTreeBuilder;
pub use index::{KDTreeRef, OwnedKDTree};
pub use r#trait::KDTreeIndex;

#[cfg(test)]
mod test;
