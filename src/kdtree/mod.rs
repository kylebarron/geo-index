//! An immutable, ABI-stable K-D Tree.

mod builder;
pub(crate) mod constants;
mod index;
mod r#trait;

pub use builder::KDTreeBuilder;
pub use index::{KDTreeMetadata, KDTreeRef, OwnedKDTree};
pub use r#trait::KDTreeIndex;

#[cfg(test)]
mod test;
