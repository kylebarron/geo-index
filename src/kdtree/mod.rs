//! An implementation of a static, ABI-stable K-D Tree.

mod builder;
pub(crate) mod constants;
mod index;
mod r#trait;

pub use builder::KDTreeBuilder;
pub use index::{KDTreeRef, OwnedKDTree};
pub use r#trait::KDTreeIndex;

#[cfg(test)]
mod test;
