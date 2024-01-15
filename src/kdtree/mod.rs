pub mod builder;
pub mod constants;
pub mod error;
pub mod index;
pub mod r#trait;

pub use builder::KDTreeBuilder;
pub use index::{KDTreeRef, OwnedKDTree};
pub use r#trait::KDTreeIndex;

#[cfg(test)]
mod test;
