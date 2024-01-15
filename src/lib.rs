#![doc = include_str!("../README.md")]

mod error;
pub mod indices;
pub mod kdtree;
pub mod rtree;
pub mod r#type;

pub use error::GeoIndexError;

#[cfg(test)]
pub(crate) mod test;
