use std::fmt::Debug;
use thiserror::Error;

/// Enum with all errors in this crate.
#[derive(Error, Debug)]
pub enum GeoIndexError {
    /// General errors
    #[error("General error: {0}")]
    General(String),
}

pub type Result<T> = std::result::Result<T, GeoIndexError>;
