use std::fmt::Debug;

use geo_traits::CoordTrait;
use num_traits::{Bounded, Num, NumCast};

use crate::kdtree::constants::KDBUSH_MAGIC;
use crate::GeoIndexError;

/// A trait for types that can be used for indexed coordinates.
///
/// This trait is sealed and cannot be implemented for external types. This is because we want to
/// ensure FFI compatibility with other implementations, including the reference implementations in
/// JavaScript ([rtree](https://github.com/mourner/flatbush),
/// [kdtree](https://github.com/mourner/kdbush))
pub trait IndexableNum:
    private::Sealed + Num + NumCast + PartialOrd + Debug + Send + Sync + bytemuck::Pod + Bounded
{
    /// The type index to match the array order of `ARRAY_TYPES` in flatbush JS
    const TYPE_INDEX: u8;
    /// The number of bytes per element
    const BYTES_PER_ELEMENT: usize;

    /// Convert to f64 for distance calculations
    fn to_f64(self) -> Option<f64> {
        NumCast::from(self)
    }

    /// Convert from f64 for distance calculations
    fn from_f64(value: f64) -> Option<Self> {
        NumCast::from(value)
    }

    /// Get the square root of this value
    fn sqrt(self) -> Option<Self> {
        self.to_f64()
            .and_then(|value| {
                if value >= 0.0 {
                    Some(value.sqrt())
                } else {
                    None
                }
            })
            .and_then(NumCast::from)
    }
}

impl IndexableNum for i8 {
    const TYPE_INDEX: u8 = 0;
    const BYTES_PER_ELEMENT: usize = 1;
}

impl IndexableNum for u8 {
    const TYPE_INDEX: u8 = 1;
    const BYTES_PER_ELEMENT: usize = 1;
}

impl IndexableNum for i16 {
    const TYPE_INDEX: u8 = 3;
    const BYTES_PER_ELEMENT: usize = 2;
}

impl IndexableNum for u16 {
    const TYPE_INDEX: u8 = 4;
    const BYTES_PER_ELEMENT: usize = 2;
}

impl IndexableNum for i32 {
    const TYPE_INDEX: u8 = 5;
    const BYTES_PER_ELEMENT: usize = 4;
}

impl IndexableNum for u32 {
    const TYPE_INDEX: u8 = 6;
    const BYTES_PER_ELEMENT: usize = 4;
}

impl IndexableNum for f32 {
    const TYPE_INDEX: u8 = 7;
    const BYTES_PER_ELEMENT: usize = 4;
}

impl IndexableNum for f64 {
    const TYPE_INDEX: u8 = 8;
    const BYTES_PER_ELEMENT: usize = 8;
}

/// For compatibility with JS, which contains a Uint8ClampedArray
const U8_CLAMPED_TYPE_INDEX: u8 = 2;

/// An enum over the allowed coordinate types in the spatial index.
///
/// This can be used to infer the coordinate type from an existing buffer.
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum CoordType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
}

impl CoordType {
    /// Infer the CoordType from an existing buffer.
    ///
    /// This can be used to discern the generic type to use when constructing an `RTreeRef` or
    /// `KDTreeRef`.
    ///
    /// ```
    /// use geo_index::rtree::RTreeBuilder;
    /// use geo_index::rtree::sort::HilbertSort;
    /// use geo_index::CoordType;
    ///
    /// let mut builder = RTreeBuilder::<u32>::new(2);
    /// builder.add(0, 0, 2, 2);
    /// builder.add(1, 1, 3, 3);
    /// let tree = builder.finish::<HilbertSort>();
    ///
    /// let coord_type = CoordType::from_buffer(&tree).unwrap();
    /// assert!(matches!(coord_type, CoordType::UInt32));
    /// ```
    ///
    /// This method works for both buffers representing RTree or KDTree trees.
    pub fn from_buffer<T: AsRef<[u8]>>(data: &T) -> Result<Self, GeoIndexError> {
        let data = data.as_ref();
        let magic = data[0];
        if magic != 0xfb && magic != KDBUSH_MAGIC {
            return Err(GeoIndexError::General(
                "Data not in Flatbush or Kdbush format.".to_string(),
            ));
        }

        let version_and_type = data[1];
        let type_ = version_and_type & 0x0f;
        let result = match type_ {
            i8::TYPE_INDEX => CoordType::Int8,
            u8::TYPE_INDEX => CoordType::UInt8,
            U8_CLAMPED_TYPE_INDEX => CoordType::UInt8,
            i16::TYPE_INDEX => CoordType::Int16,
            u16::TYPE_INDEX => CoordType::UInt16,
            i32::TYPE_INDEX => CoordType::Int32,
            u32::TYPE_INDEX => CoordType::UInt32,
            f32::TYPE_INDEX => CoordType::Float32,
            f64::TYPE_INDEX => CoordType::Float64,
            t => return Err(GeoIndexError::General(format!("Unexpected type {t}."))),
        };
        Ok(result)
    }
}

// https://rust-lang.github.io/api-guidelines/future-proofing.html#sealed-traits-protect-against-downstream-implementations-c-sealed
mod private {
    pub trait Sealed {}

    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for i16 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A single coordinate.
///
/// Used in the implementation of RectTrait for Node.
pub struct Coord<N: IndexableNum> {
    pub(crate) x: N,
    pub(crate) y: N,
}

impl<N: IndexableNum> CoordTrait for Coord<N> {
    type T = N;

    fn dim(&self) -> geo_traits::Dimensions {
        geo_traits::Dimensions::Xy
    }

    fn x(&self) -> Self::T {
        self.x
    }

    fn y(&self) -> Self::T {
        self.y
    }

    fn nth_or_panic(&self, n: usize) -> Self::T {
        match n {
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid index of coord"),
        }
    }
}
