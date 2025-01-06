use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::{ArrayRef, ArrowPrimitiveType, FixedSizeListArray, PrimitiveArray};
use arrow_buffer::alloc::Allocation;
use arrow_buffer::{ArrowNativeType, Buffer, ScalarBuffer};
use arrow_schema::Field;

// TODO: in the future, refactor this to use `Bytes::from_owner` for improved safety.
pub(crate) fn slice_to_arrow<T: ArrowPrimitiveType>(
    slice: &[T::Native],
    owner: Arc<dyn Allocation>,
    copy: bool,
) -> ArrayRef {
    if copy {
        Arc::new(PrimitiveArray::<T>::new(
            ScalarBuffer::from(slice.to_vec()),
            None,
        ))
    } else {
        let ptr = NonNull::new(slice.as_ptr() as *mut _).unwrap();
        let len = slice.len();
        let bytes_len = len * T::Native::get_byte_width();

        // Safety:
        // ptr is a non-null pointer owned by the RTree, which is passed in as the Allocation
        let buffer = unsafe { Buffer::from_custom_allocation(ptr, bytes_len, owner) };
        Arc::new(PrimitiveArray::<T>::new(
            ScalarBuffer::new(buffer, 0, len),
            None,
        ))
    }
}

pub(crate) fn boxes_to_arrow<T: ArrowPrimitiveType>(
    slice: &[T::Native],
    owner: Arc<dyn Allocation>,
    copy: bool,
) -> ArrayRef {
    let values_array = slice_to_arrow::<T>(slice, owner, copy);
    Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", values_array.data_type().clone(), false)),
        4,
        values_array,
        None,
    ))
}
