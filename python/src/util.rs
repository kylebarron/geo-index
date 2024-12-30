use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::{ArrayRef, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::alloc::Allocation;
use arrow_buffer::{ArrowNativeType, Buffer, ScalarBuffer};

pub(crate) fn slice_to_arrow<T: ArrowPrimitiveType>(
    slice: &[T::Native],
    owner: Arc<dyn Allocation>,
) -> ArrayRef {
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
