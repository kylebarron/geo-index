use geo_index::IndexableNum;
use numpy::{dtype_bound, PyArray1, PyArrayDescr, PyUntypedArray};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use std::os::raw::c_int;

/// A Rust buffer that implements the Python buffer protocol
#[pyclass(name = "Buffer")]
pub(crate) struct RustBuffer(Vec<u8>);

impl RustBuffer {
    pub(crate) fn new(buffer: Vec<u8>) -> Self {
        Self(buffer)
    }
}

#[pymethods]
impl RustBuffer {
    /// Implements the buffer protocol export
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let bytes = slf.0.as_slice();
        let ret = ffi::PyBuffer_FillInfo(
            view,
            slf.as_ptr() as *mut _,
            bytes.as_ptr() as *mut _,
            bytes.len().try_into().unwrap(),
            1, // read only
            flags,
        );
        if ret == -1 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // is there anything to do here?
    }
}

/// A Rust representation of a Python object that implements the Python buffer protocol, exporting
/// a 1-dimensional `&[u8]` slice.
pub(crate) struct PyU8Buffer(PyBuffer<u8>);

impl<'py> FromPyObject<'py> for PyU8Buffer {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let buffer = PyBuffer::<u8>::get_bound(obj)?;
        if !buffer.readonly() {
            return Err(PyValueError::new_err("Must be read-only byte buffer."));
        }
        if buffer.dimensions() != 1 {
            return Err(PyValueError::new_err("Expected 1-dimensional array."));
        }
        // Note: this is probably superfluous for 1D array
        if !buffer.is_c_contiguous() {
            return Err(PyValueError::new_err("Expected c-contiguous array."));
        }
        if buffer.len_bytes() == 0 {
            return Err(PyValueError::new_err("Buffer has no data."));
        }

        Ok(Self(buffer))
    }
}

impl AsRef<[u8]> for PyU8Buffer {
    /// Extract a slice from a Python object implementing the buffer protocol
    fn as_ref(&self) -> &[u8] {
        let len = self.0.item_count();
        let data = self.0.buf_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(data, len) }
    }
}

pub(crate) enum PyTypedArrayRef<'py, N: IndexableNum + numpy::Element> {
    // Arrow((ArrayRef, PhantomData<N>)),
    Numpy(&'py PyArray1<N>),
}

impl<'py, N: IndexableNum + numpy::Element> PyTypedArrayRef<'py, N> {
    pub(crate) fn as_slice(&self) -> &[N] {
        match self {
            Self::Numpy(arr) => unsafe { arr.as_slice() }.unwrap(),
        }
    }
}

pub(crate) enum PyArray<'py> {
    Int8(PyTypedArrayRef<'py, i8>),
    Int16(PyTypedArrayRef<'py, i16>),
    Int32(PyTypedArrayRef<'py, i32>),
    UInt8(PyTypedArrayRef<'py, u8>),
    UInt16(PyTypedArrayRef<'py, u16>),
    UInt32(PyTypedArrayRef<'py, u32>),
    Float32(PyTypedArrayRef<'py, f32>),
    Float64(PyTypedArrayRef<'py, f64>),
}

impl<'py> FromPyObject<'py> for PyArray<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let mut ob = ob.to_owned();
        // call __array__ if it exists
        if ob.hasattr("__array__")? {
            ob = ob.call_method0("__array__")?;
        }

        if let Ok(array) = ob.extract::<&'py PyUntypedArray>() {
            if array.ndim() != 1 {
                return Err(PyValueError::new_err("Expected 1-dimensional array."));
            }

            let dtype = array.dtype();

            if is_type::<i8>(dtype) {
                let arr = array.downcast::<PyArray1<i8>>()?;
                return Ok(Self::Int8(PyTypedArrayRef::Numpy(arr)));
            }

            if is_type::<i16>(dtype) {
                let arr = array.downcast::<PyArray1<i16>>()?;
                return Ok(Self::Int16(PyTypedArrayRef::Numpy(arr)));
            }

            if is_type::<i32>(dtype) {
                let arr = array.downcast::<PyArray1<i32>>()?;
                return Ok(Self::Int32(PyTypedArrayRef::Numpy(arr)));
            }

            if is_type::<u8>(dtype) {
                let arr = array.downcast::<PyArray1<u8>>()?;
                return Ok(Self::UInt8(PyTypedArrayRef::Numpy(arr)));
            }

            if is_type::<u16>(dtype) {
                let arr = array.downcast::<PyArray1<u16>>()?;
                return Ok(Self::UInt16(PyTypedArrayRef::Numpy(arr)));
            }

            if is_type::<u32>(dtype) {
                let arr = array.downcast::<PyArray1<u32>>()?;
                return Ok(Self::UInt32(PyTypedArrayRef::Numpy(arr)));
            }

            if is_type::<f32>(dtype) {
                let arr = array.downcast::<PyArray1<f32>>()?;
                return Ok(Self::Float32(PyTypedArrayRef::Numpy(arr)));
            }

            if is_type::<f64>(dtype) {
                let arr = array.downcast::<PyArray1<f64>>()?;
                return Ok(Self::Float64(PyTypedArrayRef::Numpy(arr)));
            }

            return Err(PyTypeError::new_err("Unexpected dtype of numpy array."));
        }

        Err(PyTypeError::new_err("Expected numpy array input."))
    }
}

fn is_type<T: numpy::Element>(dtype: &PyArrayDescr) -> bool {
    Python::with_gil(|py| dtype.is_equiv_to(dtype_bound::<T>(py).as_gil_ref()))
}
