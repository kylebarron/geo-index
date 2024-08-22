use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
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
