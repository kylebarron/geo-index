use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::util::f64_box_to_f32;
use geo_index::rtree::{OwnedRTree, RTreeBuilder, RTreeIndex};
use geo_index::IndexableNum;
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyBufferError, PyIndexError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
#[cfg(any(Py_3_11, not(Py_LIMITED_API)))]
#[path = ""]
mod buffer_protocol_deps {
    pub use pyo3::ffi;
    
    pub use std::ffi::{c_void, CString};
    pub use std::os::raw::c_int;
    pub use std::ptr;
}
#[cfg(any(Py_3_11, not(Py_LIMITED_API)))]
pub use buffer_protocol_deps::*;


pub enum RTreeMethod {
    Hilbert,
    STR,
}

impl<'a> FromPyObject<'a> for RTreeMethod {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "hilbert" => Ok(Self::Hilbert),
            "str" => Ok(Self::STR),
            _ => Err(PyValueError::new_err(
                "Unexpected method. Should be one of 'hilbert' or 'str'.",
            )),
        }
    }
}

enum RTreeInner {
    Float32(OwnedRTree<f32>),
    Float64(OwnedRTree<f64>),
}

/// These constructors are separated out so they can be generic
fn new_interleaved<N: IndexableNum + numpy::Element>(
    boxes: &ArrayView2<N>,
    method: RTreeMethod,
    node_size: Option<usize>,
) -> OwnedRTree<N> {
    let shape = boxes.shape();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[1], 4);

    let num_items = shape[0];

    let mut builder = if let Some(node_size) = node_size {
        RTreeBuilder::new_with_node_size(num_items, node_size)
    } else {
        RTreeBuilder::new(num_items)
    };

    for i in 0..num_items {
        builder.add(
            *boxes.get((i, 0)).unwrap(),
            *boxes.get((i, 1)).unwrap(),
            *boxes.get((i, 2)).unwrap(),
            *boxes.get((i, 3)).unwrap(),
        );
    }

    match method {
        RTreeMethod::Hilbert => builder.finish::<HilbertSort>(),
        RTreeMethod::STR => builder.finish::<STRSort>(),
    }
}

fn new_separated<N: IndexableNum + numpy::Element>(
    min_x: ArrayView1<N>,
    min_y: ArrayView1<N>,
    max_x: ArrayView1<N>,
    max_y: ArrayView1<N>,
    method: RTreeMethod,
    node_size: Option<usize>,
) -> OwnedRTree<N> {
    assert_eq!(min_x.len(), min_y.len());
    assert_eq!(min_x.len(), max_x.len());
    assert_eq!(min_x.len(), max_y.len());

    let num_items = min_x.len();

    let mut builder = if let Some(node_size) = node_size {
        RTreeBuilder::new_with_node_size(num_items, node_size)
    } else {
        RTreeBuilder::new(num_items)
    };

    for i in 0..num_items {
        builder.add(
            *min_x.get(i).unwrap(),
            *min_y.get(i).unwrap(),
            *max_x.get(i).unwrap(),
            *max_y.get(i).unwrap(),
        );
    }

    match method {
        RTreeMethod::Hilbert => builder.finish::<HilbertSort>(),
        RTreeMethod::STR => builder.finish::<STRSort>(),
    }
}

impl RTreeInner {
    fn num_items(&self) -> usize {
        match self {
            Self::Float32(index) => index.num_items(),
            Self::Float64(index) => index.num_items(),
        }
    }

    fn num_nodes(&self) -> usize {
        match self {
            Self::Float32(index) => index.num_nodes(),
            Self::Float64(index) => index.num_nodes(),
        }
    }

    fn node_size(&self) -> usize {
        match self {
            Self::Float32(index) => index.node_size(),
            Self::Float64(index) => index.node_size(),
        }
    }

    fn num_levels(&self) -> usize {
        match self {
            Self::Float32(index) => index.num_levels(),
            Self::Float64(index) => index.num_levels(),
        }
    }

    fn num_bytes(&self) -> usize {
        match self {
            Self::Float32(index) => AsRef::as_ref(index).len(),
            Self::Float64(index) => AsRef::as_ref(index).len(),
        }
    }
    fn buffer(&self) -> &[u8] {
        match self {
            Self::Float32(index) => AsRef::<[u8]>::as_ref(index),
            Self::Float64(index) => AsRef::<[u8]>::as_ref(index),
        }
    }

    fn boxes_at_level<'py>(&'py self, py: Python<'py>, level: usize) -> PyResult<PyObject> {
        match self {
            Self::Float32(index) => {
                let boxes = index
                    .boxes_at_level(level)
                    .map_err(|err| PyIndexError::new_err(err.to_string()))?;
                let array = PyArray1::from_slice_bound(py, boxes);
                Ok(array.reshape([boxes.len() / 4, 4])?.into_py(py))
            }
            Self::Float64(index) => {
                let boxes = index
                    .boxes_at_level(level)
                    .map_err(|err| PyIndexError::new_err(err.to_string()))?;
                let array = PyArray1::from_slice_bound(py, boxes);
                Ok(array.reshape([boxes.len() / 4, 4])?.into_py(py))
            }
        }
    }
}

#[pyclass]
pub struct RTree(RTreeInner);

// TODO: add support for constructing from a buffer. Need to be able to construct (and validate) an
// OwnedRTree
impl<'a> FromPyObject<'a> for RTree {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let result = {

            let inner_res = OwnedRTree::<f64>::try_new(ob.extract()?).map_err(|e| {
                e
            }).and_then(|inner| {
                Ok(Self(RTreeInner::Float64(inner)))
            })
            .or_else(|f64_err| {
                let inner = OwnedRTree::<f32>::try_new(ob.extract()?).or_else(|e| {
                    Err(PyTypeError::new_err(format!("{e} {f64_err}")))
                })?;
                Ok(Self(RTreeInner::Float32(inner)))
            });
            inner_res
        };
        result
    }
}

#[pymethods]
impl RTree {
    #[classmethod]
    #[pyo3(
        signature = (boxes, *, method = RTreeMethod::Hilbert, node_size = None),
        text_signature = "(boxes, *, method = 'hilbert', node_size = None)")
    ]
    pub fn from_interleaved(
        _cls: &Bound<PyType>,
        py: Python,
        boxes: PyObject,
        method: RTreeMethod,
        node_size: Option<usize>,
    ) -> PyResult<Self> {
        // Convert to numpy array (of the same dtype)
        let boxes = boxes.call_method0(py, intern!(py, "__array__"))?;

        let result = if let Ok(boxes) = boxes.extract::<PyReadonlyArray2<f64>>(py) {
            let boxes = boxes.as_array();
            let tree = py.allow_threads(|| new_interleaved(&boxes, method, node_size));
            Ok(Self(RTreeInner::Float64(tree)))
        } else if let Ok(boxes) = boxes.extract::<PyReadonlyArray2<f32>>(py) {
            let boxes = boxes.as_array();
            let tree = py.allow_threads(|| new_interleaved(&boxes, method, node_size));
            Ok(Self(RTreeInner::Float32(tree)))
        } else {
            let dtype = boxes.call_method0(py, intern!(py, "dtype"))?.to_string();
            Err(PyTypeError::new_err(format!(
                "Expected a numpy array of dtype float32 or float64, got {}",
                dtype
            )))
        };
        result
    }

    #[classmethod]
    #[pyo3(
        signature = (min_x, min_y, max_x, max_y, *, method = RTreeMethod::Hilbert, node_size = None),
        text_signature = "(min_x, min_y, max_x, max_y, *, method = 'hilbert', node_size = None)")
    ]
    #[allow(clippy::too_many_arguments)]
    pub fn from_separated(
        _cls: &Bound<PyType>,
        py: Python,
        min_x: PyObject,
        min_y: PyObject,
        max_x: PyObject,
        max_y: PyObject,
        method: RTreeMethod,
        node_size: Option<usize>,
    ) -> PyResult<Self> {
        // Convert to numpy array (of the same dtype)
        let min_x = min_x.call_method0(py, intern!(py, "__array__"))?;
        let min_y = min_y.call_method0(py, intern!(py, "__array__"))?;
        let max_x = max_x.call_method0(py, intern!(py, "__array__"))?;
        let max_y = max_y.call_method0(py, intern!(py, "__array__"))?;

        let result = if let Ok(min_x) = min_x.extract::<PyReadonlyArray1<f64>>(py) {
            let min_y = min_y.extract::<PyReadonlyArray1<f64>>(py)?;
            let max_x = max_x.extract::<PyReadonlyArray1<f64>>(py)?;
            let max_y = max_y.extract::<PyReadonlyArray1<f64>>(py)?;

            let min_x_array = min_x.as_array();
            let min_y_array = min_y.as_array();
            let max_x_array = max_x.as_array();
            let max_y_array = max_y.as_array();

            let tree = py.allow_threads(|| {
                new_separated(
                    min_x_array,
                    min_y_array,
                    max_x_array,
                    max_y_array,
                    method,
                    node_size,
                )
            });
            Ok(Self(RTreeInner::Float64(tree)))
        } else if let Ok(min_x) = min_x.extract::<PyReadonlyArray1<f32>>(py) {
            let min_y = min_y.extract::<PyReadonlyArray1<f32>>(py)?;
            let max_x = max_x.extract::<PyReadonlyArray1<f32>>(py)?;
            let max_y = max_y.extract::<PyReadonlyArray1<f32>>(py)?;

            let min_x_array = min_x.as_array();
            let min_y_array = min_y.as_array();
            let max_x_array = max_x.as_array();
            let max_y_array = max_y.as_array();

            let tree = py.allow_threads(|| {
                new_separated(
                    min_x_array,
                    min_y_array,
                    max_x_array,
                    max_y_array,
                    method,
                    node_size,
                )
            });
            Ok(Self(RTreeInner::Float32(tree)))
        } else {
            let dtype = min_x.call_method0(py, intern!(py, "dtype"))?.to_string();
            Err(PyTypeError::new_err(format!(
                "Expected a numpy array of dtype float32 or float64, got {}",
                dtype
            )))
        };
        result
    }

    #[classmethod]
    #[pyo3(
        signature = (buf),
        text_signature = "(buf)")
    ]
    #[allow(clippy::too_many_arguments)]
    pub fn from_buffer(
        _cls: &PyType,
        py: Python,
        buf: PyObject,
    ) -> PyResult<Self> {
        buf.extract::<RTree>(py)
    }
    #[pyo3(
        signature = (),
        text_signature = "()")
    ]
    #[allow(clippy::too_many_arguments)]
    pub fn to_buffer<'py>(
        &'py self,
        py: Python<'py>,
    ) -> PyObject {
        PyBytes::new_bound(py, self.0.buffer()).into()
    }
    // pre PEP 688 buffer protocol
    #[cfg(any(Py_3_11, not(Py_LIMITED_API)))]
    pub unsafe fn __getbuffer__(slf: PyRef<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Object is not writable"));
        }
        let ptr = slf.as_ptr();
        let data = slf.0.buffer();
        (*view).obj = ptr;
        
        (*view).buf = data.as_ptr() as *mut c_void;
        (*view).len = data.len() as isize;
        (*view).readonly = 1;
        (*view).itemsize = 1;
        (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            let msg = CString::new("B").unwrap();
            msg.into_raw()
        } else {
            ptr::null_mut()
        };

        (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
            &mut (*view).itemsize
        } else {
            ptr::null_mut()
        };

        (*view).suboffsets = ptr::null_mut();
        (*view).internal = ptr::null_mut();
        Ok(())
    }
    #[cfg(any(Py_3_11, not(Py_LIMITED_API)))]
    pub unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        // Release memory held by the format string
        drop(CString::from_raw((*view).format));
    }

    /// The total number of items contained in this RTree.
    #[getter]
    pub fn num_items(&self) -> usize {
        self.0.num_items()
    }

    /// The total number of nodes in this RTree, including both leaf and intermediate nodes.
    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.0.num_nodes()
    }

    /// The maximum number of elements in each node.
    #[getter]
    pub fn node_size(&self) -> usize {
        self.0.node_size()
    }

    /// The height of the tree
    #[getter]
    fn num_levels(&self) -> usize {
        self.0.num_levels()
    }

    /// The number of bytes taken up in memory.
    #[getter]
    fn num_bytes(&self) -> usize {
        self.0.num_bytes()
    }

    /// Access the bounding boxes at the given level of the tree.
    ///
    /// The tree is laid out from bottom to top. Level 0 is the _base_ of the tree. Each integer
    /// higher is one level higher of the tree.
    fn boxes_at_level<'py>(&'py self, py: Python<'py>, level: usize) -> PyResult<PyObject> {
        self.0.boxes_at_level(py, level)
    }

    /// Search an RTree given the provided bounding box.
    ///
    /// Results are the indexes of the inserted objects in insertion order.
    ///
    /// Args:
    ///     min_x: min x coordinate of bounding box
    ///     min_y: min y coordinate of bounding box
    ///     max_x: max x coordinate of bounding box
    ///     max_y: max y coordinate of bounding box
    pub fn search<'py>(
        &'py self,
        py: Python<'py>,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Bound<'py, PyArray1<usize>> {
        let result = py.allow_threads(|| match &self.0 {
            RTreeInner::Float32(tree) => {
                let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min_x, min_y, max_x, max_y);
                tree.search(min_x, min_y, max_x, max_y)
            }
            RTreeInner::Float64(tree) => tree.search(min_x, min_y, max_x, max_y),
        });

        PyArray1::from_vec_bound(py, result)
    }
}
