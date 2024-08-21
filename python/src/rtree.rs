use geo_index::indices::Indices;
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::util::f64_box_to_f32;
use geo_index::rtree::{OwnedRTree, RTreeBuilder, RTreeIndex, TreeMetadata};
use geo_index::{CoordType, IndexableNum};
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::os::raw::c_int;

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

struct PyU8Buffer(PyBuffer<u8>);

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
    fn as_ref(&self) -> &[u8] {
        let len = self.0.item_count();
        let data = self.0.buf_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(data, len) }
    }
}

struct Pyf64RTreeRef {
    buffer: PyU8Buffer,
    metadata: TreeMetadata<f64>,
}

impl Pyf64RTreeRef {
    fn try_new(buffer: PyU8Buffer) -> PyResult<Self> {
        let metadata = TreeMetadata::try_new(buffer.as_ref()).unwrap();
        Ok(Self { buffer, metadata })
    }
}

impl AsRef<[u8]> for Pyf64RTreeRef {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_ref()
    }
}

impl RTreeIndex<f64> for Pyf64RTreeRef {
    fn boxes(&self) -> &[f64] {
        self.metadata.boxes_slice(self.as_ref())
    }

    fn indices(&self) -> Indices {
        self.metadata.indices_slice(self.as_ref())
    }

    fn level_bounds(&self) -> &[usize] {
        self.metadata.level_bounds()
    }

    fn node_size(&self) -> usize {
        self.metadata.node_size()
    }

    fn num_items(&self) -> usize {
        self.metadata.num_items()
    }

    fn num_nodes(&self) -> usize {
        self.metadata.num_nodes()
    }
}

struct Pyf32RTreeRef {
    buffer: PyU8Buffer,
    metadata: TreeMetadata<f32>,
}

impl Pyf32RTreeRef {
    fn try_new(buffer: PyU8Buffer) -> PyResult<Self> {
        let metadata = TreeMetadata::try_new(buffer.as_ref()).unwrap();
        Ok(Self { buffer, metadata })
    }
}

impl AsRef<[u8]> for Pyf32RTreeRef {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_ref()
    }
}

impl RTreeIndex<f32> for Pyf32RTreeRef {
    fn boxes(&self) -> &[f32] {
        self.metadata.boxes_slice(self.as_ref())
    }

    fn indices(&self) -> Indices {
        self.metadata.indices_slice(self.as_ref())
    }

    fn level_bounds(&self) -> &[usize] {
        self.metadata.level_bounds()
    }

    fn node_size(&self) -> usize {
        self.metadata.node_size()
    }

    fn num_items(&self) -> usize {
        self.metadata.num_items()
    }

    fn num_nodes(&self) -> usize {
        self.metadata.num_nodes()
    }
}

pub(crate) enum PyRTreeRef {
    Float32(Pyf32RTreeRef),
    Float64(Pyf64RTreeRef),
}

impl<'py> FromPyObject<'py> for PyRTreeRef {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let buffer = PyU8Buffer::extract_bound(ob)?;
        let ct = CoordType::from_buffer(&buffer.as_ref()).unwrap();
        match ct {
            CoordType::Float32 => Ok(Self::Float32(Pyf32RTreeRef::try_new(buffer)?)),
            CoordType::Float64 => Ok(Self::Float64(Pyf64RTreeRef::try_new(buffer)?)),
            _ => todo!(),
        }
    }
}

impl PyRTreeRef {
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

// This will take the place of RTree
#[pyclass]
struct RTreeRefWrapper(PyRTreeRef);

/// Search an RTree given the provided bounding box.
///
/// Results are the indexes of the inserted objects in insertion order.
///
/// Args:
///     tree: tree or buffer to search
///     min_x: min x coordinate of bounding box
///     min_y: min y coordinate of bounding box
///     max_x: max x coordinate of bounding box
///     max_y: max y coordinate of bounding box
#[pyfunction]
pub(crate) fn search_rtree(
    py: Python,
    tree: PyRTreeRef,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> Bound<'_, PyArray1<usize>> {
    let result = py.allow_threads(|| match tree {
        PyRTreeRef::Float32(tree) => {
            let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min_x, min_y, max_x, max_y);
            tree.search(min_x, min_y, max_x, max_y)
        }
        PyRTreeRef::Float64(tree) => tree.search(min_x, min_y, max_x, max_y),
    });

    PyArray1::from_vec_bound(py, result)
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
            Self::Float32(index) => index.as_ref().len(),
            Self::Float64(index) => index.as_ref().len(),
        }
    }

    fn buffer(&self) -> &[u8] {
        match self {
            Self::Float32(index) => index.as_ref(),
            Self::Float64(index) => index.as_ref(),
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
// impl<'a> FromPyObject<'a> for RTree {
//     fn extract(ob: &'a PyAny) -> PyResult<Self> {
//         let s: Vec<u8> = ob.extract()?;
//         OwnedRTree::from(value)
//     }
// }

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

    // pre PEP 688 buffer protocol
    pub unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let bytes = slf.0.buffer();
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
    pub unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // is there anything to do here?
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
