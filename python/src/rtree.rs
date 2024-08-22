use geo_index::indices::Indices;
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::util::f64_box_to_f32;
use geo_index::rtree::{OwnedRTree, RTreeBuilder, RTreeIndex, TreeMetadata};
use geo_index::{CoordType, IndexableNum};
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::os::raw::c_int;

use crate::common::{PyU8Buffer, RustBuffer};

/// Method for constructing rtree
enum RTreeMethod {
    Hilbert,
    SortTileRecursive,
}

impl<'a> FromPyObject<'a> for RTreeMethod {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "hilbert" => Ok(Self::Hilbert),
            "str" => Ok(Self::SortTileRecursive),
            _ => Err(PyValueError::new_err(
                "Unexpected method. Should be one of 'hilbert' or 'str'.",
            )),
        }
    }
}

/// A low-level wrapper around a [PyU8Buffer] that validates that the input is a valid Flatbush
/// buffer. This wrapper implements [RTreeIndex].
pub(crate) struct PyRTreeBuffer<N: IndexableNum> {
    buffer: PyU8Buffer,
    metadata: TreeMetadata<N>,
}

impl<N: IndexableNum> PyRTreeBuffer<N> {
    fn try_new(buffer: PyU8Buffer) -> PyResult<Self> {
        let metadata = TreeMetadata::try_new(buffer.as_ref()).unwrap();
        Ok(Self { buffer, metadata })
    }

    fn from_owned_rtree(py: Python, tree: OwnedRTree<N>) -> PyResult<Self> {
        let metadata = tree.metadata().clone();
        let tree_buf = RustBuffer::new(tree.into_inner());
        Ok(Self {
            buffer: tree_buf.into_py(py).extract(py)?,
            metadata,
        })
    }
}

impl<N: IndexableNum> AsRef<[u8]> for PyRTreeBuffer<N> {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_ref()
    }
}

impl<N: IndexableNum> RTreeIndex<N> for PyRTreeBuffer<N> {
    fn boxes(&self) -> &[N] {
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

/// An enum wrapper around [PyRTreeBuffer] that allows use of multiple coordinate types from
/// Python.
pub(crate) enum PyRTreeRef {
    Int8(PyRTreeBuffer<i8>),
    Int16(PyRTreeBuffer<i16>),
    Int32(PyRTreeBuffer<i32>),
    UInt8(PyRTreeBuffer<u8>),
    UInt16(PyRTreeBuffer<u16>),
    UInt32(PyRTreeBuffer<u32>),
    Float32(PyRTreeBuffer<f32>),
    Float64(PyRTreeBuffer<f64>),
}

impl<'py> FromPyObject<'py> for PyRTreeRef {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let buffer = PyU8Buffer::extract_bound(ob)?;
        let ct = CoordType::from_buffer(&buffer.as_ref()).unwrap();
        match ct {
            CoordType::Float32 => Ok(Self::Float32(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::Float64 => Ok(Self::Float64(PyRTreeBuffer::try_new(buffer)?)),
            _ => todo!(),
        }
    }
}

impl From<PyRTreeBuffer<f32>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<f32>) -> Self {
        Self::Float32(value)
    }
}

impl From<PyRTreeBuffer<f64>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<f64>) -> Self {
        Self::Float64(value)
    }
}

impl AsRef<[u8]> for PyRTreeRef {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Int8(inner) => inner.as_ref(),
            Self::Int16(inner) => inner.as_ref(),
            Self::Int32(inner) => inner.as_ref(),
            Self::UInt8(inner) => inner.as_ref(),
            Self::UInt16(inner) => inner.as_ref(),
            Self::UInt32(inner) => inner.as_ref(),
            Self::Float32(inner) => inner.as_ref(),
            Self::Float64(inner) => inner.as_ref(),
        }
    }
}

impl PyRTreeRef {
    fn num_items(&self) -> usize {
        match self {
            Self::Int8(index) => index.num_items(),
            Self::Int16(index) => index.num_items(),
            Self::Int32(index) => index.num_items(),
            Self::UInt8(index) => index.num_items(),
            Self::UInt16(index) => index.num_items(),
            Self::UInt32(index) => index.num_items(),
            Self::Float32(index) => index.num_items(),
            Self::Float64(index) => index.num_items(),
        }
    }

    fn num_nodes(&self) -> usize {
        match self {
            Self::Int8(index) => index.num_nodes(),
            Self::Int16(index) => index.num_nodes(),
            Self::Int32(index) => index.num_nodes(),
            Self::UInt8(index) => index.num_nodes(),
            Self::UInt16(index) => index.num_nodes(),
            Self::UInt32(index) => index.num_nodes(),
            Self::Float32(index) => index.num_nodes(),
            Self::Float64(index) => index.num_nodes(),
        }
    }

    fn node_size(&self) -> usize {
        match self {
            Self::Int8(index) => index.node_size(),
            Self::Int16(index) => index.node_size(),
            Self::Int32(index) => index.node_size(),
            Self::UInt8(index) => index.node_size(),
            Self::UInt16(index) => index.node_size(),
            Self::UInt32(index) => index.node_size(),
            Self::Float32(index) => index.node_size(),
            Self::Float64(index) => index.node_size(),
        }
    }

    fn num_levels(&self) -> usize {
        match self {
            Self::Int8(index) => index.num_levels(),
            Self::Int16(index) => index.num_levels(),
            Self::Int32(index) => index.num_levels(),
            Self::UInt8(index) => index.num_levels(),
            Self::UInt16(index) => index.num_levels(),
            Self::UInt32(index) => index.num_levels(),
            Self::Float32(index) => index.num_levels(),
            Self::Float64(index) => index.num_levels(),
        }
    }

    fn num_bytes(&self) -> usize {
        match self {
            Self::Int8(index) => index.as_ref().len(),
            Self::Int16(index) => index.as_ref().len(),
            Self::Int32(index) => index.as_ref().len(),
            Self::UInt8(index) => index.as_ref().len(),
            Self::UInt16(index) => index.as_ref().len(),
            Self::UInt32(index) => index.as_ref().len(),
            Self::Float32(index) => index.as_ref().len(),
            Self::Float64(index) => index.as_ref().len(),
        }
    }

    fn boxes_at_level<'py>(&'py self, py: Python<'py>, level: usize) -> PyResult<PyObject> {
        match self {
            Self::Int8(index) => _boxes_at_level(py, index, level),
            Self::Int16(index) => _boxes_at_level(py, index, level),
            Self::Int32(index) => _boxes_at_level(py, index, level),
            Self::UInt8(index) => _boxes_at_level(py, index, level),
            Self::UInt16(index) => _boxes_at_level(py, index, level),
            Self::UInt32(index) => _boxes_at_level(py, index, level),
            Self::Float32(index) => _boxes_at_level(py, index, level),
            Self::Float64(index) => _boxes_at_level(py, index, level),
        }
    }
}

fn _boxes_at_level<N: IndexableNum + numpy::Element>(
    py: Python,
    index: &PyRTreeBuffer<N>,
    level: usize,
) -> PyResult<PyObject> {
    let boxes = index
        .boxes_at_level(level)
        .map_err(|err| PyIndexError::new_err(err.to_string()))?;
    let array = PyArray1::from_slice_bound(py, boxes);
    Ok(array.reshape([boxes.len() / 4, 4])?.into_py(py))
}

#[pyclass]
pub(crate) struct RTree(PyRTreeRef);

#[pymethods]
impl RTree {
    /// Construct an RTree from an existing RTree buffer
    #[classmethod]
    fn from_buffer(_cls: &Bound<PyType>, py: Python, obj: PyObject) -> PyResult<Self> {
        Ok(Self(obj.extract(py)?))
    }

    #[classmethod]
    #[pyo3(
        signature = (boxes, *, method = RTreeMethod::Hilbert, node_size = None),
        text_signature = "(boxes, *, method = 'hilbert', node_size = None)")
    ]
    fn from_interleaved(
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
            Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
        } else if let Ok(boxes) = boxes.extract::<PyReadonlyArray2<f32>>(py) {
            let boxes = boxes.as_array();
            let tree = py.allow_threads(|| new_interleaved(&boxes, method, node_size));
            Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
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
    fn from_separated(
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
            Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
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
            Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
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
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let bytes = slf.0.as_ref();
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
        min_x: PyObject,
        min_y: PyObject,
        max_x: PyObject,
        max_y: PyObject,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let result: Result<_, PyErr> = match &self.0 {
            PyRTreeRef::Int8(tree) => {
                let min_x = min_x.extract(py)?;
                let min_y = min_y.extract(py)?;
                let max_x = max_x.extract(py)?;
                let max_y = max_y.extract(py)?;
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
            PyRTreeRef::Int16(tree) => {
                let min_x = min_x.extract(py)?;
                let min_y = min_y.extract(py)?;
                let max_x = max_x.extract(py)?;
                let max_y = max_y.extract(py)?;
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
            PyRTreeRef::Int32(tree) => {
                let min_x = min_x.extract(py)?;
                let min_y = min_y.extract(py)?;
                let max_x = max_x.extract(py)?;
                let max_y = max_y.extract(py)?;
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
            PyRTreeRef::UInt8(tree) => {
                let min_x = min_x.extract(py)?;
                let min_y = min_y.extract(py)?;
                let max_x = max_x.extract(py)?;
                let max_y = max_y.extract(py)?;
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
            PyRTreeRef::UInt16(tree) => {
                let min_x = min_x.extract(py)?;
                let min_y = min_y.extract(py)?;
                let max_x = max_x.extract(py)?;
                let max_y = max_y.extract(py)?;
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
            PyRTreeRef::UInt32(tree) => {
                let min_x = min_x.extract(py)?;
                let min_y = min_y.extract(py)?;
                let max_x = max_x.extract(py)?;
                let max_y = max_y.extract(py)?;
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
            PyRTreeRef::Float32(tree) => {
                let min_x = min_x.extract::<f64>(py)?;
                let min_y = min_y.extract::<f64>(py)?;
                let max_x = max_x.extract::<f64>(py)?;
                let max_y = max_y.extract::<f64>(py)?;

                let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min_x, min_y, max_x, max_y);
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
            PyRTreeRef::Float64(tree) => {
                let min_x = min_x.extract(py)?;
                let min_y = min_y.extract(py)?;
                let max_x = max_x.extract(py)?;
                let max_y = max_y.extract(py)?;
                Ok(py.allow_threads(|| tree.search(min_x, min_y, max_x, max_y)))
            }
        };

        Ok(PyArray1::from_vec_bound(py, result?))
    }
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
        RTreeMethod::SortTileRecursive => builder.finish::<STRSort>(),
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
        RTreeMethod::SortTileRecursive => builder.finish::<STRSort>(),
    }
}
