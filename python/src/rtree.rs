use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use geo_index::indices::Indices;
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::util::f64_box_to_f32;
use geo_index::rtree::{OwnedRTree, RTreeBuilder, RTreeIndex, TreeMetadata};
use geo_index::{CoordType, IndexableNum};
use numpy::ndarray::ArrayView2;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_arrow::buffer::PyArrowBuffer;
use pyo3_arrow::PyArray;
use std::os::raw::c_int;

/// Method for constructing rtree
enum RTreeMethod {
    Hilbert,
    SortTileRecursive,
}

impl<'a> FromPyObject<'a> for RTreeMethod {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
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

/// A low-level wrapper around a [PyArrowBuffer] that validates that the input is a valid Flatbush
/// buffer. This wrapper implements [RTreeIndex].
pub(crate) struct PyRTreeBuffer<N: IndexableNum> {
    buffer: PyArrowBuffer,
    metadata: TreeMetadata<N>,
}

impl<N: IndexableNum> PyRTreeBuffer<N> {
    fn try_new(buffer: PyArrowBuffer) -> PyResult<Self> {
        let metadata = TreeMetadata::try_new(buffer.as_ref()).unwrap();
        Ok(Self { buffer, metadata })
    }

    fn from_owned_rtree(tree: OwnedRTree<N>) -> PyResult<Self> {
        let metadata = tree.metadata().clone();
        let buffer = PyArrowBuffer::new(Buffer::from_vec(tree.into_inner()));
        Ok(Self { buffer, metadata })
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
        let buffer = ob.extract::<PyArrowBuffer>()?;
        let ct = CoordType::from_buffer(&buffer).unwrap();
        match ct {
            CoordType::Int8 => Ok(Self::Int8(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::Int16 => Ok(Self::Int16(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::Int32 => Ok(Self::Int32(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::UInt8 => Ok(Self::UInt8(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::UInt16 => Ok(Self::UInt16(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::UInt32 => Ok(Self::UInt32(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::Float32 => Ok(Self::Float32(PyRTreeBuffer::try_new(buffer)?)),
            CoordType::Float64 => Ok(Self::Float64(PyRTreeBuffer::try_new(buffer)?)),
        }
    }
}

impl From<PyRTreeBuffer<i8>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<i8>) -> Self {
        Self::Int8(value)
    }
}

impl From<PyRTreeBuffer<i16>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<i16>) -> Self {
        Self::Int16(value)
    }
}

impl From<PyRTreeBuffer<i32>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<i32>) -> Self {
        Self::Int32(value)
    }
}

impl From<PyRTreeBuffer<u8>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<u8>) -> Self {
        Self::UInt8(value)
    }
}

impl From<PyRTreeBuffer<u16>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<u16>) -> Self {
        Self::UInt16(value)
    }
}

impl From<PyRTreeBuffer<u32>> for PyRTreeRef {
    fn from(value: PyRTreeBuffer<u32>) -> Self {
        Self::UInt32(value)
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
    ///
    /// You can pass any buffer protocol object into this constructor.
    #[classmethod]
    fn from_buffer(_cls: &Bound<PyType>, obj: PyRTreeRef) -> Self {
        Self(obj)
    }

    #[classmethod]
    #[pyo3(
        signature = (boxes, *, method = RTreeMethod::Hilbert, node_size = None),
        text_signature = "(boxes, *, method = 'hilbert', node_size = None)")
    ]
    fn from_interleaved(
        _cls: &Bound<PyType>,
        py: Python,
        boxes: PyArray,
        method: RTreeMethod,
        node_size: Option<usize>,
    ) -> PyResult<Self> {
        let data_type = boxes.array().data_type();

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
    fn from_separated<'py>(
        _cls: &Bound<PyType>,
        py: Python<'py>,
        min_x: PyArray,
        min_y: PyArray,
        max_x: PyArray,
        max_y: PyArray,
        method: RTreeMethod,
        node_size: Option<usize>,
    ) -> PyResult<Self> {
        match (min_x, min_y, max_x, max_y) {
            (
                PyArray::Int8(min_x),
                PyArray::Int8(min_y),
                PyArray::Int8(max_x),
                PyArray::Int8(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }
            (
                PyArray::Int16(min_x),
                PyArray::Int16(min_y),
                PyArray::Int16(max_x),
                PyArray::Int16(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }
            (
                PyArray::Int32(min_x),
                PyArray::Int32(min_y),
                PyArray::Int32(max_x),
                PyArray::Int32(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }
            (
                PyArray::UInt8(min_x),
                PyArray::UInt8(min_y),
                PyArray::UInt8(max_x),
                PyArray::UInt8(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }
            (
                PyArray::UInt16(min_x),
                PyArray::UInt16(min_y),
                PyArray::UInt16(max_x),
                PyArray::UInt16(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }
            (
                PyArray::UInt32(min_x),
                PyArray::UInt32(min_y),
                PyArray::UInt32(max_x),
                PyArray::UInt32(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }

            (
                PyArray::Float32(min_x),
                PyArray::Float32(min_y),
                PyArray::Float32(max_x),
                PyArray::Float32(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }
            (
                PyArray::Float64(min_x),
                PyArray::Float64(min_y),
                PyArray::Float64(max_x),
                PyArray::Float64(max_y),
            ) => {
                let tree = new_separated_slice(
                    min_x.as_slice(),
                    min_y.as_slice(),
                    max_x.as_slice(),
                    max_y.as_slice(),
                    method,
                    node_size,
                );

                Ok(Self(PyRTreeBuffer::from_owned_rtree(py, tree)?.into()))
            }
            _ => Err(PyTypeError::new_err(
                "Expected all input arrays to have the same data type",
            )),
        }
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

fn new_separated_slice<N: IndexableNum + numpy::Element>(
    min_x: &[N],
    min_y: &[N],
    max_x: &[N],
    max_y: &[N],
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
