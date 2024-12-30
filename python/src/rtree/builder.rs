use arrow_array::builder::UInt32Builder;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_cast::cast;
use arrow_schema::DataType;
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::util::f64_box_to_f32;
use geo_index::rtree::{RTree, RTreeBuilder, RTreeIndex, DEFAULT_RTREE_NODE_SIZE};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3_arrow::PyArray;
use std::os::raw::c_int;
use std::sync::Arc;

use crate::coord_type::CoordType;

#[allow(clippy::upper_case_acronyms)]
pub enum RTreeMethod {
    Hilbert,
    STR,
}

impl<'a> FromPyObject<'a> for RTreeMethod {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s = ob.extract::<PyBackedStr>()?;
        match s.to_lowercase().as_str() {
            "hilbert" => Ok(Self::Hilbert),
            "str" => Ok(Self::STR),
            _ => Err(PyValueError::new_err(
                "Unexpected method. Should be one of 'hilbert' or 'str'.",
            )),
        }
    }
}

enum PyRTreeBuilderInner {
    Float32(RTreeBuilder<f32>),
    Float64(RTreeBuilder<f64>),
}

#[pyclass(name = "RTreeBuilder")]
pub struct PyRTreeBuilder(Option<PyRTreeBuilderInner>);

impl PyRTreeBuilder {
    fn add_separated(
        py: Python<'_>,
        inner: &mut PyRTreeBuilderInner,
        mut out_array: UInt32Builder,
        min_x: &dyn arrow_array::Array,
        min_y: &dyn arrow_array::Array,
        max_x: &dyn arrow_array::Array,
        max_y: &dyn arrow_array::Array,
    ) -> PyResult<PyObject> {
        assert_eq!(min_x.data_type(), min_y.data_type());
        assert_eq!(min_x.data_type(), max_x.data_type());
        assert_eq!(min_x.data_type(), max_y.data_type());

        match inner {
            PyRTreeBuilderInner::Float32(tree) => match min_x.data_type() {
                // When Float64 data is passed but the tree is Float32, we special case converting
                // the f64 box to f32
                DataType::Float64 => {
                    assert_eq!(min_x.null_count(), 0);
                    assert_eq!(min_y.null_count(), 0);
                    assert_eq!(max_x.null_count(), 0);
                    assert_eq!(max_y.null_count(), 0);

                    let values_min_x = min_x.as_primitive::<Float64Type>().values();
                    let values_min_y = min_y.as_primitive::<Float64Type>().values();
                    let values_max_x = max_x.as_primitive::<Float64Type>().values();
                    let values_max_y = max_y.as_primitive::<Float64Type>().values();

                    for (((min_x, min_y), max_x), max_y) in values_min_x
                        .iter()
                        .zip(values_min_y)
                        .zip(values_max_x)
                        .zip(values_max_y)
                    {
                        let f32_box = f64_box_to_f32(*min_x, *min_y, *max_x, *max_y);
                        out_array
                            .append_value(tree.add(f32_box.0, f32_box.1, f32_box.2, f32_box.3));
                    }
                }
                _ => {
                    assert_eq!(min_x.null_count(), 0);
                    assert_eq!(min_y.null_count(), 0);
                    assert_eq!(max_x.null_count(), 0);
                    assert_eq!(max_y.null_count(), 0);

                    let values_min_x = cast(min_x, &DataType::Float32).unwrap();
                    let values_min_y = cast(min_y, &DataType::Float32).unwrap();
                    let values_max_x = cast(max_x, &DataType::Float32).unwrap();
                    let values_max_y = cast(max_y, &DataType::Float32).unwrap();

                    let values_min_x = values_min_x.as_primitive::<Float32Type>().values();
                    let values_min_y = values_min_y.as_primitive::<Float32Type>().values();
                    let values_max_x = values_max_x.as_primitive::<Float32Type>().values();
                    let values_max_y = values_max_y.as_primitive::<Float32Type>().values();

                    for (((min_x, min_y), max_x), max_y) in values_min_x
                        .iter()
                        .zip(values_min_y)
                        .zip(values_max_x)
                        .zip(values_max_y)
                    {
                        out_array.append_value(tree.add(*min_x, *min_y, *max_x, *max_y));
                    }
                }
            },
            PyRTreeBuilderInner::Float64(tree) => {
                let values_min_x = cast(min_x, &DataType::Float64).unwrap();
                let values_min_y = cast(min_y, &DataType::Float64).unwrap();
                let values_max_x = cast(max_x, &DataType::Float64).unwrap();
                let values_max_y = cast(max_y, &DataType::Float64).unwrap();
                assert_eq!(values_min_x.null_count(), 0);
                assert_eq!(values_min_y.null_count(), 0);
                assert_eq!(values_max_x.null_count(), 0);
                assert_eq!(values_max_y.null_count(), 0);

                let values_min_x = values_min_x.as_primitive::<Float64Type>().values();
                let values_min_y = values_min_y.as_primitive::<Float64Type>().values();
                let values_max_x = values_max_x.as_primitive::<Float64Type>().values();
                let values_max_y = values_max_y.as_primitive::<Float64Type>().values();

                for (((min_x, min_y), max_x), max_y) in values_min_x
                    .iter()
                    .zip(values_min_y)
                    .zip(values_max_x)
                    .zip(values_max_y)
                {
                    out_array.append_value(tree.add(*min_x, *min_y, *max_x, *max_y));
                }
            }
        };

        PyArray::from_array_ref(Arc::new(out_array.finish())).to_arro3(py)
    }
}

#[pymethods]
impl PyRTreeBuilder {
    #[new]
    #[pyo3(signature = (num_items, node_size = DEFAULT_RTREE_NODE_SIZE, coord_type = None))]
    fn new(num_items: u32, node_size: u16, coord_type: Option<CoordType>) -> Self {
        let coord_type = coord_type.unwrap_or(CoordType::Float64);
        match coord_type {
            CoordType::Float32 => Self(Some(PyRTreeBuilderInner::Float32(
                RTreeBuilder::<f32>::new_with_node_size(num_items, node_size),
            ))),
            CoordType::Float64 => Self(Some(PyRTreeBuilderInner::Float64(
                RTreeBuilder::<f64>::new_with_node_size(num_items, node_size),
            ))),
        }
    }

    #[pyo3(signature = (min_x, min_y = None, max_x = None, max_y = None))]
    fn add(
        &mut self,
        py: Python,
        min_x: PyArray,
        min_y: Option<PyArray>,
        max_x: Option<PyArray>,
        max_y: Option<PyArray>,
    ) -> PyResult<PyObject> {
        let min_x = min_x.as_ref();
        if min_x.null_count() > 0 {
            return Err(PyValueError::new_err("Cannot pass array with null values"));
        }
        let mut out_array = UInt32Builder::with_capacity(min_x.len());

        let inner = self.0.as_mut().take().unwrap();
        match (min_x.data_type(), min_y, max_x, max_y) {
            (DataType::FixedSizeList(inner_field, list_size), min_y, max_x, max_y) => {
                assert_eq!(
                    *list_size, 4,
                    "Expected list size to be 4 for fixed size list"
                );
                assert!(
                    min_y.is_none(),
                    "Cannot pass min_y when min_x is a FixedSizeList"
                );
                assert!(
                    max_x.is_none(),
                    "Cannot pass max_x when min_x is a FixedSizeList"
                );
                assert!(
                    max_y.is_none(),
                    "Cannot pass max_y when min_x is a FixedSizeList"
                );
                let values_arr = min_x.as_fixed_size_list().values();
                match inner {
                    PyRTreeBuilderInner::Float32(tree) => match inner_field.data_type() {
                        DataType::Float64 => {
                            let values = values_arr.as_primitive::<Float64Type>();
                            for i in (0..values.len()).step_by(4) {
                                let f32_box = f64_box_to_f32(
                                    values.value(i),
                                    values.value(i + 1),
                                    values.value(i + 2),
                                    values.value(i + 3),
                                );
                                out_array.append_value(
                                    tree.add(f32_box.0, f32_box.1, f32_box.2, f32_box.3),
                                );
                            }
                        }
                        _ => {
                            let values = cast(&values_arr, &DataType::Float32).unwrap();
                            let values = values.as_primitive::<Float32Type>();
                            for i in (0..values.len()).step_by(4) {
                                out_array.append_value(tree.add(
                                    values.value(i),
                                    values.value(i + 1),
                                    values.value(i + 2),
                                    values.value(i + 3),
                                ));
                            }
                        }
                    },
                    PyRTreeBuilderInner::Float64(tree) => {
                        let values = cast(&values_arr, &DataType::Float64).unwrap();
                        let values = values.as_primitive::<Float64Type>();
                        for i in (0..values.len()).step_by(4) {
                            out_array.append_value(tree.add(
                                values.value(i),
                                values.value(i + 1),
                                values.value(i + 2),
                                values.value(i + 3),
                            ));
                        }
                    }
                }
            }
            (DataType::Struct(inner_fields), min_y, max_x, max_y) => {
                assert_eq!(
                    inner_fields.len(),
                    4,
                    "Expected struct to have four inner fields"
                );
                assert!(min_y.is_none(), "Cannot pass min_y when min_x is a struct");
                assert!(max_x.is_none(), "Cannot pass max_x when min_x is a struct");
                assert!(max_y.is_none(), "Cannot pass max_y when min_x is a struct");

                let struct_arr = min_x.as_struct();
                let child_min_x = struct_arr.column(0);
                let child_min_y = struct_arr.column(1);
                let child_max_x = struct_arr.column(2);
                let child_max_y = struct_arr.column(3);
                return PyRTreeBuilder::add_separated(
                    py,
                    inner,
                    out_array,
                    &child_min_x,
                    &child_min_y,
                    &child_max_x,
                    &child_max_y,
                );
            }
            (_, Some(min_y), Some(max_x), Some(max_y)) => {
                return PyRTreeBuilder::add_separated(
                    py,
                    inner,
                    out_array,
                    min_x.as_ref(),
                    min_y.as_ref(),
                    max_x.as_ref(),
                    max_y.as_ref(),
                );
            }
            _ => return Err(PyValueError::new_err("Unsupported argument types")),
        };

        PyArray::from_array_ref(Arc::new(out_array.finish())).to_arro3(py)
    }

    #[pyo3(signature = (method = None))]
    fn finish(&mut self, method: Option<RTreeMethod>) -> PyResult<PyRTree> {
        let method = method.unwrap_or(RTreeMethod::Hilbert);
        let inner = self
            .0
            .take()
            .ok_or(PyValueError::new_err("Cannot call finish multiple times."))?;
        let out = match (inner, method) {
            (PyRTreeBuilderInner::Float32(tree), RTreeMethod::Hilbert) => {
                PyRTree(PyRTreeInner::Float32(tree.finish::<HilbertSort>()))
            }
            (PyRTreeBuilderInner::Float32(tree), RTreeMethod::STR) => {
                PyRTree(PyRTreeInner::Float32(tree.finish::<STRSort>()))
            }
            (PyRTreeBuilderInner::Float64(tree), RTreeMethod::Hilbert) => {
                PyRTree(PyRTreeInner::Float64(tree.finish::<HilbertSort>()))
            }
            (PyRTreeBuilderInner::Float64(tree), RTreeMethod::STR) => {
                PyRTree(PyRTreeInner::Float64(tree.finish::<STRSort>()))
            }
        };
        Ok(out)
    }
}

enum PyRTreeInner {
    Float32(RTree<f32>),
    Float64(RTree<f64>),
}

#[pyclass(name = "RTree")]
pub struct PyRTree(PyRTreeInner);

impl PyRTreeInner {
    fn num_items(&self) -> u32 {
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

    fn node_size(&self) -> u16 {
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
                let array = PyArray1::from_slice(py, boxes);
                Ok(array
                    .reshape([boxes.len() / 4, 4])?
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            }
            Self::Float64(index) => {
                let boxes = index
                    .boxes_at_level(level)
                    .map_err(|err| PyIndexError::new_err(err.to_string()))?;
                let array = PyArray1::from_slice(py, boxes);
                Ok(array
                    .reshape([boxes.len() / 4, 4])?
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            }
        }
    }
}

#[pymethods]
impl PyRTree {
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
    pub fn num_items(&self) -> u32 {
        self.0.num_items()
    }

    /// The total number of nodes in this RTree, including both leaf and intermediate nodes.
    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.0.num_nodes()
    }

    /// The maximum number of elements in each node.
    #[getter]
    pub fn node_size(&self) -> u16 {
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
}
