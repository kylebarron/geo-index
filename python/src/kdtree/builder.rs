use std::sync::Arc;

use arrow_array::builder::UInt32Builder;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_cast::cast;
use arrow_schema::DataType;
use geo_index::kdtree::{KDTree, KDTreeBuilder, KDTreeIndex, DEFAULT_KDTREE_NODE_SIZE};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use std::os::raw::c_int;

use crate::coord_type::CoordType;

enum PyKDTreeBuilderInner {
    Float32(KDTreeBuilder<f32>),
    Float64(KDTreeBuilder<f64>),
}

impl PyKDTreeBuilderInner {
    fn node_size(&self) -> u16 {
        match self {
            Self::Float32(builder) => builder.metadata().node_size(),
            Self::Float64(builder) => builder.metadata().node_size(),
        }
    }

    fn num_items(&self) -> u32 {
        match self {
            Self::Float32(builder) => builder.metadata().num_items(),
            Self::Float64(builder) => builder.metadata().num_items(),
        }
    }
}

#[pyclass(name = "KDTreeBuilder")]
pub struct PyKDTreeBuilder(Option<PyKDTreeBuilderInner>);

#[pymethods]
impl PyKDTreeBuilder {
    #[new]
    #[pyo3(signature = (num_items, node_size = DEFAULT_KDTREE_NODE_SIZE, coord_type = None))]
    fn new(num_items: u32, node_size: u16, coord_type: Option<CoordType>) -> Self {
        let coord_type = coord_type.unwrap_or(CoordType::Float64);
        match coord_type {
            CoordType::Float32 => Self(Some(PyKDTreeBuilderInner::Float32(
                KDTreeBuilder::<f32>::new_with_node_size(num_items, node_size),
            ))),
            CoordType::Float64 => Self(Some(PyKDTreeBuilderInner::Float64(
                KDTreeBuilder::<f64>::new_with_node_size(num_items, node_size),
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if let Some(inner) = self.0.as_ref() {
            format!(
                "KDTreeBuilder(num_items={}, node_size={})",
                inner.num_items(),
                inner.node_size()
            )
        } else {
            "KDTreeBuilder(finished)".to_string()
        }
    }

    #[pyo3(signature = (x, y = None))]
    fn add(&mut self, py: Python, x: PyArray, y: Option<PyArray>) -> PyResult<PyObject> {
        let (x_array, x_field) = x.into_inner();
        if x_array.null_count() > 0 {
            return Err(PyValueError::new_err("Cannot pass array with null values"));
        }
        let mut out_array = UInt32Builder::with_capacity(x_array.len());

        let inner = self.0.as_mut().unwrap();

        match (x_field.data_type(), y) {
            (DataType::FixedSizeList(_inner_field, list_size), y) => {
                assert_eq!(
                    *list_size, 2,
                    "Expected list size to be 2 for fixed size list"
                );
                assert!(y.is_none(), "Cannot pass y when x is a FixedSizeList");
                let values_arr = x_array.as_fixed_size_list().values();
                match inner {
                    PyKDTreeBuilderInner::Float32(tree) => {
                        let values = cast(&values_arr, &DataType::Float32).unwrap();
                        let values = values.as_primitive::<Float32Type>();
                        for i in (0..values.len()).step_by(2) {
                            out_array.append_value(tree.add(values.value(i), values.value(i + 1)));
                        }
                    }
                    PyKDTreeBuilderInner::Float64(tree) => {
                        let values = cast(&values_arr, &DataType::Float64).unwrap();
                        let values = values.as_primitive::<Float64Type>();
                        for i in (0..values.len()).step_by(2) {
                            out_array.append_value(tree.add(values.value(i), values.value(i + 1)));
                        }
                    }
                }
            }
            (DataType::Struct(inner_fields), y) => {
                assert_eq!(
                    inner_fields.len(),
                    2,
                    "Expected struct to have two inner fields"
                );
                assert!(y.is_none(), "Cannot pass y when x is a Struct type");
                let struct_arr = x_array.as_struct();
                let child_x = struct_arr.column(0);
                let child_y = struct_arr.column(1);

                match inner {
                    PyKDTreeBuilderInner::Float32(tree) => {
                        let values_x = cast(&child_x, &DataType::Float32).unwrap();
                        let values_y = cast(&child_y, &DataType::Float32).unwrap();
                        assert_eq!(values_x.null_count(), 0);
                        assert_eq!(values_y.null_count(), 0);

                        let values_x = values_x.as_primitive::<Float32Type>().values();
                        let values_y = values_y.as_primitive::<Float32Type>().values();

                        for (x, y) in values_x.iter().zip(values_y) {
                            out_array.append_value(tree.add(*x, *y));
                        }
                    }
                    PyKDTreeBuilderInner::Float64(tree) => {
                        let values_x = cast(&child_x, &DataType::Float64).unwrap();
                        let values_y = cast(&child_y, &DataType::Float64).unwrap();
                        assert_eq!(values_x.null_count(), 0);
                        assert_eq!(values_y.null_count(), 0);

                        let values_x = values_x.as_primitive::<Float64Type>().values();
                        let values_y = values_y.as_primitive::<Float64Type>().values();

                        for (x, y) in values_x.iter().zip(values_y) {
                            out_array.append_value(tree.add(*x, *y));
                        }
                    }
                }
            }
            (_, Some(y)) => match inner {
                PyKDTreeBuilderInner::Float32(tree) => {
                    let values_x = cast(&x_array, &DataType::Float32).unwrap();
                    let values_y = cast(y.as_ref(), &DataType::Float32).unwrap();
                    assert_eq!(values_x.null_count(), 0);
                    assert_eq!(values_y.null_count(), 0);

                    let values_x = values_x.as_primitive::<Float32Type>().values();
                    let values_y = values_y.as_primitive::<Float32Type>().values();

                    for (x, y) in values_x.iter().zip(values_y) {
                        out_array.append_value(tree.add(*x, *y));
                    }
                }
                PyKDTreeBuilderInner::Float64(tree) => {
                    let values_x = cast(&x_array, &DataType::Float64).unwrap();
                    let values_y = cast(y.as_ref(), &DataType::Float64).unwrap();
                    assert_eq!(values_x.null_count(), 0);
                    assert_eq!(values_y.null_count(), 0);

                    let values_x = values_x.as_primitive::<Float64Type>().values();
                    let values_y = values_y.as_primitive::<Float64Type>().values();

                    for (x, y) in values_x.iter().zip(values_y) {
                        out_array.append_value(tree.add(*x, *y));
                    }
                }
            },
            _ => return Err(PyValueError::new_err("Unsupported argument types")),
        };

        Ok(PyArray::from_array_ref(Arc::new(out_array.finish()))
            .to_arro3(py)?
            .unbind())
    }

    fn finish(&mut self) -> PyResult<PyKDTree> {
        let inner = self
            .0
            .take()
            .ok_or(PyValueError::new_err("Cannot call finish multiple times."))?;
        let out = match inner {
            PyKDTreeBuilderInner::Float32(tree) => PyKDTree(PyKDTreeInner::Float32(tree.finish())),
            PyKDTreeBuilderInner::Float64(tree) => PyKDTree(PyKDTreeInner::Float64(tree.finish())),
        };
        Ok(out)
    }
}

enum PyKDTreeInner {
    Float32(KDTree<f32>),
    Float64(KDTree<f64>),
}

impl PyKDTreeInner {
    fn node_size(&self) -> u16 {
        match self {
            Self::Float32(index) => index.node_size(),
            Self::Float64(index) => index.node_size(),
        }
    }

    fn num_items(&self) -> u32 {
        match self {
            Self::Float32(index) => index.num_items(),
            Self::Float64(index) => index.num_items(),
        }
    }

    fn buffer(&self) -> &[u8] {
        match self {
            Self::Float32(index) => index.as_ref(),
            Self::Float64(index) => index.as_ref(),
        }
    }
}

#[pyclass(name = "KDTree", frozen)]
pub struct PyKDTree(PyKDTreeInner);

#[pymethods]
impl PyKDTree {
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

    fn __repr__(&self) -> String {
        format!(
            "KDTree(num_items={}, node_size={})",
            self.0.num_items(),
            self.0.node_size()
        )
    }
}
