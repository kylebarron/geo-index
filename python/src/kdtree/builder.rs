use std::sync::Arc;

use arrow_array::builder::UInt32Builder;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_cast::cast;
use arrow_schema::DataType;
use geo_index::kdtree::DEFAULT_KDTREE_NODE_SIZE;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use std::os::raw::c_int;

use crate::coord_type::CoordType;

enum KDTreeBuilderInner {
    Float32(geo_index::kdtree::KDTreeBuilder<f32>),
    Float64(geo_index::kdtree::KDTreeBuilder<f64>),
}

#[pyclass]
pub struct KDTreeBuilder(Option<KDTreeBuilderInner>);

#[pymethods]
impl KDTreeBuilder {
    #[new]
    #[pyo3(signature = (num_items, node_size = DEFAULT_KDTREE_NODE_SIZE, coord_type = None))]
    fn new(num_items: u32, node_size: u16, coord_type: Option<CoordType>) -> Self {
        let coord_type = coord_type.unwrap_or(CoordType::Float64);
        match coord_type {
            CoordType::Float32 => Self(Some(KDTreeBuilderInner::Float32(
                geo_index::kdtree::KDTreeBuilder::<f32>::new_with_node_size(num_items, node_size),
            ))),
            CoordType::Float64 => Self(Some(KDTreeBuilderInner::Float64(
                geo_index::kdtree::KDTreeBuilder::<f64>::new_with_node_size(num_items, node_size),
            ))),
        }
    }

    #[pyo3(signature = (x, y = None))]
    pub fn add(&mut self, py: Python, x: PyArray, y: Option<PyArray>) -> PyResult<PyObject> {
        let (x_array, x_field) = x.into_inner();
        if x_array.null_count() > 0 {
            return Err(PyValueError::new_err("Cannot pass array with null values"));
        }
        let mut out_array = UInt32Builder::with_capacity(x_array.len());

        let inner = self.0.as_mut().take().unwrap();

        match (x_field.data_type(), y) {
            (DataType::FixedSizeList(_inner_field, list_size), y) => {
                assert_eq!(
                    *list_size, 2,
                    "Expected list size to be 2 for fixed size list"
                );
                assert!(y.is_none(), "Cannot pass y when x is a FixedSizeList");
                let values_arr = x_array.as_fixed_size_list().values();
                match inner {
                    KDTreeBuilderInner::Float32(tree) => {
                        let values = cast(&values_arr, &DataType::Float32).unwrap();
                        let values = values.as_primitive::<Float32Type>();
                        for i in (0..values.len()).step_by(2) {
                            out_array.append_value(tree.add(values.value(i), values.value(i + 1)));
                        }
                    }
                    KDTreeBuilderInner::Float64(tree) => {
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
                let child_y = struct_arr.column(0);

                match inner {
                    KDTreeBuilderInner::Float32(tree) => {
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
                    KDTreeBuilderInner::Float64(tree) => {
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
                KDTreeBuilderInner::Float32(tree) => {
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
                KDTreeBuilderInner::Float64(tree) => {
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

        PyArray::from_array_ref(Arc::new(out_array.finish())).to_arro3(py)
    }

    fn finish(&mut self) -> PyResult<KDTree> {
        let inner = self
            .0
            .take()
            .ok_or(PyValueError::new_err("Cannot call finish multiple times."))?;
        let out = match inner {
            KDTreeBuilderInner::Float32(tree) => KDTree(KDTreeInner::Float32(tree.finish())),
            KDTreeBuilderInner::Float64(tree) => KDTree(KDTreeInner::Float64(tree.finish())),
        };
        Ok(out)
    }
}

enum KDTreeInner {
    Float32(geo_index::kdtree::KDTree<f32>),
    Float64(geo_index::kdtree::KDTree<f64>),
}

impl KDTreeInner {
    fn buffer(&self) -> &[u8] {
        match self {
            Self::Float32(index) => index.as_ref(),
            Self::Float64(index) => index.as_ref(),
        }
    }
}

#[pyclass]
pub struct KDTree(KDTreeInner);

#[pymethods]
impl KDTree {
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
}
