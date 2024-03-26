use geo_index::kdtree::{KDTreeBuilder, KDTreeIndex, OwnedKDTree};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass]
pub struct KDTree(OwnedKDTree<f64>);

#[pymethods]
impl KDTree {
    #[classmethod]
    #[pyo3(
        signature = (coords, *, node_size = None),
        text_signature = "(coords, *, node_size = None)")
    ]
    pub fn from_interleaved(
        _cls: &PyType,
        coords: PyReadonlyArray2<f64>,
        node_size: Option<usize>,
    ) -> Self {
        let shape = coords.shape();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[1], 4);

        let num_items = shape[0];

        let coords = coords.as_array();

        let mut builder = if let Some(node_size) = node_size {
            KDTreeBuilder::new_with_node_size(num_items, node_size)
        } else {
            KDTreeBuilder::new(num_items)
        };

        for i in 0..num_items {
            builder.add(*coords.get((i, 0)).unwrap(), *coords.get((i, 1)).unwrap());
        }

        Self(builder.finish())
    }

    #[classmethod]
    #[pyo3(
        signature = (x, y, *, node_size = None),
        text_signature = "(x, y, *, node_size = None)")
    ]
    pub fn from_separated(
        _cls: &PyType,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
        node_size: Option<usize>,
    ) -> Self {
        assert_eq!(x.len(), y.len());

        let num_items = x.len();

        let x = x.as_array();
        let y = y.as_array();

        let mut builder = if let Some(node_size) = node_size {
            KDTreeBuilder::new_with_node_size(num_items, node_size)
        } else {
            KDTreeBuilder::new(num_items)
        };

        for i in 0..num_items {
            builder.add(*x.get(i).unwrap(), *y.get(i).unwrap());
        }

        Self(builder.finish())
    }

    /// Search the index for items within a given bounding box.
    ///
    /// Args:
    ///     min_x
    ///     min_y
    ///     max_x
    ///     max_y
    ///
    /// Returns indices of found items
    pub fn range<'py>(
        &'py self,
        py: Python<'py>,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> &'py PyArray1<usize> {
        let result = py.allow_threads(move || self.0.as_ref().range(min_x, min_y, max_x, max_y));
        PyArray1::from_vec(py, result)
    }

    /// Search the index for items within a given radius.
    ///
    /// - qx: x value of query point
    /// - qy: y value of query point
    /// - r: radius
    ///
    /// Returns indices of found items
    pub fn within<'py>(
        &'py self,
        py: Python<'py>,
        qx: f64,
        qy: f64,
        r: f64,
    ) -> &'py PyArray1<usize> {
        let result = py.allow_threads(move || self.0.as_ref().within(qx, qy, r));
        PyArray1::from_vec(py, result)
    }
}
