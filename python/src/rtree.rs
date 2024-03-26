use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::{OwnedRTree, RTreeBuilder, RTreeIndex};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

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

#[pyclass]
pub struct RTree(OwnedRTree<f64>);

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
        _cls: &PyType,
        boxes: PyReadonlyArray2<f64>,
        method: RTreeMethod,
        node_size: Option<usize>,
    ) -> Self {
        let shape = boxes.shape();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[1], 4);

        let num_items = shape[0];

        let boxes = boxes.as_array();

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
            RTreeMethod::Hilbert => Self(builder.finish::<HilbertSort>()),
            RTreeMethod::STR => Self(builder.finish::<STRSort>()),
        }
    }

    #[classmethod]
    #[pyo3(
        signature = (min_x, min_y, max_x, max_y, *, method = RTreeMethod::Hilbert, node_size = None),
        text_signature = "(min_x, min_y, max_x, max_y, *, method = 'hilbert', node_size = None)")
    ]
    pub fn from_separated(
        _cls: &PyType,
        min_x: PyReadonlyArray1<f64>,
        min_y: PyReadonlyArray1<f64>,
        max_x: PyReadonlyArray1<f64>,
        max_y: PyReadonlyArray1<f64>,
        method: RTreeMethod,
        node_size: Option<usize>,
    ) -> Self {
        assert_eq!(min_x.len(), min_y.len());
        assert_eq!(min_x.len(), max_x.len());
        assert_eq!(min_x.len(), max_y.len());

        let num_items = min_x.len();

        let min_x = min_x.as_array();
        let min_y = min_y.as_array();
        let max_x = max_x.as_array();
        let max_y = max_y.as_array();

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
            RTreeMethod::Hilbert => Self(builder.finish::<HilbertSort>()),
            RTreeMethod::STR => Self(builder.finish::<STRSort>()),
        }
    }

    pub fn search<'py>(
        &'py self,
        py: Python<'py>,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> &'py PyArray1<usize> {
        let result = py.allow_threads(move || self.0.search(min_x, min_y, max_x, max_y));
        PyArray1::from_vec(py, result)
    }
}
