use crate::indices::MutableIndices;

pub struct SortParams {
    pub(crate) num_items: usize,
    pub(crate) node_size: usize,
    // num_nodes: usize,
    // level_bounds: Vec<usize>,
    // nodes_byte_size: usize,
    // indices_byte_size: usize,
    // pub(crate) boxes: &'a mut [f64],
    // pub(crate) indices: &'a mut MutableIndices<'a>,
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) max_x: f64,
    pub(crate) max_y: f64,
}

pub trait Sort {
    fn sort(sort_params: &mut SortParams, boxes: &mut [f64], indices: &mut MutableIndices);
}
