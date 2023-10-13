use bytemuck::cast_slice;

use crate::index::{Flatbush, OwnedFlatbush};

pub trait FlatbushIndex {
    fn boxes(&self) -> &[f64];
    fn indices(&self) -> &[u32];
    fn num_items(&self) -> usize;
    fn node_size(&self) -> usize;
    fn level_bounds(&self) -> &[usize];

    fn search(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        let boxes = self.boxes();
        let indices = self.indices();

        let mut outer_node_index = Some(boxes.len() - 4);

        let mut queue = vec![];
        let mut results = vec![];

        while let Some(node_index) = outer_node_index {
            // find the end index of the node
            let end = (node_index + self.node_size() * 4)
                .min(upper_bound(node_index, self.level_bounds()));

            // search through child nodes
            for pos in (node_index..end).step_by(4) {
                // check if node bbox intersects with query bbox
                if max_x < boxes[pos] {
                    continue; // maxX < nodeMinX
                }
                if max_y < boxes[pos + 1] {
                    continue; // maxY < nodeMinY
                }
                if min_x > boxes[pos + 2] {
                    continue; // minX > nodeMaxX
                }
                if min_y > boxes[pos + 3] {
                    continue; // minY > nodeMaxY
                }

                let index = indices[pos >> 2];

                if node_index >= self.num_items() * 4 {
                    queue.push(index as usize); // node; add it to the search queue
                } else {
                    results.push(index as usize); // leaf item
                }
            }

            outer_node_index = queue.pop();
        }

        results
    }
}

impl FlatbushIndex for OwnedFlatbush {
    fn boxes(&self) -> &[f64] {
        let data = &self.buffer;

        let f64_bytes_per_element = 8;
        let nodes_byte_length = self.num_nodes * 4 * f64_bytes_per_element;

        cast_slice(&data[8..nodes_byte_length])
    }

    fn indices(&self) -> &[u32] {
        let data = &self.buffer;

        let f64_bytes_per_element = 8;
        let indices_bytes_per_element = 4;
        let nodes_byte_length = self.num_nodes * 4 * f64_bytes_per_element;
        let indices_byte_length = self.num_nodes * indices_bytes_per_element;

        cast_slice(&data[8 + nodes_byte_length..8 + nodes_byte_length + indices_byte_length])
    }

    fn level_bounds(&self) -> &[usize] {
        &self.level_bounds
    }

    fn node_size(&self) -> usize {
        self.node_size
    }

    fn num_items(&self) -> usize {
        self.num_items
    }
}

impl FlatbushIndex for Flatbush<'_> {
    fn boxes(&self) -> &[f64] {
        self.boxes
    }

    fn indices(&self) -> &[u32] {
        self.indices
    }

    fn level_bounds(&self) -> &[usize] {
        &self.level_bounds
    }

    fn node_size(&self) -> usize {
        self.node_size
    }

    fn num_items(&self) -> usize {
        self.num_items
    }
}

/**
 * Binary search for the first value in the array bigger than the given.
 * @param {number} value
 * @param {number[]} arr
 */
fn upper_bound(value: usize, arr: &[usize]) -> usize {
    let mut i = 0;
    let mut j = arr.len() - 1;

    while i < j {
        let m = (i + j) >> 1;
        if arr[m] > value {
            j = m;
        } else {
            i = m + 1;
        }
    }

    arr[i]
}
