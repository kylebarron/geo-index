use std::marker::PhantomData;

use bytemuck::cast_slice_mut;

use crate::constants::VERSION;
use crate::index::OwnedFlatbush;
use crate::util::compute_num_nodes;

const ARRAY_TYPE_INDEX: u8 = 8;

pub struct FlatbushBuilder {
    ///
    data: Vec<u8>,
    num_items: usize,
    node_size: usize,
    num_nodes: usize,
    level_bounds: Vec<usize>,

    pos: usize,

    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,

    // Used in the future to have
    // FlatbushBuilder<T>
    _type: PhantomData<f64>,
}

impl FlatbushBuilder {
    pub fn new(num_items: usize) -> Self {
        Self::new_with_node_size(num_items, 16)
    }

    pub fn new_with_node_size(num_items: usize, node_size: usize) -> Self {
        assert!((2..=65535).contains(&node_size));
        // TODO: assert num_items fits in u32

        let (num_nodes, level_bounds) = compute_num_nodes(num_items, node_size);

        let f64_bytes_per_element = 8;
        let indices_bytes_per_element = if num_nodes < 16384 { 2 } else { 4 };
        let nodes_byte_size = num_nodes * 4 * f64_bytes_per_element;

        let data_buffer_length = 8 + nodes_byte_size + num_nodes * indices_bytes_per_element;
        let mut data = vec![0; data_buffer_length];

        // Set data header
        data[0] = 0xfb;
        data[1] = (VERSION << 4) + ARRAY_TYPE_INDEX;
        cast_slice_mut(&mut data[2..4])[0] = node_size as u16;
        cast_slice_mut(&mut data[4..8])[0] = node_size as u32;

        Self {
            data,
            num_items,
            num_nodes,
            node_size,
            level_bounds,
            pos: 0,
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
            _type: PhantomData,
        }
    }

    /// Add a given rectangle to the index.
    pub fn add(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> usize {
        let index = self.pos >> 2;
        let (boxes, indices) = data_to_boxes_and_indices(&mut self.data, self.num_nodes);

        indices[index] = index as i32;
        boxes[self.pos] = min_x;
        self.pos += 1;
        boxes[self.pos] = min_y;
        self.pos += 1;
        boxes[self.pos] = max_x;
        self.pos += 1;
        boxes[self.pos] = max_y;
        self.pos += 1;

        if min_x < self.min_x {
            self.min_x = min_x
        };
        if min_y < self.min_y {
            self.min_y = min_y
        };
        if max_x > self.max_x {
            self.max_x = max_x
        };
        if max_y > self.max_y {
            self.max_y = max_y
        };

        index
    }

    pub fn finish(mut self) -> OwnedFlatbush {
        assert_eq!(
            self.pos >> 2,
            self.num_items,
            "Added {} items when expected {}.",
            self.pos >> 2,
            self.num_items
        );
        let (boxes, indices) = data_to_boxes_and_indices(&mut self.data, self.num_nodes);

        if self.num_items <= self.node_size {
            // only one node, skip sorting and just fill the root box
            boxes[self.pos] = self.min_x;
            self.pos += 1;
            boxes[self.pos] = self.min_y;
            self.pos += 1;
            boxes[self.pos] = self.max_x;
            self.pos += 1;
            boxes[self.pos] = self.max_y;
            self.pos += 1;

            return OwnedFlatbush {
                buffer: self.data,
                node_size: self.node_size,
                num_items: self.num_items,
                num_nodes: self.num_nodes,
                level_bounds: self.level_bounds,
            };
        }

        let width = self.max_x - self.min_x; // || 1.0;
        let height = self.max_y - self.min_y; // || 1.0;
        let mut hilbert_values: Vec<u32> = Vec::with_capacity(self.num_items);
        let hilbert_max = ((1 << 16) - 1) as f64;

        {
            // map item centers into Hilbert coordinate space and calculate Hilbert values
            let mut pos = 0;
            for _ in 0..self.num_items {
                let min_x = boxes[pos];
                pos += 1;
                let min_y = boxes[pos];
                pos += 1;
                let max_x = boxes[pos];
                pos += 1;
                let max_y = boxes[pos];
                pos += 1;

                let x = (hilbert_max * ((min_x + max_x) / 2. - self.min_x) / width).floor() as u32;
                let y = (hilbert_max * ((min_y + max_y) / 2. - self.min_y) / height).floor() as u32;
                hilbert_values.push(hilbert(x, y));
            }
        }

        // TODO:
        //
        // sort items by their Hilbert value (for packing later)
        // sort(hilbertValues, boxes, this._indices, 0, this.numItems - 1, this.nodeSize);

        {
            // generate nodes at each tree level, bottom-up
            let mut pos = 0;
            for end in &self.level_bounds {
                while pos < *end {
                    let node_index = pos;

                    // calculate bbox for the new node
                    let mut node_min_x = boxes[pos];
                    pos += 1;
                    let mut node_min_y = boxes[pos];
                    pos += 1;
                    let mut node_max_x = boxes[pos];
                    pos += 1;
                    let mut node_max_y = boxes[pos];
                    pos += 1;
                    for _ in 1..self.node_size {
                        if pos >= *end {
                            break;
                        }

                        node_min_x = node_min_x.min(boxes[pos]);
                        pos += 1;
                        node_min_y = node_min_y.min(boxes[pos]);
                        pos += 1;
                        node_max_x = node_max_x.max(boxes[pos]);
                        pos += 1;
                        node_max_y = node_max_y.max(boxes[pos]);
                        pos += 1;
                    }

                    // add the new node to the tree data
                    indices[self.pos >> 2] = node_index as i32;
                    boxes[self.pos] = node_min_x;
                    self.pos += 1;
                    boxes[self.pos] = node_min_y;
                    self.pos += 1;
                    boxes[self.pos] = node_max_x;
                    self.pos += 1;
                    boxes[self.pos] = node_max_y;
                    self.pos += 1;
                }
            }
        }

        OwnedFlatbush {
            buffer: self.data,
            node_size: self.node_size,
            num_items: self.num_items,
            num_nodes: self.num_nodes,
            level_bounds: self.level_bounds,
        }
    }
}

/// Mutable borrow of boxes and indices
fn data_to_boxes_and_indices(data: &mut [u8], num_nodes: usize) -> (&mut [f64], &mut [i32]) {
    let (_header, rest) = data.split_at_mut(8);
    let f64_bytes_per_element = 8;
    let boxes_byte_length = num_nodes * 4 * f64_bytes_per_element;
    let (boxes_buf, indices_buf) = rest.split_at_mut(boxes_byte_length);
    let boxes = cast_slice_mut(boxes_buf);
    let indices = cast_slice_mut(indices_buf);
    (boxes, indices)
}

// Taken from static_aabb2d_index under the mit/apache license
// https://github.com/jbuckmccready/static_aabb2d_index/blob/9e6add59d77b74d4de0ac32159db47fbcb3acc28/src/static_aabb2d_index.rs#L486C1-L544C2
fn hilbert(x: u32, y: u32) -> u32 {
    // Fast Hilbert curve algorithm by http://threadlocalmutex.com/
    // Ported from C++ https://github.com/rawrunprotected/hilbert_curves (public domain)
    let mut a_1 = x ^ y;
    let mut b_1 = 0xFFFF ^ a_1;
    let mut c_1 = 0xFFFF ^ (x | y);
    let mut d_1 = x & (y ^ 0xFFFF);

    let mut a_2 = a_1 | (b_1 >> 1);
    let mut b_2 = (a_1 >> 1) ^ a_1;
    let mut c_2 = ((c_1 >> 1) ^ (b_1 & (d_1 >> 1))) ^ c_1;
    let mut d_2 = ((a_1 & (c_1 >> 1)) ^ (d_1 >> 1)) ^ d_1;

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 2)) ^ (b_1 & (b_1 >> 2));
    b_2 = (a_1 & (b_1 >> 2)) ^ (b_1 & ((a_1 ^ b_1) >> 2));
    c_2 ^= (a_1 & (c_1 >> 2)) ^ (b_1 & (d_1 >> 2));
    d_2 ^= (b_1 & (c_1 >> 2)) ^ ((a_1 ^ b_1) & (d_1 >> 2));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 4)) ^ (b_1 & (b_1 >> 4));
    b_2 = (a_1 & (b_1 >> 4)) ^ (b_1 & ((a_1 ^ b_1) >> 4));
    c_2 ^= (a_1 & (c_1 >> 4)) ^ (b_1 & (d_1 >> 4));
    d_2 ^= (b_1 & (c_1 >> 4)) ^ ((a_1 ^ b_1) & (d_1 >> 4));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    c_2 ^= (a_1 & (c_1 >> 8)) ^ (b_1 & (d_1 >> 8));
    d_2 ^= (b_1 & (c_1 >> 8)) ^ ((a_1 ^ b_1) & (d_1 >> 8));

    a_1 = c_2 ^ (c_2 >> 1);
    b_1 = d_2 ^ (d_2 >> 1);

    let mut i0 = x ^ y;
    let mut i1 = b_1 | (0xFFFF ^ (i0 | a_1));

    i0 = (i0 | (i0 << 8)) & 0x00FF00FF;
    i0 = (i0 | (i0 << 4)) & 0x0F0F0F0F;
    i0 = (i0 | (i0 << 2)) & 0x33333333;
    i0 = (i0 | (i0 << 1)) & 0x55555555;

    i1 = (i1 | (i1 << 8)) & 0x00FF00FF;
    i1 = (i1 | (i1 << 4)) & 0x0F0F0F0F;
    i1 = (i1 | (i1 << 2)) & 0x33333333;
    i1 = (i1 | (i1 << 1)) & 0x55555555;

    (i1 << 1) | i0
}
