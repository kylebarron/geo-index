//! Utilities for generating RTrees.

use float_next_after::NextAfter;

/// Calculate the total number of nodes in the R-tree to allocate space for
/// and the index of each tree level (used in search later)
pub(crate) fn compute_num_nodes(num_items: u32, node_size: u16) -> (usize, Vec<usize>) {
    // The public API uses u32 and u16 types but internally we use usize
    let num_items = num_items as usize;
    let node_size = node_size as usize;

    let mut n = num_items;
    let mut num_nodes = n;
    let mut level_bounds = vec![n * 4];
    while n != 1 {
        n = (n as f64 / node_size as f64).ceil() as usize;
        num_nodes += n;
        level_bounds.push(num_nodes * 4);
    }
    (num_nodes, level_bounds)
}

/// Cast a bounding box with `f64` precision to `f32` precision. This uses the [`float_next_after`]
/// crate to ensure the resulting box is no smaller than the `f64` box.
#[inline]
pub fn f64_box_to_f32(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> (f32, f32, f32, f32) {
    let mut new_min_x = min_x as f32; //.next_after(f32::NEG_INFINITY);
    let mut new_min_y = min_y as f32; //.next_after(f32::NEG_INFINITY);
    let mut new_max_x = max_x as f32; //.next_after(f32::INFINITY);
    let mut new_max_y = max_y as f32; //.next_after(f32::INFINITY);

    if (new_min_x as f64) > min_x {
        new_min_x = new_min_x.next_after(f32::NEG_INFINITY);
    }
    if (new_min_y as f64) > min_y {
        new_min_y = new_min_y.next_after(f32::NEG_INFINITY);
    }
    if (new_max_x as f64) < max_x {
        new_max_x = new_max_x.next_after(f32::INFINITY);
    }
    if (new_max_y as f64) < max_y {
        new_max_y = new_max_y.next_after(f32::INFINITY);
    }

    debug_assert!((new_min_x as f64) <= min_x);
    debug_assert!((new_min_y as f64) <= min_y);
    debug_assert!((new_max_x as f64) >= max_x);
    debug_assert!((new_max_y as f64) >= max_y);

    (new_min_x, new_min_y, new_max_x, new_max_y)
}

#[cfg(test)]
mod test {
    use crate::rtree::util::f64_box_to_f32;

    #[test]
    fn test_f32_box() {
        let min_x = 1.2f64;
        let min_y = 1.3f64;
        let max_x = 2.4f64;
        let max_y = 2.5f64;
        let _new_box = f64_box_to_f32(min_x, min_y, max_x, max_y);
        dbg!(_new_box);
    }
}
