use std::borrow::Cow;

use arrayvec::ArrayVec;

use crate::indices::Indices;
use crate::kdbush::KdbushRef;

pub trait KdbushIndex {
    fn num_items(&self) -> usize;
    fn node_size(&self) -> usize;
    fn coords(&self) -> &[f64];
    fn ids(&self) -> Cow<'_, Indices>;

    /// Search the index for items within a given bounding box.
    ///
    /// - min_x: bbox
    /// - min_y: bbox
    /// - max_x: bbox
    /// - max_y: bbox
    ///
    /// Returns indices of found items
    fn range(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        let ids = self.ids();
        let coords = self.coords();
        let node_size = self.node_size();

        // Use arrayvec to avoid heap allocations
        let mut stack = ArrayVec::<_, 3>::new();
        stack.push(0);
        stack.push(ids.len() - 1);
        stack.push(0);

        let mut result = vec![];

        // recursively search for items in range in the kd-sorted arrays
        while !stack.is_empty() {
            let axis = stack.pop().unwrap_or(0);
            let right = stack.pop().unwrap_or(0);
            let left = stack.pop().unwrap_or(0);

            // if we reached "tree node", search linearly
            if right - left <= node_size {
                for i in left..right + 1 {
                    let x = coords[2 * i];
                    let y = coords[2 * i + 1];
                    if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
                        result.push(ids.get(i));
                    }
                }
                continue;
            }

            // otherwise find the middle index
            let m = (left + right) >> 1;

            // include the middle item if it's in range
            let x = coords[2 * m];
            let y = coords[2 * m + 1];
            if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
                result.push(ids.get(m));
            }

            // queue search in halves that intersect the query
            let lte = if axis == 0 { min_x <= x } else { min_y <= y };
            if lte {
                stack.push(left);
                stack.push(m - 1);
                stack.push(1 - axis);
            }

            let gte = if axis == 0 { max_x >= x } else { max_y >= y };
            if gte {
                stack.push(m + 1);
                stack.push(right);
                stack.push(1 - axis);
            }
        }

        result
    }

    /// Search the index for items within a given radius.
    ///
    /// - qx: x value of query point
    /// - qy: y value of query point
    /// - r: radius
    ///
    /// Returns indices of found items
    fn within(&self, qx: f64, qy: f64, r: f64) -> Vec<usize> {
        let ids = self.ids();
        let coords = self.coords();
        let node_size = self.node_size();

        // Use arrayvec to avoid heap allocations
        let mut stack = ArrayVec::<_, 3>::new();
        stack.push(0);
        stack.push(ids.len() - 1);
        stack.push(0);

        let mut result = vec![];
        let r2 = r * r;

        // recursively search for items within radius in the kd-sorted arrays
        while !stack.is_empty() {
            let axis = stack.pop().unwrap_or(0);
            let right = stack.pop().unwrap_or(0);
            let left = stack.pop().unwrap_or(0);

            // if we reached "tree node", search linearly
            if right - left <= node_size {
                for i in left..right + 1 {
                    if sq_dist(coords[2 * i], coords[2 * i + 1], qx, qy) <= r2 {
                        result.push(ids.get(i));
                    }
                }
                continue;
            }

            // otherwise find the middle index
            let m = (left + right) >> 1;

            // include the middle item if it's in range
            let x = coords[2 * m];
            let y = coords[2 * m + 1];
            if sq_dist(x, y, qx, qy) <= r2 {
                result.push(ids.get(m));
            }

            // queue search in halves that intersect the query
            let lte = if axis == 0 { qx - r <= x } else { qy - r <= y };
            if lte {
                stack.push(left);
                stack.push(m - 1);
                stack.push(1 - axis);
            }

            let gte = if axis == 0 { qx + r >= x } else { qy + r >= y };
            if gte {
                stack.push(m + 1);
                stack.push(right);
                stack.push(1 - axis);
            }
        }
        result
    }
}

impl KdbushIndex for KdbushRef<'_> {
    fn num_items(&self) -> usize {
        self.num_items
    }

    fn node_size(&self) -> usize {
        self.node_size
    }

    fn coords(&self) -> &[f64] {
        self.coords
    }

    fn ids(&self) -> Cow<'_, Indices> {
        Cow::Borrowed(&self.ids)
    }
}

#[inline]
fn sq_dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}
