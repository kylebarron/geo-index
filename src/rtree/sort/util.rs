use crate::indices::MutableIndices;
use crate::IndexableNum;

/// Swap two values and two corresponding boxes.
#[inline]
pub(super) fn swap<V: IndexableNum, N: IndexableNum>(
    values: &mut [V],
    boxes: &mut [N],
    indices: &mut MutableIndices,
    i: usize,
    j: usize,
) {
    values.swap(i, j);

    let k = 4 * i;
    let m = 4 * j;
    boxes.swap(k, m);
    boxes.swap(k + 1, m + 1);
    boxes.swap(k + 2, m + 2);
    boxes.swap(k + 3, m + 3);

    indices.swap(i, j);
}
