/// calculate the total number of nodes in the R-tree to allocate space for
/// and the index of each tree level (used in search later)
pub fn compute_num_nodes(num_items: usize, node_size: usize) -> (usize, Vec<usize>) {
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
