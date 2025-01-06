# geoindex-rs

[![PyPI][pypi_badge]][pypi_link]

[pypi_badge]: https://badge.fury.io/py/geoindex-rs.svg
[pypi_link]: https://pypi.org/project/geoindex-rs/

Fast, memory-efficient, zero-copy spatial indexes for Python.

This documentation is for the Python bindings. [Refer here for the Rust crate documentation](https://docs.rs/geo-index).

## Features

- **An R-tree and k-d tree written in Rust, compiled for Python.**
- **Fast.** The Rust core and immutability lends the spatial indexes to be very fast. Additionally, building the indexes accepts vectorized Numpy or [Arrow](https://arrow.apache.org/) input.
- **Memory efficient.** The index is fully _packed_, meaning that all nodes are at full capacity (except for the last node at each tree level). This means the RTree and k-d tree use less memory.
- **Bounded memory**. For any given number of items and node size, you can infer the total memory used by the RTree or KDTree.
- **Multiple R-tree sorting methods.** Currently, [hilbert](https://en.wikipedia.org/wiki/Hilbert_R-tree) and [sort-tile-recursive (STR)](https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf) sorting methods are implemented.
- **ABI-stable:** the index is contained in a single buffer, compatible with the [`flatbush`](https://github.com/mourner/flatbush) and [`kdbush`](https://github.com/mourner/kdbush) JavaScript libraries. Being ABI-stable means that the spatial index can be persisted for later use or shared zero-copy between Rust and Python.
- **Supports float64 or float32 coordinates:** for 2x memory savings, use float32 coordinates in the spatial index.

## Install

```
pip install geoindex-rs
```

or with Conda:

```
conda install geoindex-rs
```

## Examples

Building an RTree and searching for nearest neighbors.

```py
import numpy as np
from geoindex_rs import rtree as rt

# Three bounding boxes
min_x = np.arange(0, 3)
min_y = np.arange(0, 3)
max_x = np.arange(2, 5)
max_y = np.arange(2, 5)

# When creating the builder, the total number of items must be passed
builder = rt.RTreeBuilder(num_items=3)

# Add the bounding boxes to the builder
builder.add(min_x, min_y, max_x, max_y)

# Consume the builder (sorting the index) and create the RTree.
tree = builder.finish()

# Find the nearest neighbors in the RTree to the point (5, 5)
results = rt.neighbors(tree, 5, 5)

# For performance, results are returned as an Arrow array.
assert results.to_pylist() == [2, 1, 0]
```

Building a KDTree and searching within a bounding box.

```py
import numpy as np
from geoindex_rs import kdtree as kd

# Three points: (0, 2), (1, 3), (2, 4)
x = np.arange(0, 3)
y = np.arange(2, 5)

# When creating the builder, the total number of items must be passed
builder = kd.KDTreeBuilder(3)

# Add the points to the builder
builder.add(x, y)

# Consume the builder (sorting the index) and create the KDTree.
tree = builder.finish()

# Search within this bounding box:
results = kd.range(tree, 2, 4, 7, 9)

# For performance, results are returned as an Arrow array.
assert results.to_pylist() == [2]
```

## Persisting the spatial index

The `RTree` and `KDTree` classes implement the Python buffer protocol, so you
can pass an instance of the index directly to `bytes` to copy the underlying
spatial index into a buffer. Then you can save that buffer somewhere, load it
again, and use it directly for queries!

```py
import numpy as np
from geoindex_rs import rtree as rt

min_x = np.arange(0, 3)
min_y = np.arange(0, 3)
max_x = np.arange(2, 5)
max_y = np.arange(2, 5)

builder = rt.RTreeBuilder(num_items=3)
builder.add(min_x, min_y, max_x, max_y)
tree = builder.finish()

# Copy to a Python bytes object
copied_tree = bytes(tree)

# The entire RTree is contained within this 144 byte buffer
assert len(copied_tree) == 144

# We can use the bytes object (or anything else implementing the Python buffer
# protocol) directly in searches
results = rt.neighbors(copied_tree, 5, 5)
assert results.to_pylist() == [2, 1, 0]
```

## Drawbacks

- Trees are _immutable_. After creating the index, items can no longer be added or removed.
- Only two-dimensional indexes is supported. This can still be used with higher-dimensional input data as long as it's fine to only index two of the dimensions.
- Queries return insertion indexes into the input set, so you must manage your own collections.
