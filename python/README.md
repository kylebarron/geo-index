# geoindex-rs

Fast, memory-efficient 2D spatial indexes for Python.

## API

### `KDTree`

#### `KDTree.from_interleaved`

Construct an KDTree from a 2D numpy array of `x` and `y`. This must have two dimensions, with the second dimension having length two.

#### `KDTree.from_separated`

Construct an KDTree from two separated 1D numpy arrays of `x` and `y`. Each array must have one dimension and both arrays must have the same length.

#### `KDTree.range`

Search the index for items within a given bounding box.

Arguments:

- `min_x`: float
- `min_y`: float
- `max_x`: float
- `max_y`: float

Returns indices of found items

#### `KDTree.within`

Search the index for items within a given radius.

- `qx` (`float`): x value of query point
- `qy` (`float`): y value of query point
- `r` (`float`): radius

Returns indices of found items

### `RTree`

#### `RTree.from_interleaved`

Construct an RTree from a 2D numpy array of `minx`, `miny`, `maxx`, `maxy`. This must have two dimensions, with the second dimension having length four.

For example, the output of `shapely.bounds` is in this format.

```py
import numpy as np
from geoindex_rs import RTree

geometries = shapely.polygons(...)
bounds = shapely.bounds(geometries)
tree = RTree.from_separated(bounds)
```

#### `RTree.from_separated`

Construct an RTree from four separate 1D numpy arrays of `minx`, `miny`, `maxx`, `maxy`. Each array must have one dimension and all arrays must have the same length.

```py
import numpy as np
from geoindex_rs import RTree

minx = np.array([-10, ...], dtype=np.float64)
miny = np.array([-20, ...], dtype=np.float64)
maxx = np.array([10, ...], dtype=np.float64)
maxy = np.array([20, ...], dtype=np.float64)
tree = RTree.from_separated(minx, miny, maxx, maxy)
```

#### `RTree.search`

Search within an RTree for the given bounding box. Returns the indices of the input array that match the output.

```py
tree = RTree.from_separated(...)
tree.search(minx, miny, maxx, maxy)
```
