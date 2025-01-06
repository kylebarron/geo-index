from __future__ import annotations

import sys
from typing import Literal, Union

import numpy as np
from arro3.core import Array, RecordBatch
from arro3.core.types import ArrowArrayExportable

if sys.version_info > (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

ArrayLike = Union[np.ndarray, ArrowArrayExportable, Buffer]
"""A type alias for accepted input to the [`RTreeBuilder.add`][geoindex_rs.rtree.RTreeBuilder.add] method.

Accepted input includes numpy arrays, Arrow arrays, and C-contiguous buffer protocol
input.
"""

IndexLike = Union[np.ndarray, ArrowArrayExportable, Buffer, RTree]
"""A type alias for accepted input as an RTree.
"""

def boxes_at_level(index: IndexLike, level: int, *, copy: bool = False) -> Array:
    """Access the raw bounding box data contained in the RTree at a given tree level.

    Args:
        index: the RTree to search.
        level: The level of the tree to read from. Level 0 is the _base_ of the tree. Each integer higher is one level higher of the tree.

    Other Args:
        copy: if True, make a _copy_ of the data from the underlying RTree instead of
            viewing it directly. Making a copy can be preferred if you'd like to delete
            the index itself to save memory.

    Returns:
        An Arrow FixedSizeListArray containing the bounding box coordinates.

            If `copy` is `False`, the returned array is a a zero-copy view from Rust.
            Note that it will keep the entire index memory alive until the returned
            array is garbage collected.
    """

def tree_join(
    left: IndexLike,
    right: IndexLike,
) -> RecordBatch:
    """Find the overlapping elements of two RTrees.

    This is the first part of a spatial join: to find which elements from two different
    data sources potentially intersect.

    Note that this only evaluates intersection of the **bounding boxes** in the tree.
    Assuming that these bounding boxes represent actual vector geometries, the result of
    this join are _candidates_. Any pairs present implies that the bounding boxes of
    those two indices intersect. Any pairs not present implies that the bounding boxes
    of those two indices **cannot** intersect, and therefore the represented geometries
    cannot intersect.

    This returns an Arrow `RecordBatch` with positional indexes from the left and right
    trees. The `RecordBatch` has two `uint32` columns named `"left"` and `"right"`.

    Args:
        left: The "left" spatial index for the join.
        right: The "right" spatial index for the join.

    Returns:
        An Arrow `RecordBatch` with the positional indexes of intersecting tree
            elements.
    """

def neighbors(
    index: IndexLike,
    x: int | float,
    y: int | float,
    *,
    max_results: int | None,
    max_distance: int | float | None,
) -> Array:
    """Search items in order of distance from the given point.

    **Example:**

    ```py
    import numpy as np
    from geoindex_rs import rtree as rt

    builder = rt.RTreeBuilder(3)
    min_x = np.arange(0, 3)
    min_y = np.arange(0, 3)
    max_x = np.arange(2, 5)
    max_y = np.arange(2, 5)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish()

    results = rt.neighbors(tree, 5, 5)
    assert results.to_pylist() == [2, 1, 0]
    ```

    Args:
        index: the RTree to search.
        x: the `x` coordinate of the query point
        y: the `y` coordinate of the query point
        max_results: The maximum number of results to search. If not provided, all
            results (within `max_distance`) will be returned.
        max_distance: The maximum distance from the query point to search. If not
            provided, all results (up to `max_results`) will be returned.

    Returns:
        An Arrow array with the insertion indexes of query results.
    """

def partitions(index: IndexLike, *, copy=False) -> RecordBatch:
    """Extract the spatial partitions from an RTree.

    This can be used to find the sorted groups for spatially partitioning the original
    data, such as when writing spatially-partitioned
    [GeoParquet](https://geoparquet.org/).

    !!! note
        Currently, this always uses the lowest level of the tree when inferring
        partitioning. Thus, for large input you may want to use the largest node size
        possible (`65535`) for constructing a tree for use in spatial partitioning.

        Future work may allow higher levels of the tree to be used for partitioning.

    **Example:**

    ```py
    import numpy as np
    from geoindex_rs import rtree as rt

    # Create a new builder with space for 100 million boxes
    num_items = 100_000_000
    builder = rt.RTreeBuilder(num_items, 65535)
    min_x = np.random.uniform(-100, -10, num_items)
    min_y = np.random.uniform(-50, -10, num_items)
    max_x = np.random.uniform(10, 100, num_items)
    max_y = np.random.uniform(10, 50, num_items)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish()

    partitions = rt.partitions(tree)
    # There are 1526 partitioned groups
    assert partitions["partition_id"][-1].as_py() == 1525
    ```

    Args:
        index: the RTree to use.

    Other Args:
        copy: if True, make a _copy_ of the data from the underlying RTree instead of
            viewing it directly. Making a copy can be preferred if you'd like to delete
            the index itself to save memory.

    Returns:
        An Arrow `RecordBatch` with two columns: `indices` and `partition_ids`. `indices` refers to the insertion index of each row and `partition_ids` refers to the partition each row belongs to.

            If `copy` is `False`, the `indices` column is constructed as a zero-copy
            view on the provided index. Therefore, the `indices` array will have type
            `uint16` if the tree has fewer than 16,384 items; otherwise it will have
            type `uint32`.
    """

def partition_boxes(index: IndexLike, *, copy: bool = False) -> RecordBatch:
    """Extract the geometries of the spatial partitions from an RTree.

    In order for these boxes to be zero-copy from Rust, they are returned as a
    FixedSizeListArray, where each element has 4 items.

    !!! note
        While `partitions` returns a `RecordBatch` that has a length of the number of
        items, this returns a `RecordBatch` with a length matching the _number of
        partitions_.

    This is equivalent to calling `boxes_at_level(1)`, plus the `partition_id` column,
    which is a monotonically-increasing integer column.

    Args:
        index: the RTree to use.

    Other Args:
        copy: if True, make a _copy_ of the data from the underlying RTree instead of
            viewing it directly. Making a copy can be preferred if you'd like to delete
            the index itself to save memory.

    Returns:
        An Arrow `RecordBatch` with two columns: `boxes` and `partition_ids`. `boxes` stores the box geometry of each partition and `partition_ids` refers to the partition each row belongs to.

            If `copy` is `False`, the `boxes` column is constructed as a zero-copy view
            on the internal boxes data. The `partition_id` column will be `uint16` type
            if there are less than 65,536 partitions; otherwise it will be `uint32`
            type.
    """

def search(
    index: IndexLike,
    min_x: int | float,
    min_y: int | float,
    max_x: int | float,
    max_y: int | float,
) -> Array:
    """Search an RTree for elements intersecting the provided bounding box.

    Results are the insertion indexes of items that match the query.

    Args:
        index: the RTree to search
        min_x: The `min_x` coordinate of the query bounding box
        min_y: The `min_y` coordinate of the query bounding box
        max_x: The `max_x` coordinate of the query bounding box
        max_y: The `max_y` coordinate of the query bounding box

    Returns:
        An Arrow array with the insertion indexes of query results.
    """

class RTreeMetadata:
    """Common metadata to describe an RTree.

    This can be used to know the number of items, node information, or total byte size
    of an RTree.

    Additionally, this can be used to know how much memory an RTree **would use** with
    the given number of items and node size. An RTree with 1 million items and a node
    size of 20 would take up 38 MiB.

    ```py
    from geoindex_rs import rtree as rt

    metadata = rt.RTreeMetadata(num_items=1_000_000, node_size=20)
    assert metadata.num_bytes == 37_894_796
    ```
    """

    def __init__(
        self,
        num_items: int,
        node_size: int = 16,
        coord_type: Literal["float32", "float64"] = "float64",
    ) -> None:
        """Create a new RTreeMetadata given a number of items and node size.

        Args:
            num_items: The number of items in the tree
            node_size: The node size of the tree. Defaults to 16.
            coord_type: The coordinate type to use in the tree. Currently only float32
                and float64 are permitted. Defaults to None.
        """
    @classmethod
    def from_index(cls, index: IndexLike) -> RTreeMetadata:
        """Create from an existing RTree buffer."""
    def __repr__(self) -> str: ...
    @property
    def num_items(self) -> int:
        """The number of items indexed in the tree."""
    @property
    def num_nodes(self) -> int:
        """The total number of nodes at all levels in the tree."""
    @property
    def node_size(self) -> int:
        """The maximum number of items per node."""
    # @property
    # def level_bounds(self) -> int: ...
    @property
    def num_levels(self) -> int:
        """The number of levels in the tree."""
    @property
    def num_bytes(self) -> int:
        """The number of bytes that an RTree with this metadata would have."""

class RTreeBuilder:
    """A builder class to create an [RTree][geoindex_rs.rtree.RTree].

    **Example:**

    ```py
    import numpy as np
    from geoindex_rs import rtree as rt

    builder = rt.RTreeBuilder(3)
    min_x = np.arange(0, 3)
    min_y = np.arange(0, 3)
    max_x = np.arange(2, 5)
    max_y = np.arange(2, 5)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish()
    ```
    """
    def __init__(
        self,
        num_items: int,
        node_size: int = 16,
        coord_type: Literal["float32", "float64"] = "float64",
    ) -> None:
        """Initialize an RTree with the given parameters.

        This will provision memory for the RTree, to be filled in the [`add`][geoindex_rs.rtree.RTreeBuilder.add] method.

        Args:
            num_items: The number of items in the tree
            node_size: The node size of the tree. Defaults to 16.
            coord_type: The coordinate type to use in the tree. Currently only float32
                and float64 are permitted. Defaults to None.
        """
    def add(
        self,
        min_x: ArrayLike,
        min_y: ArrayLike | None = None,
        max_x: ArrayLike | None = None,
        max_y: ArrayLike | None = None,
    ) -> Array:
        """Insert bounding boxes in this RTree.

        There are multiple ways to call this method:

        - Four numpy or Arrow numeric arrays, passed as `min_x`, `min_y`, `max_x`, `max_y`.
        - A 2D numpy array, with shape `(N, 4)`. It must be C-contiguous.
        - One argument with an Arrow array of type FixedSizeList or Struct. The
          FixedSizeListArray must have a `list_size` of 4, and the `StructArray` must
          have four children, ordered `min_x`, `min_y`, `max_x`, `max_y`.

          In this case, all other parameters should be left as `None`.

        **Example: Adding shapely geometries**

        ```py
        import numpy as np
        import shapely
        from geoindex_rs import rtree as rt

        # Shapely array of geometries, such as from GeoPandas
        geometries = [...]

        builder = rt.RTreeBuilder(len(geometries))

        # Find the bounding box of each geometry
        bounds = shapely.bounds(geometries)
        builder.add(bounds)
        tree = builder.finish()

        results = rt.neighbors(tree, 5, 5)
        assert results.to_pylist() == [2, 1, 0]
        ```

        !!! note
            Most of the time, you should call `add` only once. This function is
            vectorized, and you should avoid calling it in a loop with a few rows of
            data at a time.

            In some cases, it may be useful to call this in a loop, such as if the input
            geometries do not all fit in memory at once.

        It's important to add _arrays_ at a time. This should usually not be called in a loop.

        Args:
            min_x: array-like input. If this is the only provided input, it should
                represent the entire bounding box, as described above. Otherwise, pass
                four separate parameters.
            min_y: array-like input. Defaults to None.
            max_x: array-like input. Defaults to None.
            max_y: array-like input. Defaults to None.

        Returns:
            An Arrow array with the insertion index of each element, which provides a lookup back into the original data.

                This can be converted to a [`pyarrow.Array`][] by passing to
                [`pyarrow.array`][].
        """
    def finish(self, method: Literal["hilbert", "str"] = "hilbert") -> RTree:
        """Sort the internal index and convert this class to an RTree instance.

        Args:
            method: The method used for sorting the RTree. Defaults to `"hilbert"`.

                - `"hilbert"` will use a [Hilbert Curve](https://en.wikipedia.org/wiki/Hilbert_R-tree#Packed_Hilbert_R-trees) for sorting.
                - `"str"` will use the [Sort-Tile-Recursive](https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf) algorithm.

        Returns:
            An immutable RTree instance, which can be used for spatial queries.
        """
    def __repr__(self) -> str: ...

class RTree(Buffer):
    """A fast, memory-efficient, zero-copy-compatible, 2D RTree.

    Use [`RTreeBuilder`][geoindex_rs.rtree.RTreeBuilder], and then call
    [`finish`][geoindex_rs.rtree.RTreeBuilder.finish] to construct this.

    This uses a binary-stable memory layout under the hood. So you can write the index
    to disk, then later load it and perform queries on it.

    This class implements the Python buffer protocol, so you can pass it to the Python
    `bytes` constructor to copy the underlying binary memory into a Python `bytes`
    object.

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
    protocol) directly in searches
    results = rt.neighbors(copied_tree, 5, 5)
    assert results.to_pylist() == [2, 1, 0]
    ```
    """
    def __repr__(self) -> str: ...
    @property
    def metadata(self) -> RTreeMetadata:
        """Access the metadata instance of this RTree.

        Use this to infer the number of items or nodes in this RTree.
        """
