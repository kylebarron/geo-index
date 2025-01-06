from __future__ import annotations

import sys
from typing import Literal, Union

import numpy as np
from arro3.core import Array
from arro3.core.types import ArrowArrayExportable

if sys.version_info > (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

ArrayLike = Union[np.ndarray, ArrowArrayExportable, Buffer]
"""A type alias for accepted input to the [`KDTreeBuilder.add`][geoindex_rs.kdtree.KDTreeBuilder.add] method.

Accepted input includes numpy arrays, Arrow arrays, and C-contiguous buffer protocol
input.
"""

IndexLike = Union[np.ndarray, ArrowArrayExportable, Buffer, KDTree]
"""A type alias for accepted input as a KDTree.
"""

def range(
    index: IndexLike,
    min_x: int | float,
    min_y: int | float,
    max_x: int | float,
    max_y: int | float,
) -> Array:
    """Search a KDTree for elements intersecting the provided bounding box.

    Results are the insertion indexes of items that match the query.

    **Example:**

    ```py
    import numpy as np
    from geoindex_rs import kdtree as kd

    builder = kd.KDTreeBuilder(3)
    x = np.arange(0, 3)
    y = np.arange(2, 5)
    builder.add(x, y)
    tree = builder.finish()

    results = kd.range(tree, 2, 4, 7, 9)
    assert results.to_pylist() == [2]
    ```

    Args:
        index: the KDTree to search.
        min_x: The `min_x` coordinate of the query bounding box.
        min_y: The `min_y` coordinate of the query bounding box.
        max_x: The `max_x` coordinate of the query bounding box.
        max_y: The `max_y` coordinate of the query bounding box.

    Returns:
        A uint32-typed Arrow array with the insertion indexes of query results.
    """

def within(
    index: IndexLike,
    qx: int | float,
    qy: int | float,
    r: int | float,
) -> Array:
    """Search a KDTree for elements within the given distance of the query point.

    Results are the insertion indexes of items that match the query.

    Args:
        index: the KDTree to search.
        qx: The `x` coordinate of the query point.
        qy: The `y` coordinate of the query point.
        r: The radius from the query point to use for searching.

    Returns:
        A uint32-typed Arrow array with the insertion indexes of query results.
    """

class KDTreeBuilder:
    """A builder class to create a [KDTree][geoindex_rs.kdtree.KDTree].

    **Example:**

    ```py
    import numpy as np
    from geoindex_rs import kdtree as kd

    builder = kd.KDTreeBuilder(3)
    x = np.arange(0, 3)
    y = np.arange(2, 5)
    builder.add(x, y)
    tree = builder.finish()

    results = kd.range(tree, 2, 4, 7, 9)
    assert results.to_pylist() == [2]
    ```
    """
    def __init__(
        self,
        num_items: int,
        node_size: int = 64,
        coord_type: Literal["float32", "float64"] = "float64",
    ) -> None:
        """Initialize a KDTree with the given parameters.

        This will provision memory for the KDTree, to be filled in the [`add`][geoindex_rs.kdtree.KDTreeBuilder.add] method.

        Args:
            num_items: The number of items in the tree
            node_size: The node size of the tree. Defaults to 64.
            coord_type: The coordinate type to use in the tree. Currently only float32
                and float64 are permitted. Defaults to "float64".
        """
    def add(self, x: ArrayLike, y: ArrayLike | None = None) -> Array:
        """Insert points in this KDTree.

        There are multiple ways to call this method:

        - Two numpy or Arrow numeric arrays, passed as `x`, `y`.
        - A 2D numpy array, with shape `(N, 2)`. It must be C-contiguous.
        - One argument with an Arrow array of type FixedSizeList or Struct. The
          FixedSizeListArray must have a `list_size` of 2, and the `StructArray` must
          have 2 children, ordered `x` and `y`.

          In this case, all other parameters should be left as `None`.


        !!! note
            Most of the time, you should call `add` only once. This function is
            vectorized, and you should avoid calling it in a loop with a few rows of
            data at a time.

            In some cases, it may be useful to call this in a loop, such as if the input
            geometries do not all fit in memory at once.

        It's important to add _arrays_ at a time. This should usually not be called in a loop.

        Args:
            x: array-like input
            y: array-like input. Defaults to None.

        Returns:
            An Arrow array with the insertion index of each element, which provides a lookup back into the original data.

                This can be converted to a [`pyarrow.Array`][] by passing to
                [`pyarrow.array`][].
        """
    def finish(self) -> KDTree:
        """Sort the internal index and convert this class to a KDTree instance.

        Returns:
            An immutable KDTree instance, which can be used for spatial queries.
        """
    def __repr__(self) -> str: ...

class KDTree(Buffer):
    """A fast, memory-efficient, zero-copy-compatible, 2D KDTree.

    Use [`KDTreeBuilder`][geoindex_rs.kdtree.KDTreeBuilder], and then call
    [`finish`][geoindex_rs.kdtree.KDTreeBuilder.finish] to construct this.

    This uses a binary-stable memory layout under the hood. So you can write the index
    to disk, then later load it and perform queries on it.

    This class implements the Python buffer protocol, so you can pass it to the Python
    `bytes` constructor to copy the underlying binary memory into a Python `bytes`
    object.
    """
    def __repr__(self) -> str: ...

class KDTreeMetadata:
    """Common metadata to describe a KDTree.

    This can be used to know the number of items, node information, or total byte size
    of a KDTree.

    Additionally, this can be used to know how much memory a KDTree **would use** with
    the given number of items and node size. A KDTree with 1 million items and a node
    size of 64 (the default) would take up 20 MiB.

    ```py
    from geoindex_rs import kdtree as kd

    metadata = kd.KDTreeMetadata(num_items=1_000_000, node_size=64)
    assert metadata.num_bytes == 20_000_008
    ```
    """

    def __init__(
        self,
        num_items: int,
        node_size: int = 64,
        coord_type: Literal["float32", "float64"] = "float64",
    ) -> None:
        """Create a new KDTreeMetadata given a number of items and node size.

        Args:
            num_items: The number of items in the tree
            node_size: The node size of the tree. Defaults to 16.
            coord_type: The coordinate type to use in the tree. Currently only float32
                and float64 are permitted. Defaults to None.
        """
    @classmethod
    def from_index(cls, index: IndexLike) -> KDTreeMetadata:
        """Create from an existing KDTree buffer."""
    def __repr__(self) -> str: ...
    @property
    def num_items(self) -> int:
        """The number of items indexed in the tree."""
    @property
    def node_size(self) -> int:
        """The maximum number of items per node."""
    @property
    def num_bytes(self) -> int:
        """The number of bytes that a KDTree with this metadata would have."""
