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
IndexLike = Union[np.ndarray, ArrowArrayExportable, Buffer, RTree]

def boxes_at_level(index: IndexLike, level: int) -> Array:
    """

    /// Access the bounding boxes at the given level of the tree.
    ///
    /// The tree is laid out from bottom to top. Level 0 is the _base_ of the tree. Each integer
    /// higher is one level higher of the tree.

    This is shared as a zero-copy view from Rust. Note that it will keep the entire
    index memory alive until the returned array is garbage collected.

    """

def intersection_candidates(
    left: IndexLike,
    right: IndexLike,
) -> RecordBatch: ...
def neighbors(
    index: IndexLike,
    x: int | float,
    y: int | float,
    *,
    max_results: int | None,
    max_distance: int | float | None,
) -> Array: ...
def partitions(index: IndexLike) -> RecordBatch: ...
def partition_boxes(index: IndexLike) -> RecordBatch: ...
def search(
    index: IndexLike,
    min_x: int | float,
    min_y: int | float,
    max_x: int | float,
    max_y: int | float,
) -> Array: ...

class RTreeMetadata:
    def __repr__(self) -> str: ...
    @property
    def num_items(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def node_size(self) -> int: ...
    @property
    def level_bounds(self) -> int: ...
    @property
    def num_levels(self) -> int: ...
    @property
    def num_bytes(self) -> int: ...

class RTreeBuilder:
    def __init__(
        self,
        num_items: int,
        node_size: int = 16,
        coord_type: Literal["float32", "float64", None] = None,
    ) -> None: ...
    def add(
        self,
        min_x: ArrayLike,
        min_y: ArrayLike | None = None,
        max_x: ArrayLike | None = None,
        max_y: ArrayLike | None = None,
    ) -> Array: ...
    def finish(self, method: Literal["hilbert", "str", None] = None) -> RTree: ...
    def __repr__(self) -> str: ...

class RTree(Buffer):
    def __repr__(self) -> str: ...
    @property
    def metadata(self) -> RTreeMetadata: ...
