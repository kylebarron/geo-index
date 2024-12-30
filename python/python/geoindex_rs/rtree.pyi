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

ArrayLike = Union[np.ndarray, ArrowArrayExportable, memoryview, bytes]
IndexLike = Union[np.ndarray, ArrowArrayExportable, memoryview, bytes, RTree]

def search(
    index: IndexLike,
    min_x: int | float,
    min_y: int | float,
    max_x: int | float,
    max_y: int | float,
) -> Array: ...
def intersection_candidates(
    left: IndexLike,
    right: IndexLike,
) -> Array: ...

class RTreeMetadata:
    def __repr__(self) -> str: ...

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
    def num_items(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def node_size(self) -> int: ...
    @property
    def num_levels(self) -> int: ...
    @property
    def num_bytes(self) -> int: ...
    def boxes_at_level(self, level: int) -> Array:
        """

        This is shared as a zero-copy view from Rust. Note that it will keep the entire
        index memory alive until the returned array is garbage collected.

        """
