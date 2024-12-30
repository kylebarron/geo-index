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
IndexLike = Union[np.ndarray, ArrowArrayExportable, memoryview, bytes, KDTree]

def range(
    index: IndexLike,
    min_x: int | float,
    min_y: int | float,
    max_x: int | float,
    max_y: int | float,
) -> Array: ...
def within(
    index: IndexLike,
    qx: int | float,
    qy: int | float,
    r: int | float,
) -> Array: ...

class KDTreeBuilder:
    def __init__(
        self,
        num_items: int,
        node_size: int = 64,
        coord_type: Literal["float32", "float64", None] = None,
    ) -> None: ...
    def add(self, x: ArrayLike, y: ArrayLike | None = None) -> Array: ...
    def finish(self) -> KDTree: ...

class KDTree(Buffer): ...
