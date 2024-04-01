from typing import Generic, Literal, Optional, Self, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from .enums import RTreeMethod

IntFloat = Union[int, float]
RTreeMethodT = Literal["hilbert", "str"]

class KDTree:
    @classmethod
    def from_interleaved(
        cls,
        coords: NDArray[np.float64],
        *,
        node_size: Optional[int] = None,
    ) -> Self: ...
    @classmethod
    def from_separated(
        cls,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        node_size: Optional[int] = None,
    ) -> Self: ...
    def range(
        self, min_x: IntFloat, min_y: IntFloat, max_x: IntFloat, max_y: IntFloat
    ) -> NDArray[np.uintc]: ...
    def within(self, qx: IntFloat, qy: IntFloat, r: IntFloat) -> NDArray[np.uintc]: ...

# https://stackoverflow.com/a/74634650
T = TypeVar("T", bound=np.generic, covariant=True)

class RTree(Generic[T]):
    @classmethod
    def from_interleaved(
        cls,
        boxes: NDArray[T],
        *,
        method: RTreeMethod | RTreeMethodT = RTreeMethod.Hilbert,
        node_size: Optional[int] = None,
    ) -> Self: ...
    @classmethod
    def from_separated(
        cls,
        min_x: NDArray[T],
        min_y: NDArray[T],
        max_x: NDArray[T],
        max_y: NDArray[T],
        *,
        method: RTreeMethod | RTreeMethodT = RTreeMethod.Hilbert,
        node_size: Optional[int] = None,
    ) -> Self: ...
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
    def boxes_at_level(self, level: int) -> NDArray[T]: ...
    def search(
        self, min_x: IntFloat, min_y: IntFloat, max_x: IntFloat, max_y: IntFloat
    ) -> NDArray[np.uintc]: ...
