from typing import Literal, Optional, Self, Union

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

class RTree:
    @classmethod
    def from_interleaved(
        cls,
        boxes: NDArray[np.float64],
        *,
        method: RTreeMethod | RTreeMethodT = RTreeMethod.Hilbert,
        node_size: Optional[int] = None,
    ) -> Self: ...
    @classmethod
    def from_separated(
        cls,
        min_x: NDArray[np.float64],
        min_y: NDArray[np.float64],
        max_x: NDArray[np.float64],
        max_y: NDArray[np.float64],
        *,
        method: RTreeMethod | RTreeMethodT = RTreeMethod.Hilbert,
        node_size: Optional[int] = None,
    ) -> Self: ...
    def search(
        self, min_x: IntFloat, min_y: IntFloat, max_x: IntFloat, max_y: IntFloat
    ) -> NDArray[np.uintc]: ...
