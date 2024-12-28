import numpy as np
from geoindex_rs import rtree


def test_rtree():
    builder = rtree.RTreeBuilder(5)
    min_x = np.arange(5)
    min_y = np.arange(5)
    max_x = np.arange(5, 10)
    max_y = np.arange(5, 10)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish("hilbert")

    result = rtree.search(tree, 0.5, 0.5, 1.5, 1.5)
    assert len(result) == 2
    assert result[0].as_py() == 0
    assert result[1].as_py() == 1
