import numpy as np
from geoindex_rs import kdtree


def test_kdtree():
    builder = kdtree.KDTreeBuilder(5)
    x = np.arange(5)
    y = np.arange(5)
    builder.add(x, y)
    tree = builder.finish()

    result = kdtree.range(tree, 0.5, 0.5, 1.5, 1.5)
    assert len(result) == 1
    assert result[0].as_py() == 1
