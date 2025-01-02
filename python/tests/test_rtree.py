import numpy as np
from arro3.core import list_flatten
from geoindex_rs import rtree as rt


def create_index():
    builder = rt.RTreeBuilder(5)
    min_x = np.arange(5)
    min_y = np.arange(5)
    max_x = np.arange(5, 10)
    max_y = np.arange(5, 10)
    builder.add(min_x, min_y, max_x, max_y)
    return builder.finish()


def test_search():
    tree = create_index()
    result = rt.search(tree, 0.5, 0.5, 1.5, 1.5)
    assert len(result) == 2
    assert result[0].as_py() == 0
    assert result[1].as_py() == 1


def test_rtree():
    builder = rt.RTreeBuilder(5)
    min_x = np.arange(5)
    min_y = np.arange(5)
    max_x = np.arange(5, 10)
    max_y = np.arange(5, 10)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish()

    boxes = rt.boxes_at_level(tree, 0)
    values = list_flatten(boxes)
    np_arr = np.asarray(values).reshape(-1, 4)
    assert np.all(min_x == np_arr[:, 0])
    assert np.all(min_y == np_arr[:, 1])
    assert np.all(max_x == np_arr[:, 2])
    assert np.all(max_y == np_arr[:, 3])


def test_partitions():
    builder = rt.RTreeBuilder(5, 2)
    min_x = np.arange(5)
    min_y = np.arange(5)
    max_x = np.arange(5, 10)
    max_y = np.arange(5, 10)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish()

    partitions = rt.partitions(tree)
    indices = partitions["indices"]
    partition_id = partitions["partition_id"]

    assert np.all(np.asarray(indices) == np.arange(5))
    assert len(np.unique(np.asarray(partition_id))) == 3
