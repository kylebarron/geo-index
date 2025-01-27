import numpy as np
from geoindex_rs import rtree as rt


def generate_random_boxes():
    random_points = np.random.random_sample((100, 2))
    boxes = np.concatenate([random_points, random_points], axis=1)
    return boxes


def test_buffer_protocol():
    boxes = generate_random_boxes()
    builder = rt.RTreeBuilder(len(boxes))
    builder.add(boxes)
    initial = builder.finish()
    # construct a memoryview transparently
    view = memoryview(initial)
    assert initial.metadata.num_bytes == view.nbytes
