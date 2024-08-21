import numpy as np

from .. import RTree


def generate_random_boxes():
    random_points = np.random.random_sample((100, 2))
    boxes = np.concatenate([random_points, random_points], axis=1)
    return boxes


def test_buffer_protocol():
    boxes = generate_random_boxes()
    initial = RTree.from_interleaved(boxes)
    # construct a memoryview transparently
    view = memoryview(initial)
    assert initial.num_bytes == view.nbytes
