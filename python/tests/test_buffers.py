import numpy as np

from geoindex_rs import RTree


def generate_random_boxes():
    random_points = np.random.random_sample((100, 2))
    boxes = np.concatenate([random_points, random_points], axis=1)
    return boxes


def test_buffer_protocol():
    boxes = generate_random_boxes()
    initial = RTree.from_interleaved(boxes)
    second = RTree.from_buffer(initial)

    assert initial is not second, "Not the same object"

    # Flatbush magic byte
    assert memoryview(initial)[0] == 0xFB
    assert memoryview(second)[0] == 0xFB
