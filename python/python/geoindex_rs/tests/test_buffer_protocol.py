import sys
import pytest
import numpy as np
from .. import RTree

def generate_random_boxes():
    random_points = np.random.random_sample((100, 2))
    boxes = np.concatenate([random_points, random_points], axis=1)
    return boxes

def test_garbage_buffer():
    garbage = b"Hello world"
    with pytest.raises(TypeError, match="Data not in Flatbush format") as expected_error:
        tree_instance = RTree.from_buffer(garbage)

def assert_equal_rtrees(left: RTree, right: RTree):
    assert left.num_levels == right.num_levels
    for level in range(left.num_levels):
        np.testing.assert_array_equal(
            left.boxes_at_level(level), right.boxes_at_level(level)
        )

def test_roundtrip():
    boxes = generate_random_boxes()
    initial = RTree.from_interleaved(boxes)
    # Get it out via the copying buffer interface
    output = initial.to_buffer()
    roundtripped = RTree.from_buffer(output)
    assert_equal_rtrees(initial, roundtripped)

@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires python 3.11")
def test_buffer_protocol():
    boxes = generate_random_boxes()
    initial = RTree.from_interleaved(boxes)
    print(initial)
    # construct a memoryview transparently
    view = memoryview(initial)
    # del view
    assert initial.num_bytes == view.nbytes
    tree_from_view = RTree.from_buffer(view)
    # TODO: use something like pytest-memray to prove zero-copy behaviour (where expected)
    assert_equal_rtrees(initial, tree_from_view)