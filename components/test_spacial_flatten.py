"""Test spacial flatten."""
import tensorflow as tf
from . import spacial_flatten


class TestSpacialFlatten(tf.test.TestCase):
    """Unit test cases for spacial flatten layer."""

    def test_already_flat_input(self):
        """Does not affect already flat input."""
        tensor = tf.random.normal((4, 2, 4))
        res = spacial_flatten.SpacialFlatten()(tensor)
        self.assertEqual(res.shape, [4, 2, 4])

    def test_spacial_flatten(self):
        """Flattens spacial dimensions."""
        tensor = tf.random.normal((4, 4, 2, 4))
        res = spacial_flatten.SpacialFlatten()(tensor)
        self.assertEqual(res.shape, [4, 8, 4])
