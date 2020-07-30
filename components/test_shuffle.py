"""Unit test cases for the shuffle layer."""
import tensorflow as tf

from .shuffle import Shuffle


class TestShuffle(tf.test.TestCase):
    """Test unit cases for the shuffle layer."""

    def test_shuffle_batch_axis(self):
        """It can shuffle the batch axis."""
        inputs = tf.constant([[1, 2], [3, 4]])  # (2, 2)
        tf.random.set_seed(42)
        res = Shuffle(shuffle_axis=0, seed=42)(inputs)
        expected = tf.constant([[3, 4], [1, 2]])
        self.assertAllEqual(res, expected)

    def test_shuffle_first_axis(self):
        """It can shuffle the first axis."""
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        tf.random.set_seed(42)
        res = Shuffle(shuffle_axis=1, seed=41)(inputs)
        expected = tf.constant([[2, 3, 1], [5, 6, 4]])
        self.assertAllEqual(res, expected)
