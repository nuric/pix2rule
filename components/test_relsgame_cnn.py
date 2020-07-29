"""Unit tests for relations game cnn layer."""
import tensorflow as tf

from .relsgame_cnn import RelationsGameCNN


class TestRelationsGameCNN(tf.test.TestCase):
    """Unit test cases for relations game cnn."""

    def test_number_of_objects(self):
        """The relations game CNN returns 25 objects."""
        images = tf.random.uniform((2, 36, 36, 3))  # 2 random images
        res = RelationsGameCNN()(images)  # (B, 5, 5, 32)
        self.assertEqual(res.shape, [2, 25, 32])
