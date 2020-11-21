"""Unit tests for relations game cnn layer."""
import tensorflow as tf

from . import relsgame_cnn


class TestRelationsGameCNN(tf.test.TestCase):
    """Unit test cases for relations game cnn."""

    def test_number_of_objects(self):
        """The relations game CNN returns 25 objects."""
        images = tf.random.uniform((2, 12, 12, 3), dtype=tf.float32)  # 2 random images
        res = relsgame_cnn.RelationsGameCNN()(images)  # (B, 25, 32)
        self.assertEqual(res.shape, [2, 25, 32])


class TestRelationsGamePixelCNN(tf.test.TestCase):
    """Unit test cases for relations game pixel cnn."""

    def test_number_of_pixels(self):
        """The relations game CNN returns 144 pixels."""
        images = tf.random.uniform((2, 12, 12, 3), dtype=tf.float32)  # 2 random images
        res = relsgame_cnn.RelationsGamePixelCNN(32)(images)  # (B, 25, 32)
        self.assertEqual(res.shape, [2, 144, 32])
